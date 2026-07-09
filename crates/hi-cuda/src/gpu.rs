use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct CudaQwenGpuModelInfo {
    pub tensor_count: usize,
    pub total_device_bytes: usize,
    pub largest_tensor_bytes: usize,
    pub dtype_summary: Vec<String>,
    pub matrix_count: usize,
    pub total_matrix_bytes: usize,
    pub quantized_matrix_count: usize,
    pub vector_count: usize,
    pub total_vector_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CudaMmprojProjectorInfo {
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub layout: String,
    pub layer_count: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dim: Option<usize>,
    pub total_device_bytes: usize,
    pub bias_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CudaVisionEncoderInfo {
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub architecture: String,
    pub variant: String,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub tokens_per_second: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub hidden_dim: usize,
    pub feed_forward_dim: usize,
    pub output_dim: usize,
    pub block_count: usize,
    pub head_count: usize,
    pub total_device_bytes: usize,
    pub matrix_count: usize,
    pub vector_count: usize,
    pub uses_window_attention: bool,
}

#[cfg(feature = "native-cuda")]
mod native {
    use std::cell::{Cell, RefCell, RefMut};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt;
    use std::time::Instant;

    use anyhow::{Context, Result, anyhow, bail};
    use hi_gguf::{
        GgufFile, GgufTensorType, QwenGgufConfig, dequantize_tensor_as_f32,
        qwen_dense_attention_bias_names, qwen_dense_attention_head_norm_weight_names,
        qwen_dense_attention_norm_weight_names, qwen_dense_attention_weight_names,
        qwen_dense_ffn_bias_names, qwen_dense_ffn_norm_weight_names, qwen_dense_ffn_weight_names,
        qwen_dense_gated_attention_q_bias_name, qwen_dense_gated_attention_q_weight_name,
        qwen_dense_output_bias_names, qwen_dense_output_norm_weight_names,
        qwen_dense_output_weight_names, qwen_dense_packed_ffn_gate_up_bias_names,
        qwen_dense_packed_ffn_gate_up_weight_names, qwen_dense_packed_ffn_up_gate_bias_names,
        qwen_dense_packed_ffn_up_gate_weight_names, qwen_dense_packed_qkv_bias_names,
        qwen_dense_packed_qkv_weight_names, qwen_dense_token_embd_weight_names,
        qwen_mla_attention_tensors_present, qwen_mla_kv_a_norm_weight_names,
        qwen_mla_kv_a_weight_names, qwen_mla_kv_b_weight_names, qwen_mla_q_a_norm_weight_names,
        qwen_mla_q_a_weight_names, qwen_mla_q_b_weight_names, qwen_moe_packed_expert_bias_names,
        qwen_moe_packed_expert_gate_up_bias_names, qwen_moe_packed_expert_gate_up_weight_names,
        qwen_moe_packed_expert_up_gate_bias_names, qwen_moe_packed_expert_up_gate_weight_names,
        qwen_moe_packed_expert_weight_names, qwen_moe_per_expert_bias_names,
        qwen_moe_per_expert_gate_up_bias_names, qwen_moe_per_expert_gate_up_weight_names,
        qwen_moe_per_expert_up_gate_bias_names, qwen_moe_per_expert_up_gate_weight_names,
        qwen_moe_per_expert_weight_names, qwen_moe_router_bias_names, qwen_moe_router_weight_names,
        qwen_moe_shared_expert_bias_names, qwen_moe_shared_expert_gate_bias_names,
        qwen_moe_shared_expert_gate_up_bias_names, qwen_moe_shared_expert_gate_up_weight_names,
        qwen_moe_shared_expert_gate_weight_names, qwen_moe_shared_expert_up_gate_bias_names,
        qwen_moe_shared_expert_up_gate_weight_names, qwen_moe_shared_expert_weight_names,
        qwen_ssm_a_names, qwen_ssm_ba_weight_names, qwen_ssm_conv1d_weight_names,
        qwen_ssm_dt_bias_names, qwen_ssm_gate_weight_names, qwen_ssm_in_weight_names,
        qwen_ssm_layer_tensors_present, qwen_ssm_norm_weight_names, qwen_ssm_out_weight_names,
        qwen_ssm_qkv_weight_names,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::runtime::{Cublas, CublasLt, CudaRuntime, DeviceBuffer, GemmDType, Stream};

    use super::{CudaMmprojProjectorInfo, CudaQwenGpuModelInfo, CudaVisionEncoderInfo};

    const FLASH_ONLINE_MAX_HEAD_DIM: usize = 512;
    // Flash-attention (shared-memory K/V tiling) prefill path is used for causal
    // (no sliding window) attention up to this head_dim; larger falls back to the
    // per-query tiled kernel. Must match HI_CUDA_FLASH_TILE_MAX_HEAD_DIM in kernels.cu.
    const FLASH_TILE_MAX_HEAD_DIM: usize = 128;

    fn retain_uncancelled_batch_rows<F>(active: &mut [bool], is_cancelled: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let mut any_active = false;
        for (idx, active_row) in active.iter_mut().enumerate() {
            if *active_row && is_cancelled(idx) {
                *active_row = false;
            }
            any_active |= *active_row;
        }
        any_active
    }

    fn per_request_sample_rngs(seeds: &[Option<u64>]) -> Vec<StdRng> {
        seeds
            .iter()
            .map(|seed| StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::random::<u64>())))
            .collect()
    }

    fn sampled_selection_needs_host_rank(temperature: f32, top_p: f32, top_k: Option<u32>) -> bool {
        temperature.is_finite()
            && temperature > 0.0
            && ((top_p.is_finite() && top_p < 1.0) || top_k.is_some_and(|value| value > 0))
    }

    fn argmax_host(logits: &[f32]) -> Result<u32> {
        let (idx, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|(left_id, left), (right_id, right)| {
                left.total_cmp(right).then_with(|| right_id.cmp(left_id))
            })
            .ok_or_else(|| anyhow!("cannot sample from empty logits"))?;
        u32::try_from(idx).context("sampled token index does not fit u32")
    }

    fn sample_host_ranked_logits_with_uniform(
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        sample: f32,
    ) -> Result<u32> {
        if logits.is_empty() {
            bail!("cannot sample from empty logits");
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return argmax_host(logits);
        }

        let mut scaled = logits
            .iter()
            .copied()
            .map(|logit| {
                if logit.is_finite() {
                    logit / temperature
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<_>>();
        let max = scaled
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
        if !max.is_finite() {
            return argmax_host(logits);
        }
        for value in &mut scaled {
            *value = (*value - max).exp();
        }

        let mut ranked = scaled.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(left_id, left), (right_id, right)| {
            right.total_cmp(left).then_with(|| left_id.cmp(right_id))
        });
        if let Some(top_k) = top_k.and_then(|value| usize::try_from(value).ok()) {
            if top_k > 0 {
                ranked.truncate(top_k.min(ranked.len()));
            }
        }

        let cutoff = if top_p.is_finite() {
            top_p.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let total = ranked.iter().map(|(_, weight)| *weight).sum::<f32>();
        if total <= 0.0 || !total.is_finite() {
            return argmax_host(logits);
        }

        let mut candidates = Vec::new();
        let mut cumulative_probability = 0.0f32;
        for (idx, weight) in ranked {
            if weight <= 0.0 {
                continue;
            }
            candidates.push((idx, weight));
            cumulative_probability += weight / total;
            if cutoff < 1.0 && cumulative_probability >= cutoff {
                break;
            }
        }
        if candidates.is_empty() {
            return argmax_host(logits);
        }

        let candidate_total = candidates.iter().map(|(_, weight)| *weight).sum::<f32>();
        if candidate_total <= 0.0 || !candidate_total.is_finite() {
            return argmax_host(logits);
        }
        let sample = if sample.is_finite() {
            sample.clamp(0.0, 0.99999994)
        } else {
            0.0
        };
        let target = sample * candidate_total;
        let mut cumulative = 0.0f32;
        let last_idx = candidates
            .last()
            .map(|(idx, _)| *idx)
            .ok_or_else(|| anyhow!("cannot sample from empty candidates"))?;
        for (idx, weight) in candidates {
            cumulative += weight;
            if target < cumulative {
                return u32::try_from(idx).context("sampled token index does not fit u32");
            }
        }
        u32::try_from(last_idx).context("sampled token index does not fit u32")
    }

    fn validate_batched_generation_context_budget(
        label: &str,
        prompt_len: usize,
        max_decode_steps: usize,
        context: usize,
    ) -> Result<()> {
        let requested = prompt_len
            .checked_add(max_decode_steps)
            .with_context(|| format!("{label} token budget overflows usize"))?;
        if requested > context {
            bail!(
                "context_length_exceeded: {label} prompt length {prompt_len} plus max_tokens {max_decode_steps} exceeds qwen context length {context}"
            );
        }
        Ok(())
    }

    fn validate_generation_max_tokens(label: &str, max_tokens: usize) -> Result<usize> {
        if max_tokens == 0 {
            bail!("invalid_request_parameter: {label} requires max_tokens greater than 0");
        }
        Ok(max_tokens)
    }

    fn validate_batched_stop_token_sequences(
        label: &str,
        stop_token_sequences_per_request: &[Vec<Vec<u32>>],
        batch_count: usize,
    ) -> Result<()> {
        if !stop_token_sequences_per_request.is_empty()
            && stop_token_sequences_per_request.len() != batch_count
        {
            bail!(
                "{label} got {} stop-token sequence rows for {batch_count} request(s)",
                stop_token_sequences_per_request.len()
            );
        }
        Ok(())
    }

    fn row_stop_token_sequences(
        stop_token_sequences_per_request: &[Vec<Vec<u32>>],
        idx: usize,
    ) -> &[Vec<u32>] {
        stop_token_sequences_per_request
            .get(idx)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn generated_ends_with_stop_sequence(
        generated_tokens: &[u32],
        stop_token_sequences: &[Vec<u32>],
    ) -> bool {
        stop_token_sequences.iter().any(|stop| {
            !stop.is_empty()
                && stop.len() <= generated_tokens.len()
                && generated_tokens[generated_tokens.len() - stop.len()..] == stop[..]
        })
    }

    fn is_row_stop_sequence(
        token: u32,
        generated_tokens: &[u32],
        eos_token_id: Option<u32>,
        stop_token_sequences_per_request: &[Vec<Vec<u32>>],
        idx: usize,
    ) -> bool {
        is_stop_sequence(
            token,
            generated_tokens,
            eos_token_id,
            row_stop_token_sequences(stop_token_sequences_per_request, idx),
        )
    }

    fn is_stop_sequence(
        token: u32,
        generated_tokens: &[u32],
        eos_token_id: Option<u32>,
        stop_token_sequences: &[Vec<u32>],
    ) -> bool {
        Some(token) == eos_token_id
            || generated_ends_with_stop_sequence(generated_tokens, stop_token_sequences)
    }

    fn stop_sequences_from_stop_ids(stop_token_ids: &[u32]) -> Vec<Vec<u32>> {
        stop_token_ids.iter().map(|id| vec![*id]).collect()
    }

    fn shared_prefix_token_len(inputs: &[Vec<u32>]) -> usize {
        if inputs.len() < 2 {
            return 0;
        }
        let Some(first) = inputs.first() else {
            return 0;
        };
        let mut shared = first.len();
        for input in inputs.iter().skip(1) {
            shared = shared.min(input.len());
            for pos in 0..shared {
                if first[pos] != input[pos] {
                    shared = pos;
                    break;
                }
            }
            if shared == 0 {
                break;
            }
        }
        shared
    }

    pub struct CudaQwenGpuModel {
        info: CudaQwenGpuModelInfo,
        #[allow(dead_code)]
        config: QwenGgufConfig,
        #[allow(dead_code)]
        runtime: CudaRuntime,
        #[allow(dead_code)]
        stream: Stream,
        #[allow(dead_code)]
        cublas: Cublas,
        #[allow(dead_code)]
        cublas_lt: CublasLt,
        tensors: BTreeMap<String, GpuTensor>,
        matrices: BTreeMap<String, GpuMatrix>,
        vectors: BTreeMap<String, GpuVector>,
        paged_batch_pool: RefCell<Option<CudaPagedBatchDevicePool>>,
        recurrent_page_states: RefCell<BTreeMap<usize, RecurrentSsmRequestState>>,
        // Decode-path cache of the f16-dequantized output-head weight, keyed by matrix
        // name. The lm_head never changes but is re-dequantized (e.g. Q6_K) to a
        // vocab-wide output every token; caching it once removes that. Bounded to the
        // output head so models with no dp4a-eligible weights don't cache the whole
        // model in f16.
        dequant_f16_cache: RefCell<BTreeMap<String, DeviceBuffer>>,
        // Per-generation forward-pass timing: (prefill_micros, decode_micros,
        // forward_count). The first forward after a reset is the (batched) prompt
        // prefill; every later forward is a per-token decode. Fed by ForwardTimer
        // guards on the forward primitives; read by the generation entry points so
        // the non-continuous path can report an accurate prefill/decode split
        // instead of `scheduler_token_timing`'s proportional-by-token-count guess.
        generation_timing: Cell<(u64, u64, u64)>,
        // Re-entrancy guard so only the OUTERMOST forward primitive records (many
        // forward methods delegate to one another; nested timers would double-count).
        forward_in_progress: Cell<bool>,
    }

    /// RAII guard: on drop, attributes its lifetime to the model's per-generation
    /// prefill (first forward) or decode (subsequent) accumulator. Records on every
    /// exit path including `?`, and only measures time — never affects computation.
    struct ForwardTimer<'a> {
        model: &'a CudaQwenGpuModel,
        started: Instant,
    }

    impl Drop for ForwardTimer<'_> {
        fn drop(&mut self) {
            let micros = u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX);
            self.model.record_forward_micros(micros);
            self.model.forward_in_progress.set(false);
        }
    }

    struct RecurrentSsmRequestState {
        page_key: usize,
        tokens: Vec<u32>,
        seq_len: usize,
        layers: BTreeMap<usize, RecurrentSsmLayerState>,
    }

    struct RecurrentSsmLayerState {
        conv_ring: DeviceBuffer,
        conv_next: usize,
        conv_len: usize,
        recurrent: DeviceBuffer,
        conv_elements: usize,
        recurrent_elements: usize,
    }

    impl RecurrentSsmRequestState {
        fn new(page_key: usize) -> Self {
            Self {
                page_key,
                tokens: Vec::new(),
                seq_len: 0,
                layers: BTreeMap::new(),
            }
        }
    }

    impl RecurrentSsmLayerState {
        fn new(ssm: &QwenSsmDims, stream: &Stream) -> Result<Self> {
            let conv_len = ssm
                .conv_kernel
                .checked_mul(ssm.conv_dim)
                .context("CUDA recurrent SSM convolution state size overflows usize")?;
            let recurrent_len = ssm
                .time_step_rank
                .checked_mul(ssm.state_size)
                .and_then(|value| value.checked_mul(ssm.head_v_dim))
                .context("CUDA recurrent SSM matrix state size overflows usize")?;
            let conv_ring = DeviceBuffer::alloc(conv_len * std::mem::size_of::<f32>())
                .context("allocating CUDA recurrent SSM convolution state")?;
            let recurrent = DeviceBuffer::alloc(recurrent_len * std::mem::size_of::<f32>())
                .context("allocating CUDA recurrent SSM matrix state")?;
            conv_ring.memset_zero_async(stream)?;
            recurrent.memset_zero_async(stream)?;
            stream.synchronize()?;
            Ok(Self {
                conv_ring,
                conv_next: 0,
                conv_len: 0,
                recurrent,
                conv_elements: conv_len,
                recurrent_elements: recurrent_len,
            })
        }
    }

    pub struct CudaMmprojProjector {
        info: CudaMmprojProjectorInfo,
        #[allow(dead_code)]
        runtime: CudaRuntime,
        stream: Stream,
        cublas: Cublas,
        layers: Vec<MmprojLayer>,
    }

    struct MmprojLayer {
        weight: MmprojMatrix,
        bias: Option<MmprojBias>,
    }

    struct MmprojMatrix {
        name: String,
        rows: usize,
        cols: usize,
        bytes: usize,
        buffer: DeviceBuffer,
    }

    struct MmprojBias {
        len: usize,
        bytes: usize,
        buffer: DeviceBuffer,
    }

    // The projector owns CUDA resources and is only accessed through the backend
    // object that owns it.
    unsafe impl Send for CudaMmprojProjector {}

    trait CudaBatchedKvCacheWrite {
        fn write_layer_batched(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()>;
    }

    trait CudaSingleKvCacheWrite {
        fn write_layer(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()>;
    }

    impl CudaMmprojProjector {
        pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
            let runtime = CudaRuntime::probe()?;
            let stream = Stream::create()?;
            let cublas = Cublas::create()?;
            cublas.set_stream(&stream)?;
            let layer_names = mmproj_layer_names(gguf)?;
            let layout = if layer_names.len() == 1 {
                "linear"
            } else {
                "mlp-gelu"
            }
            .to_string();

            let mut layers: Vec<MmprojLayer> = Vec::with_capacity(layer_names.len());
            let mut total_device_bytes = 0usize;
            let mut bias_count = 0usize;
            for (idx, weight_name) in layer_names.iter().enumerate() {
                let weight = MmprojMatrix::load(gguf, weight_name)
                    .with_context(|| format!("loading CUDA mmproj matrix {weight_name}"))?;
                if idx > 0 {
                    let previous = &layers[idx - 1].weight;
                    if previous.rows != weight.cols {
                        bail!(
                            "CUDA mmproj layer {weight_name} input dim {} does not match previous output dim {}",
                            weight.cols,
                            previous.rows
                        );
                    }
                }
                total_device_bytes = total_device_bytes
                    .checked_add(weight.bytes)
                    .context("CUDA mmproj device byte total overflows usize")?;
                let bias_name = weight_name
                    .strip_suffix(".weight")
                    .map(|base| format!("{base}.bias"))
                    .unwrap_or_else(|| format!("{weight_name}.bias"));
                let bias = if gguf.tensor(&bias_name).is_some() {
                    let bias = MmprojBias::load(gguf, &bias_name, weight.rows)
                        .with_context(|| format!("loading CUDA mmproj bias {bias_name}"))?;
                    total_device_bytes = total_device_bytes
                        .checked_add(bias.bytes)
                        .context("CUDA mmproj device byte total overflows usize")?;
                    bias_count += 1;
                    Some(bias)
                } else {
                    None
                };
                layers.push(MmprojLayer { weight, bias });
            }
            let first = layers
                .first()
                .ok_or_else(|| anyhow!("CUDA mmproj projector has no layers"))?;
            let last = layers
                .last()
                .ok_or_else(|| anyhow!("CUDA mmproj projector has no layers"))?;
            let total_tensor_bytes = gguf.tensors().iter().try_fold(0u64, |acc, tensor| {
                Ok::<_, anyhow::Error>(acc + tensor.byte_len()?)
            })?;
            stream.synchronize()?;
            let info = CudaMmprojProjectorInfo {
                tensor_count: gguf.tensors().len(),
                total_tensor_bytes,
                layout,
                layer_count: layers.len(),
                input_dim: first.weight.cols,
                output_dim: last.weight.rows,
                hidden_dim: (layers.len() > 1).then_some(first.weight.rows),
                total_device_bytes,
                bias_count,
            };
            Ok(Self {
                info,
                runtime,
                stream,
                cublas,
                layers,
            })
        }

        pub fn info(&self) -> &CudaMmprojProjectorInfo {
            &self.info
        }

        pub fn project_features_host(&self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
            if rows == 0 {
                bail!("CUDA mmproj projection requires at least one feature row");
            }
            let input_dim = self.info.input_dim;
            let expected = rows
                .checked_mul(input_dim)
                .context("CUDA mmproj input element count overflows usize")?;
            if features.len() != expected {
                bail!(
                    "CUDA mmproj projection got {} feature values; expected {rows} x {input_dim} = {expected}",
                    features.len()
                );
            }
            let input = DeviceBuffer::alloc(std::mem::size_of_val(features))
                .context("allocating CUDA mmproj feature input")?;
            input
                .copy_from_host(features)
                .context("copying CUDA mmproj feature input")?;
            let mut tensor = GpuF32Tensor {
                rows,
                cols: input_dim,
                buffer: input,
            };
            for (idx, layer) in self.layers.iter().enumerate() {
                tensor = self.project_layer_device(&tensor, layer)?;
                if idx + 1 != self.layers.len() {
                    tensor = self.gelu_f32_device(&tensor)?;
                }
            }
            tensor.copy_to_host()
        }

        fn project_layer_device(
            &self,
            input: &GpuF32Tensor,
            layer: &MmprojLayer,
        ) -> Result<GpuF32Tensor> {
            if input.cols != layer.weight.cols {
                bail!(
                    "CUDA mmproj layer {} input cols {} do not match expected {}",
                    layer.weight.name,
                    input.cols,
                    layer.weight.cols
                );
            }
            let output_elements = input
                .rows
                .checked_mul(layer.weight.rows)
                .context("CUDA mmproj output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA mmproj layer output")?;
            self.cublas.matmul_f32_rhs_transposed_row_major(
                &input.buffer,
                &layer.weight.buffer,
                &output,
                input.rows,
                layer.weight.rows,
                layer.weight.cols,
            )?;
            self.stream.synchronize()?;
            let projected = GpuF32Tensor {
                rows: input.rows,
                cols: layer.weight.rows,
                buffer: output,
            };
            if let Some(bias) = &layer.bias {
                if bias.len != projected.cols {
                    bail!(
                        "CUDA mmproj bias length {} does not match output dim {}",
                        bias.len,
                        projected.cols
                    );
                }
                let biased = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                    .context("allocating CUDA mmproj biased output")?;
                crate::kernels::launch_add_rowwise(
                    &projected.buffer,
                    &bias.buffer,
                    &biased,
                    projected.rows,
                    projected.cols,
                    &self.stream,
                )?;
                self.stream.synchronize()?;
                return Ok(GpuF32Tensor {
                    rows: projected.rows,
                    cols: projected.cols,
                    buffer: biased,
                });
            }
            Ok(projected)
        }

        fn gelu_f32_device(&self, input: &GpuF32Tensor) -> Result<GpuF32Tensor> {
            let elements = input.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA mmproj GELU output")?;
            crate::kernels::launch_gelu(&input.buffer, &output, elements, &self.stream)?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }
    }

    impl fmt::Debug for CudaMmprojProjector {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CudaMmprojProjector")
                .field("info", &self.info)
                .finish_non_exhaustive()
        }
    }

    pub struct CudaVisionEncoder {
        info: CudaVisionEncoderInfo,
        config: VisionConfig,
        #[allow(dead_code)]
        runtime: CudaRuntime,
        stream: Stream,
        cublas: Cublas,
        matrices: BTreeMap<String, GpuMatrix>,
        vectors: BTreeMap<String, GpuVector>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum VisionMlpKind {
        Gelu,
        SwiGlu,
    }

    #[derive(Clone, Debug)]
    struct VisionConfig {
        architecture: String,
        variant: String,
        patch_size: usize,
        temporal_patch_size: usize,
        tokens_per_second: usize,
        spatial_merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
        patch_dim: usize,
        hidden_dim: usize,
        feed_forward_dim: usize,
        output_dim: usize,
        block_count: usize,
        head_count: usize,
        head_dim: usize,
        eps: f32,
        mlp_kind: VisionMlpKind,
        gelu_mlp_names_reversed: bool,
        uses_window_attention: bool,
        window_size: Option<usize>,
        full_attention_layers: BTreeSet<usize>,
    }

    struct VisionWindowPlan {
        patch_row_order: Vec<u32>,
        merged_reverse_order: Vec<u32>,
        window_start: Vec<u32>,
        window_end: Vec<u32>,
    }

    struct VisionWindowDevicePlan {
        merged_reverse_order: Vec<u32>,
        window_start: DeviceBuffer,
        window_end: DeviceBuffer,
    }

    // The encoder owns CUDA resources and is serialized by the backend mutex.
    unsafe impl Send for CudaVisionEncoder {}

    impl CudaVisionEncoder {
        pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
            let config = VisionConfig::from_gguf(gguf)?;
            let runtime = CudaRuntime::probe()?;
            let stream = Stream::create()?;
            let cublas = Cublas::create()?;
            cublas.set_stream(&stream)?;

            let mut matrices = BTreeMap::new();
            let mut total_matrix_bytes = 0usize;
            for spec in vision_matrix_specs(gguf, &config)? {
                let matrix = GpuMatrix::load(gguf, &spec)
                    .with_context(|| format!("loading CUDA vision matrix {}", spec.name))?;
                total_matrix_bytes = total_matrix_bytes
                    .checked_add(matrix.bytes)
                    .context("CUDA vision matrix byte total overflows usize")?;
                matrices.insert(spec.name, matrix);
            }

            let mut vectors = BTreeMap::new();
            let mut total_vector_bytes = 0usize;
            for info in gguf
                .tensors()
                .iter()
                .filter(|tensor| tensor.dimensions.len() == 1)
            {
                let vector = GpuVector::load(gguf, info)
                    .with_context(|| format!("loading CUDA vision vector {}", info.name))?;
                total_vector_bytes = total_vector_bytes
                    .checked_add(vector.bytes)
                    .context("CUDA vision vector byte total overflows usize")?;
                vectors.insert(info.name.clone(), vector);
            }
            stream.synchronize()?;

            let total_tensor_bytes = gguf.tensors().iter().try_fold(0u64, |acc, tensor| {
                Ok::<_, anyhow::Error>(acc + tensor.byte_len()?)
            })?;
            let total_device_bytes = total_matrix_bytes
                .checked_add(total_vector_bytes)
                .context("CUDA vision device byte total overflows usize")?;
            let info = CudaVisionEncoderInfo {
                tensor_count: gguf.tensors().len(),
                total_tensor_bytes,
                architecture: config.architecture.clone(),
                variant: config.variant.clone(),
                patch_size: config.patch_size,
                temporal_patch_size: config.temporal_patch_size,
                tokens_per_second: config.tokens_per_second,
                spatial_merge_size: config.spatial_merge_size,
                min_pixels: config.min_pixels,
                max_pixels: config.max_pixels,
                image_mean: config.image_mean,
                image_std: config.image_std,
                hidden_dim: config.hidden_dim,
                feed_forward_dim: config.feed_forward_dim,
                output_dim: config.output_dim,
                block_count: config.block_count,
                head_count: config.head_count,
                total_device_bytes,
                matrix_count: matrices.len(),
                vector_count: vectors.len(),
                uses_window_attention: config.uses_window_attention,
            };

            Ok(Self {
                info,
                config,
                runtime,
                stream,
                cublas,
                matrices,
                vectors,
            })
        }

        pub fn info(&self) -> &CudaVisionEncoderInfo {
            &self.info
        }

        pub fn encode_patches_host(
            &self,
            patches: &[f32],
            grids: &[[usize; 3]],
        ) -> Result<Vec<f32>> {
            if grids.is_empty() {
                bail!("CUDA Qwen-VL vision encoding requires at least one image grid");
            }
            let mut offset = 0usize;
            let mut output = Vec::new();
            for grid in grids {
                let rows = grid_patch_count(*grid)?;
                let elements = rows
                    .checked_mul(self.config.patch_dim)
                    .context("CUDA vision patch element count overflows usize")?;
                let end = offset
                    .checked_add(elements)
                    .context("CUDA vision patch offset overflows usize")?;
                let image_patches = patches.get(offset..end).ok_or_else(|| {
                    anyhow!(
                        "CUDA vision patch input ended early; expected at least {end} values, got {}",
                        patches.len()
                    )
                })?;
                let image_output = self.encode_one_image_patches_host(image_patches, *grid)?;
                output.extend(image_output);
                offset = end;
            }
            if offset != patches.len() {
                bail!(
                    "CUDA vision patch input has {} trailing values after consuming {offset}",
                    patches.len() - offset
                );
            }
            Ok(output)
        }

        fn encode_one_image_patches_host(
            &self,
            patches: &[f32],
            grid: [usize; 3],
        ) -> Result<Vec<f32>> {
            let rows = grid_patch_count(grid)?;
            let expected = rows
                .checked_mul(self.config.patch_dim)
                .context("CUDA vision image patch element count overflows usize")?;
            if patches.len() != expected {
                bail!(
                    "CUDA vision image patch input has {} values; expected {rows} x {} = {expected}",
                    patches.len(),
                    self.config.patch_dim
                );
            }
            let (pos_h, pos_w) = vision_position_ids(grid, self.config.spatial_merge_size)?;
            if pos_h.len() != rows || pos_w.len() != rows {
                bail!("CUDA vision position-id generation produced invalid lengths");
            }
            let window_plan = if self.config.uses_window_attention {
                Some(vision_window_plan(
                    grid,
                    self.config.patch_size,
                    self.config.spatial_merge_size,
                    self.config.window_size.ok_or_else(|| {
                        anyhow!(
                            "CUDA Qwen2.5-VL vision window attention requires window_size metadata"
                        )
                    })?,
                )?)
            } else {
                None
            };
            let (pos_h, pos_w) = if let Some(plan) = &window_plan {
                (
                    reorder_u32_by_ids(&pos_h, &plan.patch_row_order, "vision h positions")?,
                    reorder_u32_by_ids(&pos_w, &plan.patch_row_order, "vision w positions")?,
                )
            } else {
                (pos_h, pos_w)
            };
            let pos_h_device = DeviceBuffer::alloc(pos_h.len() * std::mem::size_of::<u32>())
                .context("allocating CUDA vision h positions")?;
            let pos_w_device = DeviceBuffer::alloc(pos_w.len() * std::mem::size_of::<u32>())
                .context("allocating CUDA vision w positions")?;
            pos_h_device
                .copy_from_host(&pos_h)
                .context("copying CUDA vision h positions")?;
            pos_w_device
                .copy_from_host(&pos_w)
                .context("copying CUDA vision w positions")?;

            let mut hidden = self.f32_tensor_from_host(
                patches,
                rows,
                self.config.patch_dim,
                "CUDA vision patches",
            )?;
            hidden = self.project_f32_device("patch", &hidden)?;
            let window_device_plan = if let Some(plan) = &window_plan {
                hidden = self.gather_rows_f32_device(
                    &hidden,
                    &plan.patch_row_order,
                    "CUDA Qwen2.5-VL window row reorder",
                )?;
                Some(self.vision_window_device_plan(plan)?)
            } else {
                None
            };

            for layer in 0..self.config.block_count {
                let norm1 = self.norm_f32_device(
                    &hidden,
                    &vision_norm1_aliases(layer),
                    &vision_norm1_bias_aliases(layer),
                )?;
                let mut q = self.project_f32_device(&format!("blk.{layer}.attn_q"), &norm1)?;
                q = self.add_optional_rowwise_alias(q, &vision_attn_q_bias_aliases(layer))?;
                self.apply_vision_rope_f32_device(&q, &pos_h_device, &pos_w_device)?;

                let mut k = self.project_f32_device(&format!("blk.{layer}.attn_k"), &norm1)?;
                k = self.add_optional_rowwise_alias(k, &vision_attn_k_bias_aliases(layer))?;
                self.apply_vision_rope_f32_device(&k, &pos_h_device, &pos_w_device)?;

                let mut v = self.project_f32_device(&format!("blk.{layer}.attn_v"), &norm1)?;
                v = self.add_optional_rowwise_alias(v, &vision_attn_v_bias_aliases(layer))?;
                let attn = if let Some(plan) = &window_device_plan {
                    if self.config.full_attention_layers.contains(&layer) {
                        self.full_attention_f32_device(&q, &k, &v)?
                    } else {
                        self.window_attention_f32_device(&q, &k, &v, plan)?
                    }
                } else {
                    self.full_attention_f32_device(&q, &k, &v)?
                };
                let mut attn_out =
                    self.project_f32_device(&format!("blk.{layer}.attn_out"), &attn)?;
                attn_out = self
                    .add_optional_rowwise_alias(attn_out, &vision_attn_out_bias_aliases(layer))?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let norm2 = self.norm_f32_device(
                    &hidden,
                    &vision_norm2_aliases(layer),
                    &vision_norm2_bias_aliases(layer),
                )?;
                let mlp_out = match self.config.mlp_kind {
                    VisionMlpKind::Gelu => self.vision_gelu_mlp_f32_device(layer, &norm2)?,
                    VisionMlpKind::SwiGlu => self.vision_swiglu_mlp_f32_device(layer, &norm2)?,
                };
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let mut merged = self.merge_f32_device(hidden)?;
            if let Some(plan) = &window_device_plan {
                merged = self.gather_rows_f32_device(
                    &merged,
                    &plan.merged_reverse_order,
                    "CUDA Qwen2.5-VL merger reverse row order",
                )?;
            }
            merged.copy_to_host()
        }

        fn matrix(&self, name: &str) -> Result<&GpuMatrix> {
            self.matrices
                .get(name)
                .ok_or_else(|| anyhow!("CUDA vision matrix {name} is missing"))
        }

        fn vector_alias<'a>(&'a self, aliases: &[String]) -> Option<&'a GpuVector> {
            aliases.iter().find_map(|name| self.vectors.get(name))
        }

        fn f32_tensor_from_host(
            &self,
            values: &[f32],
            rows: usize,
            cols: usize,
            operation: &str,
        ) -> Result<GpuF32Tensor> {
            let expected = rows
                .checked_mul(cols)
                .ok_or_else(|| anyhow!("{operation} element count overflows usize"))?;
            if values.len() != expected {
                bail!(
                    "{operation} got {} f32 values; expected {rows} x {cols} = {expected}",
                    values.len()
                );
            }
            let byte_len = expected
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| anyhow!("{operation} byte count overflows usize"))?;
            let buffer =
                DeviceBuffer::alloc(byte_len).with_context(|| format!("allocating {operation}"))?;
            buffer
                .copy_from_host(values)
                .with_context(|| format!("copying {operation}"))?;
            Ok(GpuF32Tensor { rows, cols, buffer })
        }

        fn project_f32_device(
            &self,
            matrix_name: &str,
            input: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            let matrix = self.matrix(matrix_name)?;
            if input.cols != matrix.cols {
                bail!(
                    "CUDA vision projection input cols {} do not match matrix cols {} for {matrix_name}",
                    input.cols,
                    matrix.cols
                );
            }
            if matrix.is_quantized() {
                let dequantized = self
                    .dequantize_matrix_f32_device(matrix)
                    .with_context(|| format!("dequantizing CUDA vision matrix {matrix_name}"))?;
                let output_elements = input
                    .rows
                    .checked_mul(matrix.rows)
                    .context("CUDA vision quantized projection output overflows usize")?;
                let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                    .context("allocating CUDA vision quantized projection output")?;
                self.cublas.matmul_f32_rhs_transposed_row_major(
                    &input.buffer,
                    &dequantized,
                    &output,
                    input.rows,
                    matrix.rows,
                    matrix.cols,
                )?;
                self.stream.synchronize()?;
                return Ok(GpuF32Tensor {
                    rows: input.rows,
                    cols: matrix.rows,
                    buffer: output,
                });
            }

            let dtype = matrix.gemm_dtype()?;
            let output_elements = input
                .rows
                .checked_mul(matrix.rows)
                .context("CUDA vision projection output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision projection output")?;
            if matches!(dtype, GemmDType::F32) {
                self.cublas.matmul_f32_rhs_transposed_row_major(
                    &input.buffer,
                    &matrix.buffer,
                    &output,
                    input.rows,
                    matrix.rows,
                    matrix.cols,
                )?;
                self.stream.synchronize()?;
                return Ok(GpuF32Tensor {
                    rows: input.rows,
                    cols: matrix.rows,
                    buffer: output,
                });
            }

            let input_elements = input
                .rows
                .checked_mul(input.cols)
                .context("CUDA vision projection input element count overflows usize")?;
            let input_cast = DeviceBuffer::alloc(input_elements * std::mem::size_of::<u16>())
                .context("allocating CUDA vision projection cast input")?;
            match dtype {
                GemmDType::F16 => crate::kernels::launch_cast_f32_to_f16(
                    &input.buffer,
                    &input_cast,
                    input_elements,
                    &self.stream,
                )?,
                GemmDType::BF16 => crate::kernels::launch_cast_f32_to_bf16(
                    &input.buffer,
                    &input_cast,
                    input_elements,
                    &self.stream,
                )?,
                GemmDType::F32 => unreachable!("F32 projection returned before cast path"),
            }

            self.cublas.matmul_mixed_rhs_transposed_row_major(
                &input_cast,
                &matrix.buffer,
                &output,
                input.rows,
                matrix.rows,
                matrix.cols,
                dtype,
                dtype,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: matrix.rows,
                buffer: output,
            })
        }

        fn dequantize_matrix_f32_device(&self, matrix: &GpuMatrix) -> Result<DeviceBuffer> {
            let elements = matrix
                .rows
                .checked_mul(matrix.cols)
                .context("CUDA vision dequantized matrix element count overflows usize")?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision dequantized matrix")?;
            crate::kernels::launch_dequantize_matrix(
                &matrix.buffer,
                &output,
                elements,
                matrix.quant_type_id()?,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(output)
        }

        fn norm_f32_device(
            &self,
            input: &GpuF32Tensor,
            weight_aliases: &[String],
            bias_aliases: &[String],
        ) -> Result<GpuF32Tensor> {
            let weight = self.vector_alias(weight_aliases).ok_or_else(|| {
                anyhow!("CUDA vision norm weight is missing; tried {weight_aliases:?}")
            })?;
            if input.cols != weight.len {
                bail!(
                    "CUDA vision norm input cols {} do not match weight length {}",
                    input.cols,
                    weight.len
                );
            }
            let output_elements = input
                .rows
                .checked_mul(input.cols)
                .context("CUDA vision norm output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision norm output")?;
            if let Some(bias) = self.vector_alias(bias_aliases) {
                if bias.len != weight.len {
                    bail!(
                        "CUDA vision layer norm bias length {} does not match weight length {}",
                        bias.len,
                        weight.len
                    );
                }
                crate::kernels::launch_layer_norm(
                    &input.buffer,
                    &weight.buffer,
                    &bias.buffer,
                    &output,
                    input.rows,
                    input.cols,
                    self.config.eps,
                    &self.stream,
                )?;
            } else {
                crate::kernels::launch_rms_norm(
                    &input.buffer,
                    &weight.buffer,
                    &output,
                    input.rows,
                    input.cols,
                    self.config.eps,
                    &self.stream,
                )?;
            }
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn add_optional_rowwise_alias(
            &self,
            input: GpuF32Tensor,
            bias_aliases: &[String],
        ) -> Result<GpuF32Tensor> {
            let Some(bias) = self.vector_alias(bias_aliases) else {
                return Ok(input);
            };
            if input.cols != bias.len {
                bail!(
                    "CUDA vision bias length {} does not match input cols {}",
                    bias.len,
                    input.cols
                );
            }
            let elements = input.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision bias output")?;
            crate::kernels::launch_add_rowwise(
                &input.buffer,
                &bias.buffer,
                &output,
                input.rows,
                input.cols,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn add_f32_device(
            &self,
            left: &GpuF32Tensor,
            right: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            left.ensure_same_shape(right, "CUDA vision residual add")?;
            let elements = left.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision add output")?;
            crate::kernels::launch_add(
                &left.buffer,
                &right.buffer,
                &output,
                elements,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: left.rows,
                cols: left.cols,
                buffer: output,
            })
        }

        fn gather_rows_f32_device(
            &self,
            input: &GpuF32Tensor,
            row_ids: &[u32],
            operation: &str,
        ) -> Result<GpuF32Tensor> {
            if row_ids.len() != input.rows {
                bail!(
                    "{operation} got {} row ids; expected {}",
                    row_ids.len(),
                    input.rows
                );
            }
            for row_id in row_ids {
                let row = usize::try_from(*row_id)
                    .with_context(|| format!("{operation} row id does not fit usize"))?;
                if row >= input.rows {
                    bail!(
                        "{operation} row id {row} is outside input row count {}",
                        input.rows
                    );
                }
            }
            let ids = DeviceBuffer::alloc(std::mem::size_of_val(row_ids))
                .with_context(|| format!("allocating {operation} row ids"))?;
            ids.copy_from_host(row_ids)
                .with_context(|| format!("copying {operation} row ids"))?;
            let output_elements = input
                .rows
                .checked_mul(input.cols)
                .with_context(|| format!("{operation} output element count overflows usize"))?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .with_context(|| format!("allocating {operation} output"))?;
            crate::kernels::launch_gather_rows_f32_to_f32(
                &input.buffer,
                &ids,
                &output,
                input.rows,
                input.cols,
                input.rows,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn apply_vision_rope_f32_device(
            &self,
            input: &GpuF32Tensor,
            pos_h: &DeviceBuffer,
            pos_w: &DeviceBuffer,
        ) -> Result<()> {
            if input.cols != self.config.hidden_dim || input.rows == 0 {
                bail!(
                    "CUDA vision RoPE input shape {}x{} is invalid for hidden dim {}",
                    input.rows,
                    input.cols,
                    self.config.hidden_dim
                );
            }
            crate::kernels::launch_vision_rope(
                &input.buffer,
                pos_h,
                pos_w,
                input.rows,
                self.config.head_count,
                self.config.head_dim,
                10_000.0,
                &self.stream,
            )?;
            self.stream.synchronize()
        }

        fn full_attention_f32_device(
            &self,
            q: &GpuF32Tensor,
            k: &GpuF32Tensor,
            v: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            let seq_len = q.rows;
            if q.cols != self.config.hidden_dim
                || k.cols != self.config.hidden_dim
                || v.cols != self.config.hidden_dim
            {
                bail!("CUDA vision attention q/k/v hidden dimensions are invalid");
            }
            if k.rows != seq_len || v.rows != seq_len {
                bail!("CUDA vision attention q/k/v row counts do not match");
            }
            let output_elements = seq_len
                .checked_mul(self.config.hidden_dim)
                .context("CUDA vision attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision attention output")?;
            crate::kernels::launch_full_attention(
                &q.buffer,
                &k.buffer,
                &v.buffer,
                &output,
                seq_len,
                self.config.head_count,
                self.config.head_count,
                self.config.head_dim,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: seq_len,
                cols: self.config.hidden_dim,
                buffer: output,
            })
        }

        fn vision_window_device_plan(
            &self,
            plan: &VisionWindowPlan,
        ) -> Result<VisionWindowDevicePlan> {
            let start = DeviceBuffer::alloc(std::mem::size_of_val(plan.window_start.as_slice()))
                .context("allocating CUDA vision window starts")?;
            let end = DeviceBuffer::alloc(std::mem::size_of_val(plan.window_end.as_slice()))
                .context("allocating CUDA vision window ends")?;
            start
                .copy_from_host(&plan.window_start)
                .context("copying CUDA vision window starts")?;
            end.copy_from_host(&plan.window_end)
                .context("copying CUDA vision window ends")?;
            Ok(VisionWindowDevicePlan {
                merged_reverse_order: plan.merged_reverse_order.clone(),
                window_start: start,
                window_end: end,
            })
        }

        fn window_attention_f32_device(
            &self,
            q: &GpuF32Tensor,
            k: &GpuF32Tensor,
            v: &GpuF32Tensor,
            plan: &VisionWindowDevicePlan,
        ) -> Result<GpuF32Tensor> {
            let seq_len = q.rows;
            if q.cols != self.config.hidden_dim
                || k.cols != self.config.hidden_dim
                || v.cols != self.config.hidden_dim
            {
                bail!("CUDA vision window attention q/k/v hidden dimensions are invalid");
            }
            if k.rows != seq_len || v.rows != seq_len {
                bail!("CUDA vision window attention q/k/v row counts do not match");
            }
            let output_elements = seq_len
                .checked_mul(self.config.hidden_dim)
                .context("CUDA vision window attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision window attention output")?;
            crate::kernels::launch_window_attention(
                &q.buffer,
                &k.buffer,
                &v.buffer,
                &plan.window_start,
                &plan.window_end,
                &output,
                seq_len,
                self.config.head_count,
                self.config.head_count,
                self.config.head_dim,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: seq_len,
                cols: self.config.hidden_dim,
                buffer: output,
            })
        }

        fn vision_gelu_mlp_f32_device(
            &self,
            layer: usize,
            input: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            let mut up = self.project_f32_device(&format!("blk.{layer}.ffn_up"), input)?;
            let up_bias_aliases = if self.config.gelu_mlp_names_reversed {
                vision_ffn_down_bias_aliases(layer)
            } else {
                vision_ffn_up_bias_aliases(layer)
            };
            up = self.add_optional_rowwise_alias(up, &up_bias_aliases)?;
            let activated = self.gelu_f32_device(&up)?;
            let mut down = self.project_f32_device(&format!("blk.{layer}.ffn_down"), &activated)?;
            let down_bias_aliases = if self.config.gelu_mlp_names_reversed {
                vision_ffn_up_bias_aliases(layer)
            } else {
                vision_ffn_down_bias_aliases(layer)
            };
            down = self.add_optional_rowwise_alias(down, &down_bias_aliases)?;
            Ok(down)
        }

        fn vision_swiglu_mlp_f32_device(
            &self,
            layer: usize,
            input: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            let mut gate = self.project_f32_device(&format!("blk.{layer}.ffn_gate"), input)?;
            gate = self.add_optional_rowwise_alias(gate, &vision_ffn_gate_bias_aliases(layer))?;
            let mut up = self.project_f32_device(&format!("blk.{layer}.ffn_up"), input)?;
            up = self.add_optional_rowwise_alias(up, &vision_ffn_up_bias_aliases(layer))?;
            let activated = self.silu_mul_f32_device(&gate, &up)?;
            let mut down = self.project_f32_device(&format!("blk.{layer}.ffn_down"), &activated)?;
            down = self.add_optional_rowwise_alias(down, &vision_ffn_down_bias_aliases(layer))?;
            Ok(down)
        }

        fn silu_mul_f32_device(
            &self,
            gate: &GpuF32Tensor,
            up: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            gate.ensure_same_shape(up, "CUDA vision SwiGLU")?;
            let elements = gate.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision SwiGLU output")?;
            crate::kernels::launch_silu_mul(
                &gate.buffer,
                &up.buffer,
                &output,
                elements,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: gate.rows,
                cols: gate.cols,
                buffer: output,
            })
        }

        fn gelu_f32_device(&self, input: &GpuF32Tensor) -> Result<GpuF32Tensor> {
            let elements = input.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA vision GELU output")?;
            crate::kernels::launch_gelu(&input.buffer, &output, elements, &self.stream)?;
            self.stream.synchronize()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn merge_f32_device(&self, hidden: GpuF32Tensor) -> Result<GpuF32Tensor> {
            let merge_unit = self
                .config
                .spatial_merge_size
                .checked_mul(self.config.spatial_merge_size)
                .context("CUDA vision merge unit overflows usize")?;
            if merge_unit == 0 || hidden.rows % merge_unit != 0 {
                bail!(
                    "CUDA vision hidden rows {} are not divisible by merge unit {merge_unit}",
                    hidden.rows
                );
            }
            let normed = self.norm_f32_device(
                &hidden,
                &vision_merger_norm_aliases(),
                &vision_merger_norm_bias_aliases(),
            )?;
            let flattened = GpuF32Tensor {
                rows: normed.rows / merge_unit,
                cols: normed
                    .cols
                    .checked_mul(merge_unit)
                    .context("CUDA vision merger flattened cols overflow usize")?,
                buffer: normed.buffer,
            };
            let mut hidden = self.project_f32_device("merger.0", &flattened)?;
            hidden = self.add_optional_rowwise_alias(hidden, &vision_merger_0_bias_aliases())?;
            hidden = self.gelu_f32_device(&hidden)?;
            let mut output = self.project_f32_device("merger.2", &hidden)?;
            output = self.add_optional_rowwise_alias(output, &vision_merger_2_bias_aliases())?;
            Ok(output)
        }
    }

    impl fmt::Debug for CudaVisionEncoder {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CudaVisionEncoder")
                .field("info", &self.info)
                .field("matrix_names", &self.matrices.keys().collect::<Vec<_>>())
                .field("vector_names", &self.vectors.keys().collect::<Vec<_>>())
                .finish_non_exhaustive()
        }
    }

    impl VisionConfig {
        fn from_gguf(gguf: &GgufFile) -> Result<Self> {
            let architecture = gguf
                .metadata_string("general.architecture")
                .unwrap_or("unknown")
                .to_ascii_lowercase();
            let projector_type = gguf
                .metadata_string("clip.projector_type")
                .or_else(|| gguf.metadata_string("clip.vision.projector_type"))
                .unwrap_or("")
                .to_ascii_lowercase();
            let has_qwen_vision_tensor = gguf.tensor("v.patch_embd.weight").is_some()
                || gguf.tensor("visual.patch_embed.proj.weight").is_some();
            let qwen_architecture = architecture.contains("qwen2vl")
                || architecture.contains("qwen2_vl")
                || architecture.contains("qwen2.5")
                || architecture.contains("qwen2_5")
                || projector_type == "qwen2vl_merger"
                || projector_type == "qwen2.5vl_merger"
                || projector_type == "qwen2_5vl_merger";
            if !qwen_architecture || !has_qwen_vision_tensor {
                bail!(
                    "unsupported CUDA vision GGUF architecture '{architecture}'; expected Qwen2-VL/Qwen2.5-VL vision mmproj bundle"
                );
            }

            let patch_size = metadata_usize_alias(
                gguf,
                &["clip.vision.patch_size", "qwen2vl.vision.patch_size"],
                "vision patch size",
            )?;
            let temporal_patch_size = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.temporal_patch_size",
                    "qwen2vl.vision.temporal_patch_size",
                ],
            )?
            .unwrap_or(2);
            let spatial_merge_size = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.spatial_merge_size",
                    "qwen2vl.vision.spatial_merge_size",
                ],
            )?
            .unwrap_or(2);
            let tokens_per_second = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.tokens_per_second",
                    "qwen2vl.vision.tokens_per_second",
                ],
            )?
            .unwrap_or_else(|| {
                if projector_type == "qwen2.5vl_merger" || projector_type == "qwen2_5vl_merger" {
                    4
                } else {
                    1
                }
            });
            let hidden_dim = metadata_usize_alias(
                gguf,
                &[
                    "clip.vision.embedding_length",
                    "qwen2vl.vision.embedding_length",
                ],
                "vision embedding length",
            )?;
            let block_count = metadata_usize_alias(
                gguf,
                &["clip.vision.block_count", "qwen2vl.vision.block_count"],
                "vision block count",
            )?;
            let head_count = metadata_usize_alias(
                gguf,
                &[
                    "clip.vision.attention.head_count",
                    "qwen2vl.vision.attention.head_count",
                ],
                "vision attention head count",
            )?;
            if head_count == 0 || hidden_dim == 0 || hidden_dim % head_count != 0 {
                bail!(
                    "invalid CUDA vision head metadata: hidden_dim={hidden_dim}, head_count={head_count}"
                );
            }
            let head_dim = hidden_dim / head_count;
            if head_dim % 4 != 0 {
                bail!("CUDA Qwen-VL vision RoPE requires head_dim divisible by 4, got {head_dim}");
            }
            let patch_dim = 3usize
                .checked_mul(temporal_patch_size)
                .and_then(|value| value.checked_mul(patch_size))
                .and_then(|value| value.checked_mul(patch_size))
                .context("CUDA vision patch dimension overflows usize")?;
            let mlp_kind = if find_tensor_alias(gguf, &vision_ffn_gate_weight_aliases(0)).is_some()
            {
                VisionMlpKind::SwiGlu
            } else {
                VisionMlpKind::Gelu
            };
            let metadata_feed_forward_dim = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.feed_forward_length",
                    "qwen2vl.vision.feed_forward_length",
                ],
            )?;
            let inferred_feed_forward_dim = infer_vision_ff_dim(gguf, mlp_kind);
            let feed_forward_dim = match (metadata_feed_forward_dim, inferred_feed_forward_dim) {
                (Some(metadata), Some(inferred)) if metadata != inferred => inferred,
                (Some(metadata), _) => metadata,
                (None, Some(inferred)) => inferred,
                (None, None) => {
                    bail!(
                        "GGUF metadata missing vision feed_forward_length and it could not be inferred"
                    )
                }
            };
            let gelu_mlp_names_reversed = mlp_kind == VisionMlpKind::Gelu
                && vision_gelu_mlp_names_reversed(gguf, hidden_dim, feed_forward_dim);
            let merge_unit = spatial_merge_size
                .checked_mul(spatial_merge_size)
                .context("CUDA vision merge unit overflows usize")?;
            let merger_input_dim = hidden_dim
                .checked_mul(merge_unit)
                .context("CUDA vision merger input dim overflows usize")?;
            let output_dim = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.projection_dim",
                    "qwen2vl.vision.projection_dim",
                ],
            )?
            .or_else(|| infer_vision_merger_output_dim(gguf, merger_input_dim))
            .ok_or_else(|| {
                anyhow!("GGUF metadata missing vision projection_dim and it could not be inferred")
            })?;
            let min_pixels = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.image_min_pixels",
                    "qwen2vl.vision.image_min_pixels",
                ],
            )?
            .unwrap_or(56 * 56);
            let max_pixels = metadata_usize_alias_optional(
                gguf,
                &[
                    "clip.vision.image_max_pixels",
                    "qwen2vl.vision.image_max_pixels",
                ],
            )?
            .unwrap_or(14 * 14 * 4 * 1280);
            let image_mean = metadata_f32_array3(gguf, "clip.vision.image_mean")?
                .unwrap_or([0.48145466, 0.4578275, 0.40821073]);
            let image_std = metadata_f32_array3(gguf, "clip.vision.image_std")?
                .unwrap_or([0.26862954, 0.26130258, 0.27577711]);
            let eps = gguf
                .metadata_f32("clip.vision.attention.layer_norm_epsilon")
                .or_else(|| gguf.metadata_f32("qwen2vl.vision.attention.layer_norm_epsilon"))
                .unwrap_or(1.0e-6);
            let n_wa_pattern = metadata_usize_alias_optional(
                gguf,
                &["clip.vision.n_wa_pattern", "qwen2vl.vision.n_wa_pattern"],
            )?
            .unwrap_or(0);
            let window_size = metadata_usize_alias_optional(
                gguf,
                &["clip.vision.window_size", "qwen2vl.vision.window_size"],
            )?
            .or_else(|| {
                (n_wa_pattern > 0).then(|| {
                    patch_size
                        .checked_mul(spatial_merge_size)
                        .and_then(|value| value.checked_mul(4))
                        .unwrap_or(112)
                })
            });
            let full_attention_layers =
                infer_vision_full_attention_layers(gguf, block_count, n_wa_pattern)?;
            let uses_window_attention = n_wa_pattern > 0
                || window_size.is_some()
                || gguf.metadata().contains_key("clip.vision.wa_layer_indexes");
            let variant = if mlp_kind == VisionMlpKind::SwiGlu || uses_window_attention {
                "qwen2.5-vl"
            } else {
                "qwen2-vl"
            }
            .to_string();

            Ok(Self {
                architecture,
                variant,
                patch_size,
                temporal_patch_size,
                tokens_per_second,
                spatial_merge_size,
                min_pixels,
                max_pixels,
                image_mean,
                image_std,
                patch_dim,
                hidden_dim,
                feed_forward_dim,
                output_dim,
                block_count,
                head_count,
                head_dim,
                eps,
                mlp_kind,
                gelu_mlp_names_reversed,
                uses_window_attention,
                window_size,
                full_attention_layers,
            })
        }
    }

    fn metadata_usize_alias(gguf: &GgufFile, keys: &[&str], label: &str) -> Result<usize> {
        metadata_usize_alias_optional(gguf, keys)?
            .ok_or_else(|| anyhow!("GGUF metadata missing required {label}; tried {keys:?}"))
    }

    fn metadata_usize_alias_optional(gguf: &GgufFile, keys: &[&str]) -> Result<Option<usize>> {
        for key in keys {
            if let Some(value) = gguf.metadata_u32(key) {
                return usize::try_from(value)
                    .map(Some)
                    .with_context(|| format!("GGUF metadata {key} does not fit usize"));
            }
        }
        Ok(None)
    }

    fn metadata_i32_array_alias_optional(
        gguf: &GgufFile,
        keys: &[&str],
    ) -> Result<Option<Vec<i32>>> {
        for key in keys {
            if let Some(values) = gguf.metadata_i32_array(key)? {
                return Ok(Some(values));
            }
        }
        Ok(None)
    }

    fn infer_vision_full_attention_layers(
        gguf: &GgufFile,
        block_count: usize,
        n_wa_pattern: usize,
    ) -> Result<BTreeSet<usize>> {
        let mut layers = BTreeSet::new();
        if let Some(values) = metadata_i32_array_alias_optional(
            gguf,
            &[
                "clip.vision.fullatt_block_indexes",
                "clip.vision.full_attention_layer_indexes",
                "qwen2vl.vision.fullatt_block_indexes",
            ],
        )? {
            for value in values {
                if value < 0 {
                    bail!("vision full attention layer index {value} is negative");
                }
                let layer = usize::try_from(value)
                    .context("vision full attention layer index does not fit usize")?;
                if layer >= block_count {
                    bail!(
                        "vision full attention layer index {layer} is outside block count {block_count}"
                    );
                }
                layers.insert(layer);
            }
        } else if n_wa_pattern > 0 {
            let mut layer = n_wa_pattern - 1;
            while layer < block_count {
                layers.insert(layer);
                layer = layer
                    .checked_add(n_wa_pattern)
                    .context("vision window-attention pattern overflows usize")?;
            }
        }
        Ok(layers)
    }

    fn metadata_f32_array3(gguf: &GgufFile, key: &str) -> Result<Option<[f32; 3]>> {
        let Some(values) = gguf.metadata_f32_array(key)? else {
            return Ok(None);
        };
        if values.len() != 3 {
            bail!(
                "GGUF metadata {key} must contain exactly 3 floats, got {}",
                values.len()
            );
        }
        Ok(Some([values[0], values[1], values[2]]))
    }

    fn find_tensor_alias<'a>(gguf: &GgufFile, aliases: &'a [String]) -> Option<&'a str> {
        aliases
            .iter()
            .find(|name| gguf.tensor(name).is_some())
            .map(String::as_str)
    }

    fn matrix_alias_spec(
        gguf: &GgufFile,
        canonical: impl Into<String>,
        aliases: Vec<String>,
        rows: usize,
        cols: usize,
    ) -> Result<MatrixSpec> {
        let name = canonical.into();
        let tensor_name = find_tensor_alias(gguf, &aliases)
            .ok_or_else(|| anyhow!("missing CUDA vision matrix {name}; tried {aliases:?}"))?
            .to_string();
        Ok(MatrixSpec {
            name,
            tensor_name,
            rows,
            cols,
            expert_index: None,
            row_slice: None,
        })
    }

    fn vision_matrix_specs(gguf: &GgufFile, config: &VisionConfig) -> Result<Vec<MatrixSpec>> {
        let merge_unit = config
            .spatial_merge_size
            .checked_mul(config.spatial_merge_size)
            .context("CUDA vision merge unit overflows usize")?;
        let merger_dim = config
            .hidden_dim
            .checked_mul(merge_unit)
            .context("CUDA vision merger dim overflows usize")?;
        let mut specs = Vec::new();
        specs.push(matrix_alias_spec(
            gguf,
            "patch",
            vision_patch_weight_aliases(),
            config.hidden_dim,
            config.patch_dim,
        )?);
        for layer in 0..config.block_count {
            specs.extend([
                matrix_alias_spec(
                    gguf,
                    format!("blk.{layer}.attn_q"),
                    vision_attn_q_weight_aliases(layer),
                    config.hidden_dim,
                    config.hidden_dim,
                )?,
                matrix_alias_spec(
                    gguf,
                    format!("blk.{layer}.attn_k"),
                    vision_attn_k_weight_aliases(layer),
                    config.hidden_dim,
                    config.hidden_dim,
                )?,
                matrix_alias_spec(
                    gguf,
                    format!("blk.{layer}.attn_v"),
                    vision_attn_v_weight_aliases(layer),
                    config.hidden_dim,
                    config.hidden_dim,
                )?,
                matrix_alias_spec(
                    gguf,
                    format!("blk.{layer}.attn_out"),
                    vision_attn_out_weight_aliases(layer),
                    config.hidden_dim,
                    config.hidden_dim,
                )?,
            ]);
            if config.mlp_kind == VisionMlpKind::SwiGlu {
                specs.push(matrix_alias_spec(
                    gguf,
                    format!("blk.{layer}.ffn_gate"),
                    vision_ffn_gate_weight_aliases(layer),
                    config.feed_forward_dim,
                    config.hidden_dim,
                )?);
            }
            if config.mlp_kind == VisionMlpKind::Gelu && config.gelu_mlp_names_reversed {
                specs.extend([
                    matrix_alias_spec(
                        gguf,
                        format!("blk.{layer}.ffn_up"),
                        vision_ffn_down_weight_aliases(layer),
                        config.feed_forward_dim,
                        config.hidden_dim,
                    )?,
                    matrix_alias_spec(
                        gguf,
                        format!("blk.{layer}.ffn_down"),
                        vision_ffn_up_weight_aliases(layer),
                        config.hidden_dim,
                        config.feed_forward_dim,
                    )?,
                ]);
            } else {
                specs.extend([
                    matrix_alias_spec(
                        gguf,
                        format!("blk.{layer}.ffn_up"),
                        vision_ffn_up_weight_aliases(layer),
                        config.feed_forward_dim,
                        config.hidden_dim,
                    )?,
                    matrix_alias_spec(
                        gguf,
                        format!("blk.{layer}.ffn_down"),
                        vision_ffn_down_weight_aliases(layer),
                        config.hidden_dim,
                        config.feed_forward_dim,
                    )?,
                ]);
            }
        }
        specs.extend([
            matrix_alias_spec(
                gguf,
                "merger.0",
                vision_merger_0_weight_aliases(),
                merger_dim,
                merger_dim,
            )?,
            matrix_alias_spec(
                gguf,
                "merger.2",
                vision_merger_2_weight_aliases(),
                config.output_dim,
                merger_dim,
            )?,
        ]);
        Ok(specs)
    }

    fn infer_vision_ff_dim(gguf: &GgufFile, mlp_kind: VisionMlpKind) -> Option<usize> {
        let aliases = if mlp_kind == VisionMlpKind::SwiGlu {
            vision_ffn_gate_weight_aliases(0)
        } else {
            vision_ffn_up_weight_aliases(0)
        };
        infer_matrix_other_dim(gguf, &aliases, None)
    }

    fn infer_vision_merger_output_dim(gguf: &GgufFile, merger_input_dim: usize) -> Option<usize> {
        infer_matrix_other_dim(
            gguf,
            &vision_merger_2_weight_aliases(),
            Some(merger_input_dim),
        )
    }

    fn vision_gelu_mlp_names_reversed(
        gguf: &GgufFile,
        hidden_dim: usize,
        feed_forward_dim: usize,
    ) -> bool {
        let up_bias = vision_vector_len(gguf, &vision_ffn_up_bias_aliases(0));
        let down_bias = vision_vector_len(gguf, &vision_ffn_down_bias_aliases(0));
        matches!(
            (up_bias, down_bias),
            (Some(up), Some(down)) if up == hidden_dim && down == feed_forward_dim
        )
    }

    fn vision_vector_len(gguf: &GgufFile, aliases: &[String]) -> Option<usize> {
        let tensor = find_tensor_alias(gguf, aliases).and_then(|name| gguf.tensor(name))?;
        if tensor.info.dimensions.len() != 1 {
            return None;
        }
        usize::try_from(tensor.info.dimensions[0]).ok()
    }

    fn infer_matrix_other_dim(
        gguf: &GgufFile,
        aliases: &[String],
        expected_other: Option<usize>,
    ) -> Option<usize> {
        let tensor = find_tensor_alias(gguf, aliases).and_then(|name| gguf.tensor(name))?;
        if tensor.info.dimensions.len() != 2 {
            return None;
        }
        let dims = tensor
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).ok())
            .collect::<Option<Vec<_>>>()?;
        match expected_other {
            Some(expected) if dims[0] == expected => Some(dims[1]),
            Some(expected) if dims[1] == expected => Some(dims[0]),
            Some(_) => None,
            None => dims.into_iter().max(),
        }
    }

    fn grid_patch_count(grid: [usize; 3]) -> Result<usize> {
        let [t, h, w] = grid;
        if t == 0 || h == 0 || w == 0 {
            bail!("CUDA vision image_grid_thw values must be non-zero, got {grid:?}");
        }
        t.checked_mul(h)
            .and_then(|value| value.checked_mul(w))
            .context("CUDA vision grid patch count overflows usize")
    }

    fn vision_position_ids(
        grid: [usize; 3],
        spatial_merge_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let [t, h, w] = grid;
        if spatial_merge_size == 0 || h % spatial_merge_size != 0 || w % spatial_merge_size != 0 {
            bail!(
                "CUDA vision grid {grid:?} is not divisible by spatial merge size {spatial_merge_size}"
            );
        }
        let rows = grid_patch_count(grid)?;
        let mut pos_h = Vec::with_capacity(rows);
        let mut pos_w = Vec::with_capacity(rows);
        for _ in 0..t {
            for block_h in 0..(h / spatial_merge_size) {
                for block_w in 0..(w / spatial_merge_size) {
                    for merge_h in 0..spatial_merge_size {
                        for merge_w in 0..spatial_merge_size {
                            pos_h.push(
                                u32::try_from(block_h * spatial_merge_size + merge_h)
                                    .context("vision h position does not fit u32")?,
                            );
                            pos_w.push(
                                u32::try_from(block_w * spatial_merge_size + merge_w)
                                    .context("vision w position does not fit u32")?,
                            );
                        }
                    }
                }
            }
        }
        Ok((pos_h, pos_w))
    }

    fn vision_window_plan(
        grid: [usize; 3],
        patch_size: usize,
        spatial_merge_size: usize,
        window_size: usize,
    ) -> Result<VisionWindowPlan> {
        let [t, h, w] = grid;
        let rows = grid_patch_count(grid)?;
        if patch_size == 0 || spatial_merge_size == 0 || window_size == 0 {
            bail!(
                "CUDA vision window plan requires non-zero patch, merge, and window sizes; got patch={patch_size}, merge={spatial_merge_size}, window={window_size}"
            );
        }
        if h % spatial_merge_size != 0 || w % spatial_merge_size != 0 {
            bail!(
                "CUDA vision grid {grid:?} is not divisible by spatial merge size {spatial_merge_size}"
            );
        }
        let merge_unit = spatial_merge_size
            .checked_mul(spatial_merge_size)
            .context("CUDA vision merge unit overflows usize")?;
        if rows % merge_unit != 0 {
            bail!("CUDA vision rows {rows} are not divisible by merge unit {merge_unit}");
        }
        let denom = patch_size
            .checked_mul(spatial_merge_size)
            .context("CUDA vision window denominator overflows usize")?;
        let window_grid = window_size / denom;
        if window_grid == 0 {
            bail!(
                "CUDA vision window_size {window_size} is smaller than patch_size*spatial_merge_size {denom}"
            );
        }

        let llm_h = h / spatial_merge_size;
        let llm_w = w / spatial_merge_size;
        let group_count = t
            .checked_mul(llm_h)
            .and_then(|value| value.checked_mul(llm_w))
            .context("CUDA vision window group count overflows usize")?;
        let mut merged_order = Vec::with_capacity(group_count);
        let mut window_group_ranges = Vec::new();
        let num_windows_h = llm_h.div_ceil(window_grid);
        let num_windows_w = llm_w.div_ceil(window_grid);
        for frame in 0..t {
            let frame_offset = frame
                .checked_mul(llm_h)
                .and_then(|value| value.checked_mul(llm_w))
                .context("CUDA vision window frame offset overflows usize")?;
            for window_h in 0..num_windows_h {
                for window_w in 0..num_windows_w {
                    let start = merged_order.len();
                    for local_h in 0..window_grid {
                        let group_h = window_h
                            .checked_mul(window_grid)
                            .and_then(|value| value.checked_add(local_h))
                            .context("CUDA vision window h index overflows usize")?;
                        if group_h >= llm_h {
                            continue;
                        }
                        for local_w in 0..window_grid {
                            let group_w = window_w
                                .checked_mul(window_grid)
                                .and_then(|value| value.checked_add(local_w))
                                .context("CUDA vision window w index overflows usize")?;
                            if group_w >= llm_w {
                                continue;
                            }
                            let group = frame_offset
                                .checked_add(
                                    group_h
                                        .checked_mul(llm_w)
                                        .and_then(|value| value.checked_add(group_w))
                                        .context(
                                            "CUDA vision window group index overflows usize",
                                        )?,
                                )
                                .context("CUDA vision window group offset overflows usize")?;
                            merged_order.push(group);
                        }
                    }
                    let end = merged_order.len();
                    if end > start {
                        window_group_ranges.push((start, end));
                    }
                }
            }
        }
        if merged_order.len() != group_count {
            bail!(
                "CUDA vision window plan produced {} merged rows; expected {group_count}",
                merged_order.len()
            );
        }

        let mut patch_row_order = Vec::with_capacity(rows);
        for group in &merged_order {
            let base = group
                .checked_mul(merge_unit)
                .context("CUDA vision window patch row base overflows usize")?;
            for local in 0..merge_unit {
                patch_row_order.push(
                    u32::try_from(base + local)
                        .context("CUDA vision window patch row id does not fit u32")?,
                );
            }
        }

        let mut merged_reverse_order = vec![0u32; group_count];
        for (new_idx, old_idx) in merged_order.iter().copied().enumerate() {
            let slot = merged_reverse_order
                .get_mut(old_idx)
                .ok_or_else(|| anyhow!("CUDA vision window reverse index {old_idx} is invalid"))?;
            *slot = u32::try_from(new_idx)
                .context("CUDA vision window reverse row id does not fit u32")?;
        }

        let mut window_start = Vec::with_capacity(rows);
        let mut window_end = Vec::with_capacity(rows);
        for (start_group, end_group) in window_group_ranges {
            let start = start_group
                .checked_mul(merge_unit)
                .context("CUDA vision window start overflows usize")?;
            let end = end_group
                .checked_mul(merge_unit)
                .context("CUDA vision window end overflows usize")?;
            for _ in start..end {
                window_start
                    .push(u32::try_from(start).context("vision window start does not fit u32")?);
                window_end.push(u32::try_from(end).context("vision window end does not fit u32")?);
            }
        }
        if window_start.len() != rows || window_end.len() != rows {
            bail!(
                "CUDA vision window bounds produced {} rows; expected {rows}",
                window_start.len()
            );
        }

        Ok(VisionWindowPlan {
            patch_row_order,
            merged_reverse_order,
            window_start,
            window_end,
        })
    }

    fn reorder_u32_by_ids(values: &[u32], row_ids: &[u32], operation: &str) -> Result<Vec<u32>> {
        if values.len() != row_ids.len() {
            bail!(
                "{operation} reorder got {} values and {} row ids",
                values.len(),
                row_ids.len()
            );
        }
        let mut reordered = Vec::with_capacity(values.len());
        for row_id in row_ids {
            let row = usize::try_from(*row_id)
                .with_context(|| format!("{operation} row id does not fit usize"))?;
            let value = values
                .get(row)
                .ok_or_else(|| anyhow!("{operation} row id {row} is outside value length"))?;
            reordered.push(*value);
        }
        Ok(reordered)
    }

    fn vision_patch_weight_aliases() -> Vec<String> {
        vec![
            "v.patch_embd.weight".to_string(),
            "visual.patch_embed.proj.weight".to_string(),
        ]
    }

    fn vision_attn_q_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.attn_q.weight"),
            format!("visual.blocks.{layer}.attn.q.weight"),
        ]
    }

    fn vision_attn_k_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.attn_k.weight"),
            format!("visual.blocks.{layer}.attn.k.weight"),
        ]
    }

    fn vision_attn_v_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.attn_v.weight"),
            format!("visual.blocks.{layer}.attn.v.weight"),
        ]
    }

    fn vision_attn_out_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.attn_out.weight"),
            format!("visual.blocks.{layer}.attn.proj.weight"),
        ]
    }

    fn vision_ffn_gate_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ffn_gate.weight"),
            format!("visual.blocks.{layer}.mlp.gate_proj.weight"),
        ]
    }

    fn vision_ffn_up_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ffn_up.weight"),
            format!("visual.blocks.{layer}.mlp.fc1.weight"),
            format!("visual.blocks.{layer}.mlp.up_proj.weight"),
        ]
    }

    fn vision_ffn_down_weight_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ffn_down.weight"),
            format!("visual.blocks.{layer}.mlp.fc2.weight"),
            format!("visual.blocks.{layer}.mlp.down_proj.weight"),
        ]
    }

    fn bias_aliases(weight_aliases: Vec<String>) -> Vec<String> {
        weight_aliases
            .into_iter()
            .map(|name| {
                name.strip_suffix(".weight")
                    .map(|base| format!("{base}.bias"))
                    .unwrap_or_else(|| format!("{name}.bias"))
            })
            .collect()
    }

    fn vision_attn_q_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_attn_q_weight_aliases(layer))
    }

    fn vision_attn_k_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_attn_k_weight_aliases(layer))
    }

    fn vision_attn_v_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_attn_v_weight_aliases(layer))
    }

    fn vision_attn_out_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_attn_out_weight_aliases(layer))
    }

    fn vision_ffn_gate_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_ffn_gate_weight_aliases(layer))
    }

    fn vision_ffn_up_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_ffn_up_weight_aliases(layer))
    }

    fn vision_ffn_down_bias_aliases(layer: usize) -> Vec<String> {
        bias_aliases(vision_ffn_down_weight_aliases(layer))
    }

    fn vision_norm1_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ln1.weight"),
            format!("visual.blocks.{layer}.norm1.weight"),
        ]
    }

    fn vision_norm1_bias_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ln1.bias"),
            format!("visual.blocks.{layer}.norm1.bias"),
        ]
    }

    fn vision_norm2_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ln2.weight"),
            format!("visual.blocks.{layer}.norm2.weight"),
        ]
    }

    fn vision_norm2_bias_aliases(layer: usize) -> Vec<String> {
        vec![
            format!("v.blk.{layer}.ln2.bias"),
            format!("visual.blocks.{layer}.norm2.bias"),
        ]
    }

    fn vision_merger_norm_aliases() -> Vec<String> {
        vec![
            "v.post_ln.weight".to_string(),
            "visual.merger.ln_q.weight".to_string(),
        ]
    }

    fn vision_merger_norm_bias_aliases() -> Vec<String> {
        vec![
            "v.post_ln.bias".to_string(),
            "visual.merger.ln_q.bias".to_string(),
        ]
    }

    fn vision_merger_0_weight_aliases() -> Vec<String> {
        vec![
            "mm.0.weight".to_string(),
            "visual.merger.mlp.0.weight".to_string(),
        ]
    }

    fn vision_merger_2_weight_aliases() -> Vec<String> {
        vec![
            "mm.2.weight".to_string(),
            "visual.merger.mlp.2.weight".to_string(),
        ]
    }

    fn vision_merger_0_bias_aliases() -> Vec<String> {
        bias_aliases(vision_merger_0_weight_aliases())
    }

    fn vision_merger_2_bias_aliases() -> Vec<String> {
        bias_aliases(vision_merger_2_weight_aliases())
    }

    impl MmprojMatrix {
        fn load(gguf: &GgufFile, name: &str) -> Result<Self> {
            let tensor = gguf
                .tensor(name)
                .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| {
                    usize::try_from(*dim).context("mmproj matrix dimension does not fit usize")
                })
                .collect::<Result<Vec<_>>>()?;
            if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
                bail!("mmproj matrix tensor {name} must be rank 2, got {dims:?}");
            }
            let cols = dims[0];
            let rows = dims[1];
            let elements = rows
                .checked_mul(cols)
                .context("mmproj matrix element count overflows usize")?;
            let values = read_tensor_as_f32(tensor.bytes, tensor.info.dtype, elements)
                .with_context(|| format!("reading mmproj matrix {name} as f32"))?;
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .context("mmproj matrix byte count overflows usize")?;
            let buffer = DeviceBuffer::alloc(bytes)
                .with_context(|| format!("allocating CUDA mmproj matrix {name}"))?;
            buffer
                .copy_from_host(&values)
                .with_context(|| format!("copying CUDA mmproj matrix {name}"))?;
            Ok(Self {
                name: name.to_string(),
                rows,
                cols,
                bytes,
                buffer,
            })
        }
    }

    impl MmprojBias {
        fn load(gguf: &GgufFile, name: &str, expected_len: usize) -> Result<Self> {
            let tensor = gguf
                .tensor(name)
                .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| {
                    usize::try_from(*dim).context("mmproj bias dimension does not fit usize")
                })
                .collect::<Result<Vec<_>>>()?;
            if dims.as_slice() != [expected_len] {
                bail!("mmproj bias tensor {name} has shape {dims:?}; expected [{expected_len}]");
            }
            let values = read_tensor_as_f32(tensor.bytes, tensor.info.dtype, expected_len)
                .with_context(|| format!("reading mmproj bias {name} as f32"))?;
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .context("mmproj bias byte count overflows usize")?;
            let buffer = DeviceBuffer::alloc(bytes)
                .with_context(|| format!("allocating CUDA mmproj bias {name}"))?;
            buffer
                .copy_from_host(&values)
                .with_context(|| format!("copying CUDA mmproj bias {name}"))?;
            Ok(Self {
                len: expected_len,
                bytes,
                buffer,
            })
        }
    }

    fn mmproj_layer_names(gguf: &GgufFile) -> Result<Vec<String>> {
        for prefix in [
            "mm",
            "mm_projector",
            "model.mm_projector",
            "projector",
            "model.projector",
            "vision_projector",
            "model.vision_projector",
            "visual.merger.mlp",
            "model.visual.merger.mlp",
            "multi_modal_projector",
            "model.multi_modal_projector",
            "multi_modal_projector.layers",
            "model.multi_modal_projector.layers",
            "mm_projector.layers",
            "model.mm_projector.layers",
            "projector.layers",
            "model.projector.layers",
            "vision_projector.layers",
            "model.vision_projector.layers",
        ] {
            if let Some(names) = mmproj_numbered_layer_names(gguf, prefix) {
                return Ok(names);
            }
        }
        for prefix in [
            "multi_modal_projector",
            "model.multi_modal_projector",
            "mm_projector",
            "model.mm_projector",
            "projector",
            "model.projector",
            "vision_projector",
            "model.vision_projector",
        ] {
            if let Some(names) = mmproj_linear_n_layer_names(gguf, prefix) {
                return Ok(names);
            }
        }

        const CANDIDATES: &[&[&str]] = &[
            &["mm.0.weight", "mm.2.weight"],
            &["mm_projector.0.weight", "mm_projector.2.weight"],
            &["model.mm_projector.0.weight", "model.mm_projector.2.weight"],
            &["visual.merger.mlp.0.weight", "visual.merger.mlp.2.weight"],
            &[
                "multi_modal_projector.linear_1.weight",
                "multi_modal_projector.linear_2.weight",
            ],
            &[
                "model.multi_modal_projector.linear_1.weight",
                "model.multi_modal_projector.linear_2.weight",
            ],
            &[
                "mm_projector.linear_1.weight",
                "mm_projector.linear_2.weight",
            ],
            &[
                "model.mm_projector.linear_1.weight",
                "model.mm_projector.linear_2.weight",
            ],
            &["projector.linear_1.weight", "projector.linear_2.weight"],
            &[
                "model.projector.linear_1.weight",
                "model.projector.linear_2.weight",
            ],
            &[
                "vision_projector.linear_1.weight",
                "vision_projector.linear_2.weight",
            ],
            &[
                "model.vision_projector.linear_1.weight",
                "model.vision_projector.linear_2.weight",
            ],
            &[
                "multi_modal_projector.fc1.weight",
                "multi_modal_projector.fc2.weight",
            ],
            &[
                "model.multi_modal_projector.fc1.weight",
                "model.multi_modal_projector.fc2.weight",
            ],
            &["mm_projector.fc1.weight", "mm_projector.fc2.weight"],
            &[
                "model.mm_projector.fc1.weight",
                "model.mm_projector.fc2.weight",
            ],
            &["projector.fc1.weight", "projector.fc2.weight"],
            &["model.projector.fc1.weight", "model.projector.fc2.weight"],
            &["vision_projector.fc1.weight", "vision_projector.fc2.weight"],
            &[
                "model.vision_projector.fc1.weight",
                "model.vision_projector.fc2.weight",
            ],
            &["projector.0.weight", "projector.2.weight"],
            &["mm_projector.weight"],
            &["model.mm_projector.weight"],
            &["projector.weight"],
            &["model.projector.weight"],
            &["multi_modal_projector.linear.weight"],
            &["model.multi_modal_projector.linear.weight"],
            &["multi_modal_projector.projector.weight"],
            &["model.multi_modal_projector.projector.weight"],
            &["vision_projector.weight"],
            &["model.vision_projector.weight"],
            &["visual_projection.weight"],
            &["model.visual_projection.weight"],
            &["image_projection.weight"],
            &["model.image_projection.weight"],
            &["visual.merger.proj.weight"],
            &["mm.0.weight"],
        ];
        for candidate in CANDIDATES {
            if candidate.iter().all(|name| gguf.tensor(name).is_some()) {
                return Ok(candidate.iter().map(|name| (*name).to_string()).collect());
            }
        }
        let rank2 = gguf
            .tensors()
            .iter()
            .filter(|tensor| tensor.dimensions.len() == 2)
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        if rank2.len() == 1 {
            return Ok(rank2);
        }
        bail!(
            "unsupported CUDA mmproj tensor layout; expected known linear/MLP projector tensors, found {} rank-2 tensor(s)",
            rank2.len()
        )
    }

    fn mmproj_numbered_layer_names(gguf: &GgufFile, prefix: &str) -> Option<Vec<String>> {
        let even = mmproj_collect_indexed_layer_names(gguf, prefix, 0, 2);
        let contiguous = mmproj_collect_indexed_layer_names(gguf, prefix, 0, 1);
        let names = if contiguous.len() > even.len() {
            contiguous
        } else {
            even
        };
        (names.len() >= 2).then_some(names)
    }

    fn mmproj_collect_indexed_layer_names(
        gguf: &GgufFile,
        prefix: &str,
        start: usize,
        step: usize,
    ) -> Vec<String> {
        let mut names = Vec::new();
        let mut idx = start;
        while idx <= 64 {
            let name = format!("{prefix}.{idx}.weight");
            if gguf.tensor(&name).is_none() {
                break;
            }
            names.push(name);
            idx = idx.saturating_add(step);
            if step == 0 {
                break;
            }
        }
        names
    }

    fn mmproj_linear_n_layer_names(gguf: &GgufFile, prefix: &str) -> Option<Vec<String>> {
        let mut names = Vec::new();
        for idx in 1..=64 {
            let name = format!("{prefix}.linear_{idx}.weight");
            if gguf.tensor(&name).is_none() {
                break;
            }
            names.push(name);
        }
        (names.len() >= 2).then_some(names)
    }

    // The model owns CUDA resources and `CudaBackend` serializes access through a
    // mutex before exposing it through the HTTP backend trait.
    unsafe impl Send for CudaQwenGpuModel {}

    /// Decide whether to convert quantized weights to a resident FP16 copy at
    /// load. `HI_CUDA_WEIGHTS_F16` forces the choice (`1`/`0`); unset means
    /// **auto** — enable FP16 (a large decode speedup) when the FP16 weights fit
    /// in ~80% of free VRAM, leaving room for the KV cache, activations, and
    /// prefill scratch, and otherwise keep the low-VRAM quantized path so large
    /// models still load.
    /// Whether the fused dp4a Q4_0 GEMV decode path is enabled (default: yes; set
    /// `HI_CUDA_NO_Q4_GEMV` to fall back to dequant-per-op). Only reached when weights
    /// are kept quantized on the GPU, i.e. when f16 doesn't fit (large models), where
    /// it is ~6x faster than dequantizing each matmul. Small models where f16 fits use
    /// cuBLAS f16 and never hit this path.
    /// Force tensor-core (WMMA) prefill attention ON via `HI_CUDA_WMMA_ATTN` even for
    /// non-quantized (f16-stored) models. WMMA is f16 (not bit-parity with the f32
    /// kernel), so it defaults on only for quantized models (see the dispatch); this
    /// env opts small models in too.
    fn wmma_attn_forced_on() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("HI_CUDA_WMMA_ATTN").is_ok())
    }

    /// Force WMMA prefill attention OFF via `HI_CUDA_NO_WMMA_ATTN` (safety fallback to
    /// the f32 flash kernel, e.g. if f16 precision ever matters).
    fn wmma_attn_forced_off() -> bool {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| std::env::var("HI_CUDA_NO_WMMA_ATTN").is_ok())
    }

    fn q4_0_gemv_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("HI_CUDA_NO_Q4_GEMV").is_err())
    }

    fn weights_f16_choice(specs: &[MatrixSpec]) -> bool {
        match std::env::var("HI_CUDA_WEIGHTS_F16")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("1" | "true" | "yes" | "on") => return true,
            Some("0" | "false" | "no" | "off") => return false,
            _ => {}
        }
        let f16_bytes = specs.iter().fold(0usize, |total, spec| {
            total.saturating_add(spec.rows.saturating_mul(spec.cols).saturating_mul(2))
        });
        match crate::runtime::free_memory_bytes() {
            // Enable FP16 only if the weights leave ~20% of free VRAM for the KV
            // cache, activations, and transient prefill scratch.
            Ok(free) => f16_bytes.saturating_mul(5) <= free.saturating_mul(4),
            // Can't measure free memory — stay on the safe quantized path.
            Err(_) => false,
        }
    }

    impl CudaQwenGpuModel {
        /// Barrier between two device ops on the model's (single) stream. Ops are
        /// stream-ordered, so op N+1 already waits for op N without a host round-trip;
        /// host reads go through the self-synchronizing `copy_to_host`; and intermediate
        /// buffers free (a synchronizing `cudaFree`) at scope end. So these per-op host
        /// syncs are redundant and just serialize the pipeline. Skipped unless
        /// `HI_CUDA_OP_SYNC` is set (opt-in safety fallback to the old behavior).
        fn op_barrier(&self) -> Result<()> {
            use std::sync::OnceLock;
            static FORCE_SYNC: OnceLock<bool> = OnceLock::new();
            if *FORCE_SYNC.get_or_init(|| std::env::var("HI_CUDA_OP_SYNC").is_ok()) {
                self.stream.synchronize()?;
            }
            Ok(())
        }

        /// Start a fresh per-generation prefill/decode timing window.
        pub(crate) fn reset_generation_timing(&self) {
            self.generation_timing.set((0, 0, 0));
        }

        /// Attribute one forward pass's elapsed micros: the first forward after a
        /// reset is the batched prompt prefill; every later one is a decode step.
        fn record_forward_micros(&self, micros: u64) {
            let (prefill, decode, count) = self.generation_timing.get();
            if count == 0 {
                self.generation_timing
                    .set((prefill.saturating_add(micros), decode, 1));
            } else {
                self.generation_timing
                    .set((prefill, decode.saturating_add(micros), count + 1));
            }
        }

        /// (prefill_micros, decode_micros) accumulated since the last reset.
        pub(crate) fn take_generation_timing(&self) -> (u64, u64) {
            let (prefill, decode, _) = self.generation_timing.get();
            (prefill, decode)
        }

        /// RAII timer that attributes its scope to the current forward pass. Add
        /// `let _t = self.forward_timer();` as the first statement of a forward
        /// primitive to include it in the prefill/decode split. Returns `None` when
        /// a forward is already being timed (nested delegation) so only the
        /// outermost call is counted.
        fn forward_timer(&self) -> Option<ForwardTimer<'_>> {
            if self.forward_in_progress.get() {
                return None;
            }
            self.forward_in_progress.set(true);
            Some(ForwardTimer {
                model: self,
                started: Instant::now(),
            })
        }

        pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
            let config = gguf.qwen_config()?;
            gguf.validate_qwen_tensors()?;
            let runtime = CudaRuntime::probe()?;
            let stream = Stream::create()?;
            let cublas = Cublas::create()?;
            cublas.set_stream(&stream)?;
            let cublas_lt = CublasLt::create()?;

            let mut tensors = BTreeMap::new();
            let mut largest_tensor_bytes = 0usize;
            let mut dtypes = BTreeSet::new();
            for info in gguf.tensors() {
                let bytes = usize::try_from(info.byte_len()?).with_context(|| {
                    format!("tensor {} byte length does not fit usize", info.name)
                })?;
                largest_tensor_bytes = largest_tensor_bytes.max(bytes);
                dtypes.insert(info.dtype.label().to_string());
                tensors.insert(
                    info.name.clone(),
                    GpuTensor {
                        shape: info.dimensions.clone(),
                        dtype: info.dtype,
                        bytes,
                    },
                );
            }

            // Convert quantized weights to a resident FP16 copy at load so decode
            // skips the per-token dequant-to-f32 and runs the FP16 GEMM directly
            // — a large decode speedup at the cost of ~2 bytes/param VRAM. On by
            // default when the FP16 weights fit (auto), overridable via
            // HI_CUDA_WEIGHTS_F16; large models that only fit quantized stay
            // quantized.
            let matrix_specs = qwen_matrix_specs(gguf, &config)?;
            let weights_f16 = weights_f16_choice(&matrix_specs);
            let mut matrices = BTreeMap::new();
            let mut total_matrix_bytes = 0usize;
            let mut quantized_matrix_count = 0usize;
            for spec in matrix_specs {
                let mut matrix = GpuMatrix::load(gguf, &spec)
                    .with_context(|| format!("loading CUDA matrix {}", spec.name))?;
                if weights_f16 && matrix.is_quantized() {
                    matrix = matrix
                        .into_f16(&stream)
                        .with_context(|| format!("converting CUDA matrix {} to f16", spec.name))?;
                }
                if matrix.is_quantized() {
                    quantized_matrix_count += 1;
                }
                total_matrix_bytes = total_matrix_bytes
                    .checked_add(matrix.bytes)
                    .context("CUDA normalized matrix byte total overflows usize")?;
                matrices.insert(spec.name, matrix);
            }

            let mut vectors = BTreeMap::new();
            let mut total_vector_bytes = 0usize;
            for info in gguf
                .tensors()
                .iter()
                .filter(|tensor| tensor.dimensions.len() == 1)
            {
                let vector = GpuVector::load(gguf, info)
                    .with_context(|| format!("loading CUDA vector {}", info.name))?;
                total_vector_bytes = total_vector_bytes
                    .checked_add(vector.bytes)
                    .context("CUDA vector byte total overflows usize")?;
                vectors.insert(info.name.clone(), vector);
            }
            for spec in qwen_vector_alias_specs(gguf, &config)? {
                if vectors.contains_key(&spec.name) {
                    continue;
                }
                let info = gguf
                    .tensors()
                    .iter()
                    .find(|info| info.name == spec.tensor_name)
                    .ok_or_else(|| anyhow!("GGUF tensor {} is missing", spec.tensor_name))?;
                let vector = GpuVector::load_from_spec(gguf, info, &spec).with_context(|| {
                    format!(
                        "loading CUDA vector {} from normalized tensor {}",
                        spec.name, spec.tensor_name
                    )
                })?;
                total_vector_bytes = total_vector_bytes
                    .checked_add(vector.bytes)
                    .context("CUDA vector byte total overflows usize")?;
                vectors.insert(spec.name, vector);
            }
            stream.synchronize()?;
            let total_device_bytes = total_matrix_bytes
                .checked_add(total_vector_bytes)
                .context("CUDA retained device byte total overflows usize")?;

            let info = CudaQwenGpuModelInfo {
                tensor_count: tensors.len(),
                total_device_bytes,
                largest_tensor_bytes,
                dtype_summary: dtypes.into_iter().collect(),
                matrix_count: matrices.len(),
                total_matrix_bytes,
                quantized_matrix_count,
                vector_count: vectors.len(),
                total_vector_bytes,
            };
            Ok(Self {
                info,
                config,
                runtime,
                stream,
                cublas,
                cublas_lt,
                tensors,
                matrices,
                vectors,
                paged_batch_pool: RefCell::new(None),
                dequant_f16_cache: RefCell::new(BTreeMap::new()),
                recurrent_page_states: RefCell::new(BTreeMap::new()),
                generation_timing: Cell::new((0, 0, 0)),
                forward_in_progress: Cell::new(false),
            })
        }

        pub fn info(&self) -> &CudaQwenGpuModelInfo {
            &self.info
        }

        pub fn has_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        pub fn tensor(&self, name: &str) -> Option<&GpuTensor> {
            self.tensors.get(name)
        }

        pub fn has_matrix(&self, name: &str) -> bool {
            self.matrices.contains_key(name)
        }

        pub fn matrix(&self, name: &str) -> Option<&GpuMatrix> {
            self.matrices.get(name)
        }

        pub fn has_vector(&self, name: &str) -> bool {
            self.vectors.contains_key(name)
        }

        pub fn vector(&self, name: &str) -> Option<&GpuVector> {
            self.vectors.get(name)
        }

        pub fn project_f32_host(
            &self,
            matrix_name: &str,
            input: &[f32],
            rows: usize,
        ) -> Result<Vec<f32>> {
            let matrix = self
                .matrix(matrix_name)
                .ok_or_else(|| anyhow!("CUDA matrix {matrix_name} is missing"))?;
            if rows == 0 {
                bail!("CUDA projection rows must be greater than zero");
            }
            let input_elements = rows
                .checked_mul(matrix.cols)
                .context("CUDA projection input element count overflows usize")?;
            if input.len() != input_elements {
                bail!(
                    "CUDA projection input has {} elements; expected {rows} x {} = {input_elements}",
                    input.len(),
                    matrix.cols
                );
            }

            let input_buffer = DeviceBuffer::alloc(input.len() * std::mem::size_of::<f32>())
                .context("allocating CUDA projection f32 input")?;
            input_buffer
                .copy_from_host(input)
                .context("copying CUDA projection f32 input")?;
            let projected = self.project_f32_device(
                matrix_name,
                &GpuF32Tensor {
                    rows,
                    cols: matrix.cols,
                    buffer: input_buffer,
                },
            )?;
            projected.copy_to_host()
        }

        fn project_f32_device(
            &self,
            matrix_name: &str,
            input: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            let matrix = self
                .matrix(matrix_name)
                .ok_or_else(|| anyhow!("CUDA matrix {matrix_name} is missing"))?;
            if input.cols != matrix.cols {
                bail!(
                    "CUDA projection input cols {} do not match matrix cols {} for {matrix_name}",
                    input.cols,
                    matrix.cols
                );
            }
            if matrix.is_quantized() {
                return self.project_quantized_f32_device(matrix_name, matrix, input);
            }
            let dtype = matrix.gemm_dtype()?;
            let output_elements = input
                .rows
                .checked_mul(matrix.rows)
                .context("CUDA projection output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA projection output")?;
            if matches!(dtype, GemmDType::F32) {
                self.cublas.matmul_f32_rhs_transposed_row_major(
                    &input.buffer,
                    &matrix.buffer,
                    &output,
                    input.rows,
                    matrix.rows,
                    matrix.cols,
                )?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: input.rows,
                    cols: matrix.rows,
                    buffer: output,
                });
            }

            let input_elements = input
                .rows
                .checked_mul(input.cols)
                .context("CUDA projection input element count overflows usize")?;
            let input_cast = DeviceBuffer::alloc(input_elements * std::mem::size_of::<u16>())
                .context("allocating CUDA projection cast input")?;
            match dtype {
                GemmDType::F16 => crate::kernels::launch_cast_f32_to_f16(
                    &input.buffer,
                    &input_cast,
                    input_elements,
                    &self.stream,
                )?,
                GemmDType::BF16 => crate::kernels::launch_cast_f32_to_bf16(
                    &input.buffer,
                    &input_cast,
                    input_elements,
                    &self.stream,
                )?,
                GemmDType::F32 => unreachable!("F32 projection returned before cast path"),
            }

            self.cublas.matmul_mixed_rhs_transposed_row_major(
                &input_cast,
                &matrix.buffer,
                &output,
                input.rows,
                matrix.rows,
                matrix.cols,
                dtype,
                dtype,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: matrix.rows,
                buffer: output,
            })
        }

        fn project_quantized_f32_device(
            &self,
            matrix_name: &str,
            matrix: &GpuMatrix,
            input: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            // Single-token decode fast path: quantize the activation to int8 and run a
            // fused dp4a Q4_0 GEMV, reading 4-bit weights + int8 activation directly.
            if q4_0_gemv_enabled()
                && input.rows == 1
                && matches!(matrix.dtype, GgufTensorType::Q4_0)
                && matrix.cols % 32 == 0
            {
                let k = matrix.cols;
                let nblocks = k / 32;
                let xq = DeviceBuffer::alloc(k).context("allocating Q8 activation quants")?;
                let dx = DeviceBuffer::alloc(nblocks * std::mem::size_of::<f32>())
                    .context("allocating Q8 activation scales")?;
                let xsum = DeviceBuffer::alloc(nblocks * std::mem::size_of::<i32>())
                    .context("allocating Q8 activation sums")?;
                crate::kernels::launch_quantize_q8_row(
                    &input.buffer,
                    &xq,
                    &dx,
                    &xsum,
                    k,
                    &self.stream,
                )
                .with_context(|| format!("Q8 activation quant for {matrix_name}"))?;
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA Q4_0 dp4a GEMV output")?;
                crate::kernels::launch_q4_0_dp4a_gemv(
                    &matrix.buffer,
                    &xq,
                    &dx,
                    &xsum,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA Q4_0 dp4a GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            let weight_elements = matrix
                .rows
                .checked_mul(matrix.cols)
                .context("CUDA quantized weight element count overflows usize")?;
            let is_output_head = input.rows == 1
                && self
                    .config
                    .vocab_size
                    .and_then(|v| usize::try_from(v).ok())
                    .map(|vocab| matrix.rows == vocab)
                    .unwrap_or(false);
            // Fused Q6_K GEMV (M=1 decode, non-output-head layer weights): read Q6_K
            // directly instead of dequantizing the whole matrix to f32 every token — the
            // per-op dequant is ~12x slower for Q6_K models kept quantized (f16 doesn't
            // fit). The output head stays on the f16 cache below (cuBLAS, marginally
            // faster and reused every token).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::Q6_K)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA Q6_K GEMV output")?;
                crate::kernels::launch_q6_k_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA Q6_K GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused Q4_K GEMV (M=1 decode, non-output-head layer weights) — same idea as
            // the Q6_K path, for the most common quant (Q4_K_M). Keeps large Q4_K models
            // usable when weights don't fit f16 (else per-op dequant is ~5x slower).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::Q4_K)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA Q4_K GEMV output")?;
                crate::kernels::launch_q4_k_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA Q4_K GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused Q5_K GEMV (M=1 decode, non-output-head layer weights).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::Q5_K)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA Q5_K GEMV output")?;
                crate::kernels::launch_q5_k_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA Q5_K GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused Q3_K GEMV (M=1 decode, non-output-head layer weights).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::Q3_K)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA Q3_K GEMV output")?;
                crate::kernels::launch_q3_k_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA Q3_K GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused Q2_K GEMV (M=1 decode, non-output-head layer weights).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::Q2_K)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA Q2_K GEMV output")?;
                crate::kernels::launch_q2_k_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA Q2_K GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused IQ4_NL GEMV (M=1 decode, non-output-head layer weights). Block-32
            // format (cols % 32, not 256), non-linear lookup table.
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::IQ4_NL)
                && matrix.cols % 32 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA IQ4_NL GEMV output")?;
                crate::kernels::launch_iq4_nl_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA IQ4_NL GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused IQ4_XS GEMV (M=1 decode, non-output-head layer weights). Block-256
            // I-quant (per-sub-block scale + IQ4_NL lookup table).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::IQ4_XS)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA IQ4_XS GEMV output")?;
                crate::kernels::launch_iq4_xs_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA IQ4_XS GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Fused IQ3_S GEMV (M=1 decode, non-output-head layer weights). Block-256
            // I-quant (grid codebook + per-weight signs + sub-block scale).
            if input.rows == 1
                && !is_output_head
                && matches!(matrix.dtype, GgufTensorType::IQ3_S)
                && matrix.cols % 256 == 0
            {
                let output = DeviceBuffer::alloc(matrix.rows * std::mem::size_of::<f32>())
                    .context("allocating CUDA IQ3_S GEMV output")?;
                crate::kernels::launch_iq3_s_gemv(
                    &matrix.buffer,
                    &input.buffer,
                    &output,
                    matrix.rows,
                    matrix.cols,
                    &self.stream,
                )
                .with_context(|| format!("CUDA IQ3_S GEMV for {matrix_name}"))?;
                self.op_barrier()?;
                return Ok(GpuF32Tensor {
                    rows: 1,
                    cols: matrix.rows,
                    buffer: output,
                });
            }
            // Quantized weight matmul via tensor-core f16 GEMM. Dequantizing to f16 is
            // expensive, so cache the f16 weight and reuse it — weights never change,
            // avoiding re-dequantizing the Q6_K lm_head every token. Bounded to the
            // output head (the single weight projecting to a vocab-wide output): a
            // general per-weight cache would, on models with no dp4a-eligible weights
            // (e.g. IQ4_NL), cache the whole model in f16 and OOM. Everything else
            // dequantizes per-op (prefill M>1, and layer weights that miss dp4a).
            let output = if is_output_head {
                if !self.dequant_f16_cache.borrow().contains_key(matrix_name) {
                    let w = self.dequantize_matrix_to_f16(matrix, weight_elements)?;
                    self.dequant_f16_cache
                        .borrow_mut()
                        .insert(matrix_name.to_string(), w);
                }
                let cache = self.dequant_f16_cache.borrow();
                let weight_f16 = cache
                    .get(matrix_name)
                    .expect("cached f16 weight just inserted");
                self.matmul_input_f16_weight(input, weight_f16, matrix)?
            } else {
                let weight_f16 = self.dequantize_matrix_to_f16(matrix, weight_elements)?;
                self.matmul_input_f16_weight(input, &weight_f16, matrix)?
            };
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: matrix.rows,
                buffer: output,
            })
        }

        /// Dequantize a quantized weight matrix to f16 (dequant -> f32 -> cast f16).
        fn dequantize_matrix_to_f16(
            &self,
            matrix: &GpuMatrix,
            weight_elements: usize,
        ) -> Result<DeviceBuffer> {
            let dequantized = self.dequantize_matrix_f32_device(matrix)?;
            let weight_f16 = DeviceBuffer::alloc(weight_elements * std::mem::size_of::<u16>())
                .context("allocating CUDA f16 weight scratch")?;
            crate::kernels::launch_cast_f32_to_f16(
                &dequantized,
                &weight_f16,
                weight_elements,
                &self.stream,
            )?;
            Ok(weight_f16)
        }

        /// Cast the input to f16 and run the tensor-core f16 GEMM against an already-f16
        /// weight, returning the f32 output buffer.
        fn matmul_input_f16_weight(
            &self,
            input: &GpuF32Tensor,
            weight_f16: &DeviceBuffer,
            matrix: &GpuMatrix,
        ) -> Result<DeviceBuffer> {
            let input_elements = input
                .rows
                .checked_mul(input.cols)
                .context("CUDA quantized projection input element count overflows usize")?;
            let input_f16 = DeviceBuffer::alloc(input_elements * std::mem::size_of::<u16>())
                .context("allocating CUDA f16 input scratch")?;
            crate::kernels::launch_cast_f32_to_f16(
                &input.buffer,
                &input_f16,
                input_elements,
                &self.stream,
            )?;
            let output_elements = input
                .rows
                .checked_mul(matrix.rows)
                .context("CUDA quantized projection output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA quantized projection output")?;
            self.cublas.matmul_mixed_rhs_transposed_row_major(
                &input_f16,
                weight_f16,
                &output,
                input.rows,
                matrix.rows,
                matrix.cols,
                GemmDType::F16,
                GemmDType::F16,
            )?;
            self.op_barrier()?;
            Ok(output)
        }

        fn dequantize_matrix_f32_device(&self, matrix: &GpuMatrix) -> Result<DeviceBuffer> {
            let elements = matrix
                .rows
                .checked_mul(matrix.cols)
                .context("CUDA dequantized matrix element count overflows usize")?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA dequantized matrix")?;
            crate::kernels::launch_dequantize_matrix(
                &matrix.buffer,
                &output,
                elements,
                matrix.quant_type_id()?,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(output)
        }

        fn matrix_f32_device_owned(
            &self,
            matrix_name: &str,
            matrix: &GpuMatrix,
        ) -> Result<Option<DeviceBuffer>> {
            let elements = matrix
                .rows
                .checked_mul(matrix.cols)
                .context("CUDA matrix f32 element count overflows usize")?;
            if matrix.is_quantized() {
                return self
                    .dequantize_matrix_f32_device(matrix)
                    .with_context(|| format!("dequantizing CUDA matrix {matrix_name}"))
                    .map(Some);
            }
            match matrix.dtype {
                GgufTensorType::F32 => Ok(None),
                GgufTensorType::F16 | GgufTensorType::BF16 => {
                    let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                        .with_context(|| {
                            format!("allocating CUDA f32 cast matrix {matrix_name}")
                        })?;
                    match matrix.dtype {
                        GgufTensorType::F16 => crate::kernels::launch_cast_f16_to_f32(
                            &matrix.buffer,
                            &output,
                            elements,
                            &self.stream,
                        )?,
                        GgufTensorType::BF16 => crate::kernels::launch_cast_bf16_to_f32(
                            &matrix.buffer,
                            &output,
                            elements,
                            &self.stream,
                        )?,
                        _ => unreachable!("matrix dtype was matched above"),
                    }
                    self.op_barrier()?;
                    Ok(Some(output))
                }
                other => bail!(
                    "CUDA matrix {matrix_name} dtype {} cannot be cast to f32",
                    other.label()
                ),
            }
        }

        fn matrix_f32_host(&self, matrix_name: &str) -> Result<Vec<f32>> {
            let matrix = self
                .matrix(matrix_name)
                .ok_or_else(|| anyhow!("CUDA matrix {matrix_name} is missing"))?;
            let elements = matrix
                .rows
                .checked_mul(matrix.cols)
                .context("CUDA matrix host element count overflows usize")?;
            if matrix.is_quantized() {
                let dequantized = self
                    .dequantize_matrix_f32_device(matrix)
                    .with_context(|| format!("dequantizing CUDA matrix {matrix_name}"))?;
                return dequantized.copy_to_host(elements);
            }
            let bytes: Vec<u8> = matrix
                .buffer
                .copy_to_host(matrix.bytes)
                .with_context(|| format!("copying CUDA matrix {matrix_name} to host"))?;
            read_tensor_as_f32(&bytes, matrix.dtype, elements)
                .with_context(|| format!("reading CUDA matrix {matrix_name} as f32"))
        }

        pub fn rms_norm_f32_host(
            &self,
            vector_name: &str,
            input: &[f32],
            rows: usize,
            eps: f32,
        ) -> Result<Vec<f32>> {
            if rows == 0 {
                bail!("CUDA RMSNorm rows must be greater than zero");
            }
            let weight = self
                .vector(vector_name)
                .ok_or_else(|| anyhow!("CUDA vector {vector_name} is missing"))?;
            let expected = rows
                .checked_mul(weight.len)
                .context("CUDA RMSNorm input element count overflows usize")?;
            if input.len() != expected {
                bail!(
                    "CUDA RMSNorm input has {} elements; expected {rows} x {} = {expected}",
                    input.len(),
                    weight.len
                );
            }
            let input_device = DeviceBuffer::alloc(std::mem::size_of_val(input))
                .context("allocating CUDA RMSNorm input")?;
            input_device
                .copy_from_host(input)
                .context("copying CUDA RMSNorm input")?;
            let output = DeviceBuffer::alloc(std::mem::size_of_val(input))
                .context("allocating CUDA RMSNorm output")?;
            crate::kernels::launch_rms_norm(
                &input_device,
                &weight.buffer,
                &output,
                rows,
                weight.len,
                eps,
                &self.stream,
            )?;
            self.op_barrier()?;
            output.copy_to_host(input.len())
        }

        fn rms_norm_f32_device(
            &self,
            vector_name: &str,
            input: &GpuF32Tensor,
            eps: f32,
        ) -> Result<GpuF32Tensor> {
            let weight = self
                .vector(vector_name)
                .ok_or_else(|| anyhow!("CUDA vector {vector_name} is missing"))?;
            if input.cols != weight.len {
                bail!(
                    "CUDA RMSNorm input cols {} do not match vector length {} for {vector_name}",
                    input.cols,
                    weight.len
                );
            }
            let output_elements = input
                .rows
                .checked_mul(input.cols)
                .context("CUDA RMSNorm output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA RMSNorm output")?;
            crate::kernels::launch_rms_norm(
                &input.buffer,
                &weight.buffer,
                &output,
                input.rows,
                input.cols,
                eps,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        pub fn embed_tokens_host(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
            let tensor = self.embed_tokens_device(token_ids)?;
            self.op_barrier()?;
            tensor.copy_to_host()
        }

        fn embed_tokens_device(&self, token_ids: &[u32]) -> Result<GpuF32Tensor> {
            if token_ids.is_empty() {
                bail!("CUDA embedding lookup requires at least one token id");
            }
            let embeddings = self
                .matrix("token_embd.weight")
                .ok_or_else(|| anyhow!("CUDA token embedding matrix is missing"))?;
            for token_id in token_ids {
                let token_id = usize::try_from(*token_id).context("token id does not fit usize")?;
                if token_id >= embeddings.rows {
                    bail!(
                        "token id {token_id} is outside CUDA embedding vocab size {}",
                        embeddings.rows
                    );
                }
            }

            let ids = DeviceBuffer::alloc(std::mem::size_of_val(token_ids))
                .context("allocating CUDA embedding token ids")?;
            ids.copy_from_host(token_ids)
                .context("copying CUDA embedding token ids")?;
            let output_elements = token_ids
                .len()
                .checked_mul(embeddings.cols)
                .context("CUDA embedding output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA embedding output")?;
            match embeddings.dtype {
                GgufTensorType::F16 => crate::kernels::launch_gather_rows_f16_to_f32(
                    &embeddings.buffer,
                    &ids,
                    &output,
                    token_ids.len(),
                    embeddings.cols,
                    embeddings.rows,
                    &self.stream,
                )?,
                GgufTensorType::BF16 => crate::kernels::launch_gather_rows_bf16_to_f32(
                    &embeddings.buffer,
                    &ids,
                    &output,
                    token_ids.len(),
                    embeddings.cols,
                    embeddings.rows,
                    &self.stream,
                )?,
                GgufTensorType::F32 => crate::kernels::launch_gather_rows_f32_to_f32(
                    &embeddings.buffer,
                    &ids,
                    &output,
                    token_ids.len(),
                    embeddings.cols,
                    embeddings.rows,
                    &self.stream,
                )?,
                GgufTensorType::MXFP4
                | GgufTensorType::NVFP4
                | GgufTensorType::Q1_0
                | GgufTensorType::Q4_0
                | GgufTensorType::Q4_0_4_4
                | GgufTensorType::Q4_0_4_8
                | GgufTensorType::Q4_0_8_8
                | GgufTensorType::Q4_1
                | GgufTensorType::Q5_0
                | GgufTensorType::Q5_1
                | GgufTensorType::Q8_0
                | GgufTensorType::Q8_1
                | GgufTensorType::IQ2_XXS
                | GgufTensorType::IQ2_XS
                | GgufTensorType::IQ3_XXS
                | GgufTensorType::IQ1_S
                | GgufTensorType::IQ2_S
                | GgufTensorType::IQ3_S
                | GgufTensorType::IQ4_NL
                | GgufTensorType::IQ4_NL_4_4
                | GgufTensorType::IQ4_NL_4_8
                | GgufTensorType::IQ4_NL_8_8
                | GgufTensorType::IQ4_XS
                | GgufTensorType::IQ1_M
                | GgufTensorType::Q2_K
                | GgufTensorType::Q3_K
                | GgufTensorType::Q4_K
                | GgufTensorType::Q5_K
                | GgufTensorType::Q6_K
                | GgufTensorType::Q8_K
                | GgufTensorType::TQ1_0
                | GgufTensorType::TQ2_0 => {
                    let matrix = self.dequantize_matrix_f32_device(embeddings)?;
                    crate::kernels::launch_gather_rows_f32_to_f32(
                        &matrix,
                        &ids,
                        &output,
                        token_ids.len(),
                        embeddings.cols,
                        embeddings.rows,
                        &self.stream,
                    )?
                }
                GgufTensorType::I8
                | GgufTensorType::I16
                | GgufTensorType::I32
                | GgufTensorType::I64
                | GgufTensorType::F64 => bail!(
                    "CUDA embedding matrix has unsupported {} dtype",
                    embeddings.dtype.label()
                ),
            }
            self.op_barrier()?;
            let tensor = GpuF32Tensor {
                rows: token_ids.len(),
                cols: embeddings.cols,
                buffer: output,
            };
            if self.config.is_gemma() {
                // Gemma scales the token embeddings by sqrt(hidden_size) before the
                // first layer. The residual stream carries this scaled embedding, so
                // it must be applied (RMSNorm inside each layer would otherwise
                // normalize the un-scaled magnitude away).
                let scale = (embeddings.cols as f32).sqrt();
                let mut host = tensor.copy_to_host()?;
                for value in host.iter_mut() {
                    *value *= scale;
                }
                return self.f32_tensor_from_host(
                    &host,
                    tensor.rows,
                    tensor.cols,
                    "CUDA Gemma scaled embedding",
                );
            }
            Ok(tensor)
        }

        pub fn embed_norm_project_host(
            &self,
            token_ids: &[u32],
            norm_vector_name: &str,
            matrix_name: &str,
            eps: f32,
        ) -> Result<Vec<f32>> {
            let embeddings = self.embed_tokens_device(token_ids)?;
            let normed = self.rms_norm_f32_device(norm_vector_name, &embeddings, eps)?;
            let projected = self.project_f32_device(matrix_name, &normed)?;
            projected.copy_to_host()
        }

        pub fn full_context_logits_host(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
            let logits = self.full_context_logits_device(token_ids)?;
            self.op_barrier()?;
            logits.copy_to_host()
        }

        pub fn last_logits_host(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
            let logits = self.full_context_logits_device(token_ids)?;
            self.op_barrier()?;
            let all = logits.copy_to_host()?;
            let start = (logits.rows - 1)
                .checked_mul(logits.cols)
                .context("CUDA logits row offset overflows usize")?;
            Ok(all[start..start + logits.cols].to_vec())
        }

        pub fn greedy_next_token(&self, token_ids: &[u32]) -> Result<u32> {
            let logits = self.full_context_logits_device(token_ids)?;
            self.argmax_last_row(&logits)
        }

        pub fn generate_greedy_tokens(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                &[],
            )
        }

        pub fn generate_greedy_tokens_with_stop_ids(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            let stop_token_sequences = stop_sequences_from_stop_ids(stop_token_ids);
            self.generate_greedy_tokens_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                &stop_token_sequences,
            )
        }

        pub fn generate_greedy_tokens_with_stop_sequences(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_with_stop_sequences_and_cancellation(
                input_ids,
                max_tokens,
                eos_token_id,
                stop_token_sequences,
                || false,
            )
        }

        pub fn generate_greedy_tokens_with_stop_sequences_and_cancellation<F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens("CUDA greedy generation", max_tokens)?;
            if input_ids.is_empty() {
                bail!("CUDA greedy generation requires at least one input token");
            }
            if input_ids.len() > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    input_ids.len(),
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA greedy generation",
                input_ids.len(),
                max_tokens,
                dims.context,
            )?;

            if self.config.recurrent_ssm_tensor_layout {
                return self.generate_greedy_tokens_full_context_with_cancellation(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    stop_token_sequences,
                    is_cancelled,
                );
            }

            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits =
                self.full_context_logits_device_with_cache(input_ids, Some(&mut cache))?;
            let mut generated = Vec::new();
            let mut next_position = input_ids.len();

            for step in 0..max_tokens {
                if is_cancelled() {
                    break;
                }
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        fn generate_greedy_tokens_full_context_with_cancellation<F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            let dims = self.qwen_dims()?;
            let mut context = input_ids.to_vec();
            let mut generated = Vec::new();
            for _ in 0..max_tokens {
                if is_cancelled() {
                    break;
                }
                let logits = self.full_context_logits_device(&context)?;
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if context.len() >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        context.len() + 1,
                        dims.context
                    );
                }
                context.push(next);
            }
            Ok(generated)
        }

        pub fn generate_greedy_tokens_paged(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_paged_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                page_size,
                &[],
            )
        }

        pub fn generate_greedy_tokens_paged_with_stop_ids(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            let stop_token_sequences = stop_sequences_from_stop_ids(stop_token_ids);
            self.generate_greedy_tokens_paged_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                page_size,
                &stop_token_sequences,
            )
        }

        pub fn generate_greedy_tokens_paged_with_stop_sequences(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_paged_with_stop_sequences_and_cancellation(
                input_ids,
                max_tokens,
                eos_token_id,
                page_size,
                stop_token_sequences,
                || false,
            )
        }

        pub fn generate_greedy_tokens_paged_with_stop_sequences_and_cancellation<F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            let dims = self.qwen_dims()?;
            let max_tokens =
                validate_generation_max_tokens("CUDA paged greedy generation", max_tokens)?;
            if input_ids.is_empty() {
                bail!("CUDA paged greedy generation requires at least one input token");
            }
            if input_ids.len() > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    input_ids.len(),
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged greedy generation",
                input_ids.len(),
                max_tokens,
                dims.context,
            )?;

            if self.config.recurrent_ssm_tensor_layout {
                return self.generate_greedy_tokens_full_context_with_cancellation(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    stop_token_sequences,
                    is_cancelled,
                );
            }

            let token_capacity = input_ids
                .len()
                .checked_add(max_tokens)
                .context("CUDA paged greedy generation token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits =
                self.full_context_logits_device_with_paged_cache(input_ids, &mut cache)?;
            let mut generated = Vec::new();
            let mut next_position = input_ids.len();

            for step in 0..max_tokens {
                if is_cancelled() {
                    break;
                }
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_with_prefix_embeddings(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_with_prefix_embeddings_and_stop_sequences(
                prefix_embeddings,
                prefix_rows,
                input_ids,
                max_tokens,
                eos_token_id,
                &[],
            )
        }

        pub fn generate_greedy_tokens_with_prefix_embeddings_and_stop_sequences(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens =
                validate_generation_max_tokens("CUDA greedy multimodal generation", max_tokens)?;
            if input_ids.is_empty() {
                bail!("CUDA greedy multimodal generation requires at least one input token");
            }
            let input_len = prefix_rows
                .checked_add(input_ids.len())
                .context("CUDA multimodal input length overflows usize")?;
            if input_len > dims.context {
                bail!(
                    "multimodal input length {input_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            validate_batched_generation_context_budget(
                "CUDA greedy multimodal generation",
                input_len,
                max_tokens,
                dims.context,
            )?;

            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits = self.full_context_logits_device_with_prefix_cache(
                prefix_embeddings,
                prefix_rows,
                input_ids,
                Some(&mut cache),
            )?;
            let mut generated = Vec::new();
            let mut next_position = input_len;

            for step in 0..max_tokens {
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_with_prompt_embeddings_and_stop_sequences(
                prompt_embeddings,
                prompt_rows,
                max_tokens,
                eos_token_id,
                &[],
            )
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA greedy multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA greedy multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA multimodal prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits =
                self.full_context_logits_from_hidden_device(hidden, Some(&mut cache))?;
            let mut generated = Vec::new();
            let mut next_position = prompt_rows;

            for step in 0..max_tokens {
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings_and_positions(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            self.generate_greedy_tokens_with_prompt_embeddings_and_positions_and_stop_sequences(
                prompt_embeddings,
                prompt_rows,
                position_ids,
                next_rope_position,
                max_tokens,
                eos_token_id,
                &[],
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_with_prompt_embeddings_and_positions_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA greedy multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            if next_rope_position > dims.context {
                bail!(
                    "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA greedy multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA multimodal prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits = self.full_context_logits_from_hidden_device_with_position_ids(
                hidden,
                position_ids,
                Some(&mut cache),
            )?;
            let mut generated = Vec::new();
            let mut next_cache_position = prompt_rows;
            let mut next_rope_position = next_rope_position;

            for step in 0..max_tokens {
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_cache_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_cache_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device_with_rope_position(
                    next,
                    next_cache_position,
                    next_rope_position,
                    &mut cache,
                )?;
                next_cache_position += 1;
                next_rope_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_with_prefix_embeddings_paged_and_stop_sequences(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA paged greedy multimodal generation",
                max_tokens,
            )?;
            if input_ids.is_empty() {
                bail!("CUDA paged greedy multimodal generation requires at least one input token");
            }
            let input_len = prefix_rows
                .checked_add(input_ids.len())
                .context("CUDA paged multimodal input length overflows usize")?;
            if input_len > dims.context {
                bail!(
                    "multimodal input length {input_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            validate_batched_generation_context_budget(
                "CUDA paged greedy multimodal generation",
                input_len,
                max_tokens,
                dims.context,
            )?;

            let token_capacity = input_len
                .checked_add(max_tokens)
                .context("CUDA paged greedy multimodal token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_device_with_paged_prefix_cache(
                prefix_embeddings,
                prefix_rows,
                input_ids,
                &mut cache,
            )?;
            let mut generated = Vec::new();
            let mut next_position = input_len;

            for step in 0..max_tokens {
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings_paged_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA paged greedy multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged greedy multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA paged multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_tokens)
                .context("CUDA paged greedy multimodal prompt token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits =
                self.full_context_logits_from_hidden_device_with_paged_cache(hidden, &mut cache)?;
            let mut generated = Vec::new();
            let mut next_position = prompt_rows;

            for step in 0..max_tokens {
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_with_prompt_embeddings_and_positions_paged_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA paged greedy multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            if next_rope_position > dims.context {
                bail!(
                    "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged greedy multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA paged multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_tokens)
                .context("CUDA paged greedy multimodal prompt token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits = self
                .full_context_logits_from_hidden_device_with_position_ids_paged_cache(
                    hidden,
                    position_ids,
                    &mut cache,
                )?;
            let mut generated = Vec::new();
            let mut next_cache_position = prompt_rows;
            let mut next_rope_position = next_rope_position;

            for step in 0..max_tokens {
                let next = self.argmax_last_row(&logits)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_cache_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_cache_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device_with_rope_position(
                    next,
                    next_cache_position,
                    next_rope_position,
                    &mut cache,
                )?;
                next_cache_position += 1;
                next_rope_position += 1;
            }

            Ok(generated)
        }

        pub fn supports_batched_text_generation(&self) -> bool {
            if self.config.recurrent_ssm_tensor_layout {
                return true;
            }
            if self.config.expert_count.is_some() {
                self.config.expert_used_count.is_some()
                    && self.has_matrix("blk.0.ffn_gate_inp.weight")
            } else {
                self.has_matrix("blk.0.ffn_gate.weight")
                    && self.has_matrix("blk.0.ffn_up.weight")
                    && self.has_matrix("blk.0.ffn_down.weight")
            }
        }

        pub fn supports_batched_multimodal_generation(&self) -> bool {
            !self.config.recurrent_ssm_tensor_layout && self.supports_batched_text_generation()
        }

        fn validate_batched_text_generation_request(
            &self,
            label: &str,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            seeds: Option<&[Option<u64>]>,
        ) -> Result<(QwenDims, usize, usize, usize)> {
            if inputs.is_empty() {
                bail!("{label} requires at least one request");
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                label,
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "{label} got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if let Some(seeds) = seeds {
                if seeds.len() != batch_count {
                    bail!(
                        "{label} got {} seeds for {batch_count} requests",
                        seeds.len()
                    );
                }
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("{label} requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("{label} requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("{label} requires equal prompt lengths");
            }
            validate_batched_generation_context_budget(
                label,
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;
            Ok((dims, batch_count, prompt_len, max_decode_steps))
        }

        fn generate_recurrent_ssm_greedy_batch_full_context<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            let mut generated = Vec::with_capacity(inputs.len());
            for (idx, input) in inputs.iter().enumerate() {
                let stop_sequences = stop_token_sequences_per_request
                    .get(idx)
                    .map(|sequences| sequences.as_slice())
                    .unwrap_or(&[]);
                generated.push(
                    self.generate_greedy_tokens_with_stop_sequences_and_cancellation(
                        input,
                        max_tokens_per_request[idx],
                        eos_token_id,
                        stop_sequences,
                        || is_cancelled(idx),
                    )?,
                );
            }
            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_recurrent_ssm_sampled_batch_full_context<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            let mut generated = Vec::with_capacity(inputs.len());
            for (idx, input) in inputs.iter().enumerate() {
                let stop_sequences = stop_token_sequences_per_request
                    .get(idx)
                    .map(|sequences| sequences.as_slice())
                    .unwrap_or(&[]);
                generated.push(
                    self.generate_sampled_tokens_with_stop_sequences_and_cancellation(
                        input,
                        max_tokens_per_request[idx],
                        eos_token_id,
                        temperature,
                        top_p,
                        top_k,
                        seeds[idx],
                        stop_sequences,
                        || is_cancelled(idx),
                    )?,
                );
            }
            Ok(generated)
        }

        pub fn generate_greedy_tokens_batch(
            &self,
            inputs: &[Vec<u32>],
            max_tokens: usize,
            eos_token_id: Option<u32>,
        ) -> Result<Vec<Vec<u32>>> {
            let max_tokens =
                validate_generation_max_tokens("CUDA batched greedy generation", max_tokens)?;
            let max_tokens_per_request = vec![max_tokens; inputs.len()];
            self.generate_greedy_tokens_batch_with_limits(
                inputs,
                &max_tokens_per_request,
                eos_token_id,
            )
        }

        pub fn generate_greedy_tokens_batch_with_limits(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
        ) -> Result<Vec<Vec<u32>>> {
            self.generate_greedy_tokens_batch_with_limits_and_cancellation(
                inputs,
                max_tokens_per_request,
                eos_token_id,
                &[],
                |_| false,
            )
        }

        pub fn generate_greedy_tokens_batch_with_limits_and_cancellation<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!("CUDA batched greedy generation requires at least one request");
            }
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_batched_text_generation_request(
                    "CUDA batched greedy generation",
                    inputs,
                    max_tokens_per_request,
                    stop_token_sequences_per_request,
                    None,
                )?;
                return self.generate_recurrent_ssm_greedy_batch_full_context(
                    inputs,
                    max_tokens_per_request,
                    eos_token_id,
                    stop_token_sequences_per_request,
                    is_cancelled,
                );
            }
            if !self.supports_batched_text_generation() {
                bail!(
                    "CUDA batched greedy generation currently supports loaded decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA batched greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA batched greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("CUDA batched greedy generation requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("CUDA batched greedy generation requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("CUDA batched greedy generation requires equal prompt lengths");
            }
            validate_batched_generation_context_budget(
                "CUDA batched greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let mut flat = Vec::with_capacity(batch_count * prompt_len);
            for input in inputs {
                flat.extend_from_slice(input);
            }
            let mut cache = CudaKvCache::new_batched(
                self.config.block_count,
                &dims,
                batch_count,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_device_batched(
                &flat,
                batch_count,
                prompt_len,
                Some(&mut cache),
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = self.argmax_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_and_limits_and_cancellation<F>(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!("CUDA batched multimodal greedy generation requires at least one request");
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA batched multimodal greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA batched multimodal greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal greedy generation got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("CUDA batched multimodal greedy generation requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!("CUDA batched multimodal greedy generation requires non-empty prompts");
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!("CUDA batched multimodal greedy generation requires equal prompt lengths");
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows
                .checked_add(token_prompt_len)
                .context("CUDA batched multimodal prompt length overflows usize")?;
            if prompt_len > dims.context {
                bail!(
                    "multimodal input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA batched multimodal greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count
                    .checked_mul(prompt_len)
                    .context("CUDA batched multimodal hidden row count overflows usize")?,
                dims.embed,
                "CUDA batched multimodal prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new_batched(
                self.config.block_count,
                &dims,
                batch_count,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_from_hidden_device_batched(
                hidden,
                batch_count,
                prompt_len,
                Some(&mut cache),
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = self.argmax_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA batched multimodal MRoPE greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA batched multimodal MRoPE greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA batched multimodal MRoPE greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if position_ids_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE greedy generation got {} position rows for {batch_count} requests",
                    position_ids_per_request.len()
                );
            }
            if next_rope_positions.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE greedy generation got {} next RoPE positions for {batch_count} requests",
                    next_rope_positions.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA batched multimodal MRoPE greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 {
                bail!("CUDA batched multimodal MRoPE greedy generation requires non-empty prompts");
            }
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA batched multimodal MRoPE greedy generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context("CUDA batched multimodal MRoPE hidden value count overflows usize")?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA batched multimodal MRoPE greedy generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            if hidden_host.len() != total_values {
                bail!(
                    "CUDA batched multimodal MRoPE hidden build produced {} values; expected {total_values}",
                    hidden_host.len()
                );
            }

            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count
                    .checked_mul(prompt_rows)
                    .context("CUDA batched multimodal MRoPE hidden row count overflows usize")?,
                dims.embed,
                "CUDA batched multimodal MRoPE prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new_batched(
                self.config.block_count,
                &dims,
                batch_count,
                &self.stream,
            )?;
            let mut logits = self
                .full_context_logits_from_hidden_device_batched_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    Some(&mut cache),
                )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_cache_position = prompt_rows;
            let mut next_rope_positions = next_rope_positions.to_vec();

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = self.argmax_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_rows } else { 1 },
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_cache_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_cache_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits = self.decode_batch_logits_device_with_rope_positions(
                    &next_tokens,
                    next_cache_position,
                    &next_rope_positions,
                    &mut cache,
                )?;
                next_cache_position += 1;
                for position in &mut next_rope_positions {
                    *position = position
                        .checked_add(1)
                        .context("next MRoPE decode position overflows usize")?;
                }
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_batched_greedy_from_logits_with_cancellation<F, D>(
            &self,
            mut logits: GpuF32Tensor,
            batch_count: usize,
            prompt_len: usize,
            max_decode_steps: usize,
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
            mut decode_next: D,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
            D: FnMut(&[u32], usize) -> Result<GpuF32Tensor>,
        {
            let dims = self.qwen_dims()?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = self.argmax_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits = decode_next(&next_tokens, next_position)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_batched_sampled_from_logits_with_cancellation<F, D>(
            &self,
            mut logits: GpuF32Tensor,
            batch_count: usize,
            prompt_len: usize,
            max_decode_steps: usize,
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            mut rngs: Vec<StdRng>,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
            mut decode_next: D,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
            D: FnMut(&[u32], usize) -> Result<GpuF32Tensor>,
        {
            if rngs.len() != batch_count {
                bail!(
                    "CUDA batched sampled generation got {} RNGs for {batch_count} request(s)",
                    rngs.len()
                );
            }
            let dims = self.qwen_dims()?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let samples = rngs
                    .iter_mut()
                    .map(|rng| rng.gen_range(0.0f32..1.0f32))
                    .collect::<Vec<_>>();
                let batch_tokens = self.sample_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                    temperature,
                    top_p,
                    top_k,
                    &samples,
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits = decode_next(&next_tokens, next_position)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA paged batched multimodal greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA paged batched multimodal greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA paged batched multimodal greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal greedy generation got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA paged batched multimodal greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!("CUDA paged batched multimodal greedy generation requires non-empty prompts");
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!(
                    "CUDA paged batched multimodal greedy generation requires equal prompt lengths"
                );
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows
                .checked_add(token_prompt_len)
                .context("CUDA paged batched multimodal prompt length overflows usize")?;
            if prompt_len > dims.context {
                bail!(
                    "multimodal input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged batched multimodal greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count
                    .checked_mul(prompt_len)
                    .context("CUDA paged batched multimodal hidden row count overflows usize")?,
                dims.embed,
                "CUDA paged batched multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context("CUDA paged batched multimodal greedy token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedBatchKvCache::new(
                self.config.block_count,
                &dims,
                batch_count,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let logits = self.full_context_logits_from_hidden_device_batched_paged_cache(
                hidden,
                batch_count,
                prompt_len,
                &mut cache,
            )?;

            self.generate_batched_greedy_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_len,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_position| {
                    self.decode_batch_logits_paged_device(next_tokens, next_position, &mut cache)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA paged batched multimodal MRoPE greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if position_ids_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation got {} position rows for {batch_count} requests",
                    position_ids_per_request.len()
                );
            }
            if next_rope_positions.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation got {} next RoPE positions for {batch_count} requests",
                    next_rope_positions.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 {
                bail!(
                    "CUDA paged batched multimodal MRoPE greedy generation requires non-empty prompts"
                );
            }
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged batched multimodal MRoPE greedy generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context(
                    "CUDA paged batched multimodal MRoPE hidden value count overflows usize",
                )?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA paged batched multimodal MRoPE greedy generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            if hidden_host.len() != total_values {
                bail!(
                    "CUDA paged batched multimodal MRoPE hidden build produced {} values; expected {total_values}",
                    hidden_host.len()
                );
            }

            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_rows).context(
                    "CUDA paged batched multimodal MRoPE hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA paged batched multimodal MRoPE prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_decode_steps)
                .context(
                    "CUDA paged batched multimodal MRoPE greedy token capacity overflows usize",
                )?
                .min(dims.context);
            let mut cache = CudaPagedBatchKvCache::new(
                self.config.block_count,
                &dims,
                batch_count,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let logits = self
                .full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    &mut cache,
                )?;
            let mut next_rope_positions = next_rope_positions.to_vec();

            self.generate_batched_greedy_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_rows,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_cache_position| {
                    let decoded = self.decode_batch_logits_paged_device_with_rope_positions(
                        next_tokens,
                        next_cache_position,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                    for position in &mut next_rope_positions {
                        *position = position
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                    Ok(decoded)
                },
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_limits(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
        ) -> Result<Vec<Vec<u32>>> {
            self.generate_greedy_tokens_batch_paged_with_limits_and_cancellation(
                inputs,
                max_tokens_per_request,
                eos_token_id,
                page_size,
                &[],
                |_| false,
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_limits_and_cancellation<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!("CUDA paged batched greedy generation requires at least one request");
            }
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_batched_text_generation_request(
                    "CUDA paged batched greedy generation",
                    inputs,
                    max_tokens_per_request,
                    stop_token_sequences_per_request,
                    None,
                )?;
                return self.generate_recurrent_ssm_greedy_batch_full_context(
                    inputs,
                    max_tokens_per_request,
                    eos_token_id,
                    stop_token_sequences_per_request,
                    is_cancelled,
                );
            }
            if !self.supports_batched_text_generation() {
                bail!(
                    "CUDA paged batched greedy generation currently supports loaded decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA paged batched greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("CUDA paged batched greedy generation requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("CUDA paged batched greedy generation requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("CUDA paged batched greedy generation requires equal prompt lengths");
            }
            validate_batched_generation_context_budget(
                "CUDA paged batched greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let mut flat = Vec::with_capacity(batch_count * prompt_len);
            for input in inputs {
                flat.extend_from_slice(input);
            }
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context("CUDA paged batched greedy token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedBatchKvCache::new(
                self.config.block_count,
                &dims,
                batch_count,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_device_batched_paged_cache(
                &flat,
                batch_count,
                prompt_len,
                &mut cache,
                false,
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = self.argmax_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_paged_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_greedy_tokens_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<Vec<u32>>> {
            self.generate_greedy_tokens_batch_paged_with_page_tables_and_cancellation(
                inputs,
                max_tokens_per_request,
                eos_token_id,
                page_size,
                page_tables,
                physical_page_count,
                &[],
                |_| false,
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_page_tables_and_cancellation<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched greedy generation requires at least one request"
                );
            }
            if self.config.recurrent_ssm_tensor_layout {
                let (dims, batch_count, prompt_len, max_decode_steps) = self
                    .validate_batched_text_generation_request(
                        "CUDA lease-backed paged batched greedy generation",
                        inputs,
                        max_tokens_per_request,
                        stop_token_sequences_per_request,
                        None,
                    )?;
                let token_capacity = prompt_len
                    .checked_add(max_decode_steps)
                    .context(
                        "CUDA lease-backed paged batched greedy token capacity overflows usize",
                    )?
                    .min(dims.context);
                self.validate_page_tables_for_token_capacity(
                    "CUDA lease-backed paged batched greedy generation",
                    batch_count,
                    page_size,
                    token_capacity,
                    page_tables,
                    physical_page_count,
                    dims.context,
                )?;
                return self.generate_recurrent_ssm_greedy_batch_full_context(
                    inputs,
                    max_tokens_per_request,
                    eos_token_id,
                    stop_token_sequences_per_request,
                    is_cancelled,
                );
            }
            if !self.supports_batched_text_generation() {
                bail!(
                    "CUDA lease-backed paged batched greedy generation currently supports loaded decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!(
                    "CUDA lease-backed paged batched greedy generation requires non-empty prompts"
                );
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!(
                    "CUDA lease-backed paged batched greedy generation requires equal prompt lengths"
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context("CUDA lease-backed paged batched greedy token capacity overflows usize")?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let (mut cache, mut logits, initial_logits_seq_len) = self
                .full_context_logits_device_batched_paged_cache_with_shared_prefix(
                    inputs,
                    page_size,
                    token_capacity,
                    page_tables,
                    pool,
                )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = self.argmax_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { initial_logits_seq_len } else { 1 },
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_paged_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn greedy_next_tokens_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_page_table_context_batch(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged greedy decode",
                )?;
                self.remember_recurrent_page_contexts(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged greedy decode",
                )?;
                return self.recurrent_ssm_greedy_next_tokens_full_context(inputs);
            }
            let (logits, seq_len) = self.next_token_logits_batch_paged_with_page_tables(
                inputs,
                page_size,
                page_tables,
                physical_page_count,
                "CUDA continuous paged greedy decode",
            )?;
            self.argmax_batched_last_token(&logits, inputs.len(), seq_len)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn sampled_next_tokens_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            samples: &[f32],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_page_table_context_batch(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged sampled decode",
                )?;
                self.remember_recurrent_page_contexts(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged sampled decode",
                )?;
                return self.recurrent_ssm_sampled_next_tokens_full_context(
                    inputs,
                    temperature,
                    top_p,
                    top_k,
                    samples,
                );
            }
            let (logits, seq_len) = self.next_token_logits_batch_paged_with_page_tables(
                inputs,
                page_size,
                page_tables,
                physical_page_count,
                "CUDA continuous paged sampled decode",
            )?;
            self.sample_batched_last_token(
                &logits,
                inputs.len(),
                seq_len,
                temperature,
                top_p,
                top_k,
                samples,
            )
        }

        pub fn prefill_greedy_next_tokens_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_page_table_context_batch(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged greedy prefill",
                )?;
                self.remember_recurrent_page_contexts(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged greedy prefill",
                )?;
                return self.recurrent_ssm_greedy_next_tokens_full_context(inputs);
            }
            let (logits, seq_len) = self.prefill_logits_batch_paged_with_page_tables(
                inputs,
                page_size,
                page_tables,
                physical_page_count,
                "CUDA continuous paged greedy prefill",
            )?;
            self.argmax_batched_last_token(&logits, inputs.len(), seq_len)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn prefill_sampled_next_tokens_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            samples: &[f32],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_page_table_context_batch(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged sampled prefill",
                )?;
                self.remember_recurrent_page_contexts(
                    inputs,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged sampled prefill",
                )?;
                return self.recurrent_ssm_sampled_next_tokens_full_context(
                    inputs,
                    temperature,
                    top_p,
                    top_k,
                    samples,
                );
            }
            let (logits, seq_len) = self.prefill_logits_batch_paged_with_page_tables(
                inputs,
                page_size,
                page_tables,
                physical_page_count,
                "CUDA continuous paged sampled prefill",
            )?;
            self.sample_batched_last_token(
                &logits,
                inputs.len(),
                seq_len,
                temperature,
                top_p,
                top_k,
                samples,
            )
        }

        pub fn decode_greedy_next_tokens_batch_paged_with_page_tables(
            &self,
            token_ids: &[u32],
            position: usize,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            if self.config.recurrent_ssm_tensor_layout {
                return self.recurrent_ssm_decode_next_tokens_from_states(
                    token_ids,
                    position,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged greedy append decode",
                    |model, logits, _| model.argmax_last_row(logits),
                );
            }
            let logits = self.decode_logits_batch_paged_with_page_tables(
                token_ids,
                position,
                page_size,
                page_tables,
                physical_page_count,
                "CUDA continuous paged greedy append decode",
            )?;
            self.argmax_batched_last_token(&logits, token_ids.len(), 1)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn decode_sampled_next_tokens_batch_paged_with_page_tables(
            &self,
            token_ids: &[u32],
            position: usize,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            samples: &[f32],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            if self.config.recurrent_ssm_tensor_layout {
                if samples.len() != token_ids.len() {
                    bail!(
                        "CUDA continuous paged sampled append decode got {} random samples for {} requests",
                        samples.len(),
                        token_ids.len()
                    );
                }
                return self.recurrent_ssm_decode_next_tokens_from_states(
                    token_ids,
                    position,
                    page_size,
                    page_tables,
                    physical_page_count,
                    "CUDA continuous paged sampled append decode",
                    |model, logits, idx| {
                        model.sample_last_row_with_sample(
                            logits,
                            temperature,
                            top_p,
                            top_k,
                            samples[idx],
                        )
                    },
                );
            }
            let logits = self.decode_logits_batch_paged_with_page_tables(
                token_ids,
                position,
                page_size,
                page_tables,
                physical_page_count,
                "CUDA continuous paged sampled append decode",
            )?;
            self.sample_batched_last_token(
                &logits,
                token_ids.len(),
                1,
                temperature,
                top_p,
                top_k,
                samples,
            )
        }

        fn prefill_logits_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            label: &str,
        ) -> Result<(GpuF32Tensor, usize)> {
            if inputs.is_empty() {
                bail!("{label} requires at least one request");
            }
            if !self.supports_batched_text_generation() {
                bail!("{label} currently supports loaded decoder model layouts only");
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            if page_tables.len() != batch_count {
                bail!(
                    "{label} got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("{label} requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("{label} requires equal prompt lengths");
            }

            let token_capacity = prompt_len.min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            let (_cache, logits, logits_seq_len) = self
                .full_context_logits_device_batched_paged_cache_with_shared_prefix(
                    inputs,
                    page_size,
                    token_capacity,
                    page_tables,
                    pool,
                )?;
            Ok((logits, logits_seq_len))
        }

        /// Prefill only the divergent suffix `input_ids[reuse_tokens..]`, reusing
        /// KV already resident in the first `reuse_tokens` positions of
        /// `page_table` (written by a prior request whose prompt shared this
        /// prefix). This mirrors the suffix loop of
        /// `full_context_logits_device_batched_paged_cache_with_shared_prefix`,
        /// except the prefix KV is pre-populated in the pages rather than
        /// recomputed here — so an agent loop pays O(suffix) instead of
        /// O(prompt) each turn. Single sequence (batch of one) only.
        pub fn prefill_greedy_next_tokens_paged_reusing_prefix(
            &self,
            input_ids: &[u32],
            reuse_tokens: usize,
            page_size: usize,
            page_table: &[usize],
            physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            let label = "CUDA paged greedy prefix-reuse prefill";
            if self.config.recurrent_ssm_tensor_layout {
                bail!("{label} is not supported for recurrent SSM layouts");
            }
            if !self.supports_batched_text_generation() {
                bail!("{label} currently supports loaded decoder model layouts only");
            }
            let dims = self.qwen_dims()?;
            let prompt_len = input_ids.len();
            if reuse_tokens == 0 || reuse_tokens >= prompt_len {
                bail!(
                    "{label} requires 0 < reuse_tokens ({reuse_tokens}) < prompt_len ({prompt_len})"
                );
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }

            let token_capacity = prompt_len.min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;

            let page_tables = vec![page_table.to_vec()];
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                1,
                page_size,
                token_capacity,
                &page_tables,
                pool,
                &self.stream,
            )?;
            // Append the suffix one position at a time, attending to the reused
            // prefix KV through the page table (paged attention reads physical
            // pages by index, regardless of which request wrote them). RoPE
            // positions stay consistent because the reused K was stored with its
            // original position's rotation.
            let mut logits = None;
            for position in reuse_tokens..prompt_len {
                logits = Some(self.decode_batch_logits_paged_device(
                    std::slice::from_ref(&input_ids[position]),
                    position,
                    &mut cache,
                )?);
            }
            let logits = logits.ok_or_else(|| anyhow!("{label} produced no suffix logits"))?;
            self.argmax_batched_last_token(&logits, 1, 1)
        }

        fn decode_logits_batch_paged_with_page_tables(
            &self,
            token_ids: &[u32],
            position: usize,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            label: &str,
        ) -> Result<GpuF32Tensor> {
            if token_ids.is_empty() {
                bail!("{label} requires at least one request");
            }
            if self.config.recurrent_ssm_tensor_layout {
                bail!(
                    "{label} requires a recurrent-state cache for Qwen recurrent SSM; full-context prefill/next-token generation is supported, but append-only decode has only token ids and cannot reconstruct ssm_conv1d/ssm_dt/ssm_a/ssm_ba/ssm_norm/ssm_out state"
                );
            }
            if !self.supports_batched_text_generation() {
                bail!("{label} currently supports loaded decoder model layouts only");
            }
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if page_tables.len() != batch_count {
                bail!(
                    "{label} got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            if position >= dims.context {
                bail!(
                    "decode position {position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let token_capacity = position
                .checked_add(1)
                .context("CUDA continuous paged append decode token capacity overflows usize")?;
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            self.decode_batch_logits_paged_device(token_ids, position, &mut cache)
        }

        fn next_token_logits_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            label: &str,
        ) -> Result<(GpuF32Tensor, usize)> {
            if inputs.is_empty() {
                bail!("{label} requires at least one request");
            }
            if !self.supports_batched_text_generation() {
                bail!("{label} currently supports loaded decoder model layouts only");
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            if page_tables.len() != batch_count {
                bail!(
                    "{label} got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("{label} requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("{label} requires equal context lengths");
            }

            let token_capacity = prompt_len.min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let (_cache, logits, logits_seq_len) = self
                .full_context_logits_device_batched_paged_cache_with_shared_prefix(
                    inputs,
                    page_size,
                    token_capacity,
                    page_tables,
                    pool,
                )?;
            Ok((logits, logits_seq_len))
        }

        fn validate_page_table_context_batch(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            label: &str,
        ) -> Result<(QwenDims, usize, usize)> {
            if inputs.is_empty() {
                bail!("{label} requires at least one request");
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            if page_tables.len() != batch_count {
                bail!(
                    "{label} got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("{label} requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("{label} requires equal context lengths");
            }
            self.validate_page_tables_for_token_capacity(
                label,
                batch_count,
                page_size,
                prompt_len,
                page_tables,
                physical_page_count,
                dims.context,
            )?;
            Ok((dims, batch_count, prompt_len))
        }

        #[allow(clippy::too_many_arguments)]
        fn validate_page_tables_for_token_capacity(
            &self,
            label: &str,
            batch_count: usize,
            page_size: usize,
            token_capacity: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            context: usize,
        ) -> Result<()> {
            if page_size == 0 {
                bail!("{label} page_size must be greater than zero");
            }
            if token_capacity == 0 {
                bail!("{label} token capacity must be greater than zero");
            }
            if token_capacity > context {
                bail!(
                    "{label} token capacity {token_capacity} exceeds qwen context length {context}"
                );
            }
            let page_table_len = token_capacity.div_ceil(page_size);
            if page_table_len == 0 {
                bail!("{label} page table must contain at least one page");
            }
            for (batch_idx, pages) in page_tables.iter().enumerate().take(batch_count) {
                if pages.len() < page_table_len {
                    bail!(
                        "{label} page table {batch_idx} has {} page(s), expected at least {page_table_len}",
                        pages.len()
                    );
                }
                for page in pages.iter().take(page_table_len).copied() {
                    if page >= physical_page_count {
                        bail!(
                            "{label} page index {page} exceeds physical page count {physical_page_count}"
                        );
                    }
                    let _ = u32::try_from(page)
                        .with_context(|| format!("{label} page index does not fit u32"))?;
                }
            }
            Ok(())
        }

        fn recurrent_page_context_key(&self, label: &str, page_table: &[usize]) -> Result<usize> {
            page_table.first().copied().ok_or_else(|| {
                anyhow!("{label} recurrent page table must contain at least one page")
            })
        }

        fn recurrent_ssm_attention_layer_count(&self) -> usize {
            (0..self.config.block_count)
                .filter(|layer| {
                    let prefix = format!("blk.{layer}");
                    !self.layer_uses_recurrent_ssm(&prefix)
                })
                .count()
        }

        fn recurrent_ssm_prepare_paged_pool(
            &self,
            dims: &QwenDims,
            page_size: usize,
            physical_page_count: usize,
        ) -> Result<RefMut<'_, Option<CudaPagedBatchDevicePool>>> {
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            Ok(pool_slot)
        }

        fn remember_recurrent_page_contexts(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            label: &str,
        ) -> Result<()> {
            if page_tables.len() != inputs.len() {
                bail!(
                    "{label} got {} page table(s) for {} recurrent context(s)",
                    page_tables.len(),
                    inputs.len()
                );
            }
            let dims = self.qwen_dims()?;
            let attention_layers = self.recurrent_ssm_attention_layer_count();
            let pool_guard = if attention_layers > 0 {
                let token_capacity = inputs.iter().map(Vec::len).max().unwrap_or(0).max(1);
                Some(self.recurrent_ssm_prepare_paged_pool(
                    &dims,
                    page_size,
                    physical_page_count,
                )?)
                .map(|guard| (guard, token_capacity))
            } else {
                None
            };
            let mut prepared = Vec::with_capacity(inputs.len());
            for (idx, (input, page_table)) in inputs.iter().zip(page_tables).enumerate() {
                let key = self.recurrent_page_context_key(label, page_table)?;
                let mut state = RecurrentSsmRequestState::new(key);
                if let Some((pool_guard, token_capacity)) = pool_guard.as_ref() {
                    let request_page_tables = [page_table.clone()];
                    let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                        &dims,
                        1,
                        page_size,
                        *token_capacity,
                        &request_page_tables,
                        pool_guard.as_ref().ok_or_else(|| {
                            anyhow!("{label} recurrent SSM paged pool is missing")
                        })?,
                        &self.stream,
                    )
                    .with_context(|| {
                        format!(
                            "{label} building recurrent SSM attention KV cache for request {idx}"
                        )
                    })?;
                    self.prefill_recurrent_ssm_state(input, &mut state, Some(&mut cache), label)?;
                } else {
                    self.prefill_recurrent_ssm_state(input, &mut state, None, label)?;
                }
                prepared.push((key, state));
            }
            drop(pool_guard);
            let mut states = self.recurrent_page_states.borrow_mut();
            for (key, state) in prepared {
                states.insert(key, state);
            }
            Ok(())
        }

        pub fn forget_recurrent_page_contexts(
            &self,
            page_tables: &[Vec<usize>],
            label: &str,
        ) -> Result<usize> {
            let mut states = self.recurrent_page_states.borrow_mut();
            let mut removed = 0usize;
            for page_table in page_tables {
                let key = self.recurrent_page_context_key(label, page_table)?;
                if states.remove(&key).is_some() {
                    removed = removed.saturating_add(1);
                }
            }
            Ok(removed)
        }

        pub fn recurrent_page_context_count(&self) -> usize {
            self.recurrent_page_states.borrow().len()
        }

        fn recurrent_ssm_decode_next_tokens_from_states<F>(
            &self,
            token_ids: &[u32],
            position: usize,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            label: &str,
            mut select: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut(&Self, &GpuF32Tensor, usize) -> Result<u32>,
        {
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("{label} requires at least one request");
            }
            if page_tables.len() != batch_count {
                bail!(
                    "{label} got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            if position >= dims.context {
                bail!(
                    "decode position {position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let token_capacity = position
                .checked_add(1)
                .context("CUDA continuous paged recurrent append token capacity overflows usize")?;
            self.validate_page_tables_for_token_capacity(
                label,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                physical_page_count,
                dims.context,
            )?;

            let mut keys = Vec::with_capacity(batch_count);
            let mut seen = BTreeSet::new();
            for (idx, page_table) in page_tables.iter().enumerate() {
                let key = self.recurrent_page_context_key(label, page_table)?;
                if !seen.insert(key) {
                    bail!("{label} got duplicate recurrent page ownership key {key}");
                }
                keys.push((idx, key));
            }

            {
                let states = self.recurrent_page_states.borrow();
                for (idx, key) in keys.iter().copied() {
                    let state = states.get(&key).ok_or_else(|| {
                    anyhow!(
                        "{label} requires a persistent recurrent state for request {idx}; call prefill/next-token with the same page table before append decode"
                    )
                    })?;
                    if state.page_key != key {
                        bail!(
                            "{label} recurrent state for request {idx} is owned by page key {}, expected {key}",
                            state.page_key
                        );
                    }
                    if state.seq_len != position {
                        bail!(
                            "{label} recurrent state for request {idx} has length {}; expected decode position {position}",
                            state.seq_len
                        );
                    }
                }
            }
            let mut decoded_states = Vec::with_capacity(batch_count);
            {
                let mut states = self.recurrent_page_states.borrow_mut();
                for (idx, key) in keys.iter().copied() {
                    let state = states.remove(&key).ok_or_else(|| {
                        anyhow!(
                            "{label} persistent recurrent state for request {idx} disappeared before decode"
                        )
                    })?;
                    decoded_states.push((idx, key, state));
                }
            }

            let attention_layers = self.recurrent_ssm_attention_layer_count();
            let pool_guard = if attention_layers > 0 {
                Some(self.recurrent_ssm_prepare_paged_pool(
                    &dims,
                    page_size,
                    physical_page_count,
                )?)
            } else {
                None
            };
            let mut tokens = Vec::with_capacity(batch_count);
            for (idx, _key, state) in &mut decoded_states {
                let logits = if let Some(pool_guard) = pool_guard.as_ref() {
                    let request_page_tables = [page_tables[*idx].clone()];
                    let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                        &dims,
                        1,
                        page_size,
                        token_capacity,
                        &request_page_tables,
                        pool_guard.as_ref().ok_or_else(|| {
                            anyhow!("{label} recurrent SSM paged pool is missing")
                        })?,
                        &self.stream,
                    )
                    .with_context(|| {
                        format!(
                            "{label} reopening recurrent SSM attention KV cache for request {idx}"
                        )
                    })?;
                    self.recurrent_ssm_decode_one_logits_with_state(
                        token_ids[*idx],
                        state,
                        Some(&mut cache),
                        label,
                    )?
                } else {
                    self.recurrent_ssm_decode_one_logits_with_state(
                        token_ids[*idx],
                        state,
                        None,
                        label,
                    )?
                };
                tokens.push(select(self, &logits, *idx)?);
            }

            drop(pool_guard);
            let mut states = self.recurrent_page_states.borrow_mut();
            for (_, key, state) in decoded_states {
                states.insert(key, state);
            }
            Ok(tokens)
        }

        fn recurrent_ssm_greedy_next_tokens_full_context(
            &self,
            inputs: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            let mut tokens = Vec::with_capacity(inputs.len());
            for input in inputs {
                let logits = self.full_context_logits_device(input)?;
                tokens.push(self.argmax_last_row(&logits)?);
            }
            Ok(tokens)
        }

        fn recurrent_ssm_sampled_next_tokens_full_context(
            &self,
            inputs: &[Vec<u32>],
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            samples: &[f32],
        ) -> Result<Vec<u32>> {
            if samples.len() != inputs.len() {
                bail!(
                    "CUDA recurrent SSM full-context sampled next-token got {} random samples for {} requests",
                    samples.len(),
                    inputs.len()
                );
            }
            let mut tokens = Vec::with_capacity(inputs.len());
            for (input, sample) in inputs.iter().zip(samples.iter().copied()) {
                let logits = self.full_context_logits_device(input)?;
                tokens.push(self.sample_last_row_with_sample(
                    &logits,
                    temperature,
                    top_p,
                    top_k,
                    sample,
                )?);
            }
            Ok(tokens)
        }

        pub fn generate_sampled_tokens_batch_with_limits(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
        ) -> Result<Vec<Vec<u32>>> {
            self.generate_sampled_tokens_batch_with_limits_and_cancellation(
                inputs,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seeds,
                &[],
                |_| false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_limits_and_cancellation<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!("CUDA batched sampled generation requires at least one request");
            }
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_batched_text_generation_request(
                    "CUDA batched sampled generation",
                    inputs,
                    max_tokens_per_request,
                    stop_token_sequences_per_request,
                    Some(seeds),
                )?;
                return self.generate_recurrent_ssm_sampled_batch_full_context(
                    inputs,
                    max_tokens_per_request,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    seeds,
                    stop_token_sequences_per_request,
                    is_cancelled,
                );
            }
            if !self.supports_batched_text_generation() {
                bail!(
                    "CUDA batched sampled generation currently supports loaded decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA batched sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA batched sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA batched sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("CUDA batched sampled generation requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("CUDA batched sampled generation requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("CUDA batched sampled generation requires equal prompt lengths");
            }
            validate_batched_generation_context_budget(
                "CUDA batched sampled generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let mut rngs = per_request_sample_rngs(seeds);
            let mut flat = Vec::with_capacity(batch_count * prompt_len);
            for input in inputs {
                flat.extend_from_slice(input);
            }
            let mut cache = CudaKvCache::new_batched(
                self.config.block_count,
                &dims,
                batch_count,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_device_batched(
                &flat,
                batch_count,
                prompt_len,
                Some(&mut cache),
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let samples = rngs
                    .iter_mut()
                    .map(|rng| rng.gen_range(0.0f32..1.0f32))
                    .collect::<Vec<_>>();
                let batch_tokens = self.sample_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                    temperature,
                    top_p,
                    top_k,
                    &samples,
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_and_limits_and_cancellation<F>(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!("CUDA batched multimodal sampled generation requires at least one request");
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA batched multimodal sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA batched multimodal sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal sampled generation got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA batched multimodal sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("CUDA batched multimodal sampled generation requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!("CUDA batched multimodal sampled generation requires non-empty prompts");
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!("CUDA batched multimodal sampled generation requires equal prompt lengths");
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows
                .checked_add(token_prompt_len)
                .context("CUDA batched multimodal prompt length overflows usize")?;
            if prompt_len > dims.context {
                bail!(
                    "multimodal input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA batched multimodal sampled generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let mut rngs = per_request_sample_rngs(seeds);
            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count
                    .checked_mul(prompt_len)
                    .context("CUDA batched multimodal hidden row count overflows usize")?,
                dims.embed,
                "CUDA batched multimodal prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new_batched(
                self.config.block_count,
                &dims,
                batch_count,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_from_hidden_device_batched(
                hidden,
                batch_count,
                prompt_len,
                Some(&mut cache),
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let samples = rngs
                    .iter_mut()
                    .map(|rng| rng.gen_range(0.0f32..1.0f32))
                    .collect::<Vec<_>>();
                let batch_tokens = self.sample_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                    temperature,
                    top_p,
                    top_k,
                    &samples,
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA batched multimodal MRoPE sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if position_ids_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation got {} position rows for {batch_count} requests",
                    position_ids_per_request.len()
                );
            }
            if next_rope_positions.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation got {} next RoPE positions for {batch_count} requests",
                    next_rope_positions.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 {
                bail!(
                    "CUDA batched multimodal MRoPE sampled generation requires non-empty prompts"
                );
            }
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA batched multimodal MRoPE sampled generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let mut rngs = per_request_sample_rngs(seeds);
            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context("CUDA batched multimodal MRoPE hidden value count overflows usize")?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA batched multimodal MRoPE sampled generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            if hidden_host.len() != total_values {
                bail!(
                    "CUDA batched multimodal MRoPE hidden build produced {} values; expected {total_values}",
                    hidden_host.len()
                );
            }

            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count
                    .checked_mul(prompt_rows)
                    .context("CUDA batched multimodal MRoPE hidden row count overflows usize")?,
                dims.embed,
                "CUDA batched multimodal MRoPE prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new_batched(
                self.config.block_count,
                &dims,
                batch_count,
                &self.stream,
            )?;
            let mut logits = self
                .full_context_logits_from_hidden_device_batched_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    Some(&mut cache),
                )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_cache_position = prompt_rows;
            let mut next_rope_positions = next_rope_positions.to_vec();

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let samples = rngs
                    .iter_mut()
                    .map(|rng| rng.gen_range(0.0f32..1.0f32))
                    .collect::<Vec<_>>();
                let batch_tokens = self.sample_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_rows } else { 1 },
                    temperature,
                    top_p,
                    top_k,
                    &samples,
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_cache_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_cache_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits = self.decode_batch_logits_device_with_rope_positions(
                    &next_tokens,
                    next_cache_position,
                    &next_rope_positions,
                    &mut cache,
                )?;
                next_cache_position += 1;
                for position in &mut next_rope_positions {
                    *position = position
                        .checked_add(1)
                        .context("next MRoPE decode position overflows usize")?;
                }
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA paged batched multimodal sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA paged batched multimodal sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA paged batched multimodal sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal sampled generation got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA paged batched multimodal sampled generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!(
                    "CUDA paged batched multimodal sampled generation requires non-empty prompts"
                );
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!(
                    "CUDA paged batched multimodal sampled generation requires equal prompt lengths"
                );
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows
                .checked_add(token_prompt_len)
                .context("CUDA paged batched multimodal prompt length overflows usize")?;
            if prompt_len > dims.context {
                bail!(
                    "multimodal input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged batched multimodal sampled generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let rngs = per_request_sample_rngs(seeds);
            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count
                    .checked_mul(prompt_len)
                    .context("CUDA paged batched multimodal hidden row count overflows usize")?,
                dims.embed,
                "CUDA paged batched multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context("CUDA paged batched multimodal sampled token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedBatchKvCache::new(
                self.config.block_count,
                &dims,
                batch_count,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let logits = self.full_context_logits_from_hidden_device_batched_paged_cache(
                hidden,
                batch_count,
                prompt_len,
                &mut cache,
            )?;

            self.generate_batched_sampled_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_len,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                rngs,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_position| {
                    self.decode_batch_logits_paged_device(next_tokens, next_position, &mut cache)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA paged batched multimodal MRoPE sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if position_ids_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation got {} position rows for {batch_count} requests",
                    position_ids_per_request.len()
                );
            }
            if next_rope_positions.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation got {} next RoPE positions for {batch_count} requests",
                    next_rope_positions.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 {
                bail!(
                    "CUDA paged batched multimodal MRoPE sampled generation requires non-empty prompts"
                );
            }
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged batched multimodal MRoPE sampled generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let rngs = per_request_sample_rngs(seeds);
            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context(
                    "CUDA paged batched multimodal MRoPE hidden value count overflows usize",
                )?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA paged batched multimodal MRoPE sampled generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            if hidden_host.len() != total_values {
                bail!(
                    "CUDA paged batched multimodal MRoPE hidden build produced {} values; expected {total_values}",
                    hidden_host.len()
                );
            }

            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_rows).context(
                    "CUDA paged batched multimodal MRoPE hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA paged batched multimodal MRoPE prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_decode_steps)
                .context(
                    "CUDA paged batched multimodal MRoPE sampled token capacity overflows usize",
                )?
                .min(dims.context);
            let mut cache = CudaPagedBatchKvCache::new(
                self.config.block_count,
                &dims,
                batch_count,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let logits = self
                .full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    &mut cache,
                )?;
            let mut next_rope_positions = next_rope_positions.to_vec();

            self.generate_batched_sampled_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_rows,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                rngs,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_cache_position| {
                    let decoded = self.decode_batch_logits_paged_device_with_rope_positions(
                        next_tokens,
                        next_cache_position,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                    for position in &mut next_rope_positions {
                        *position = position
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                    Ok(decoded)
                },
            )
        }

        #[cfg(feature = "native-cuda")]
        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched multimodal greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires non-empty prompts"
                );
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires equal prompt lengths"
                );
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows.checked_add(token_prompt_len).context(
                "CUDA lease-backed paged batched multimodal prompt length overflows usize",
            )?;
            if prompt_len > dims.context {
                bail!(
                    "multimodal input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched multimodal greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_len).context(
                    "CUDA lease-backed paged batched multimodal hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA lease-backed paged batched multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context(
                    "CUDA lease-backed paged batched multimodal greedy token capacity overflows usize",
                )?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let logits = self.full_context_logits_from_hidden_device_batched_paged_cache(
                hidden,
                batch_count,
                prompt_len,
                &mut cache,
            )?;

            self.generate_batched_greedy_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_len,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_position| {
                    self.decode_batch_logits_paged_device(next_tokens, next_position, &mut cache)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows_per_request: &[usize],
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged ragged multimodal greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count
                || prefix_embeddings_per_request.len() != batch_count
                || prefix_rows_per_request.len() != batch_count
                || max_tokens_per_request.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged ragged multimodal greedy generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged ragged multimodal greedy generation requires positive token limits"
                );
            }

            let mut prompt_lens = Vec::with_capacity(batch_count);
            let mut token_capacity = 0usize;
            for (((input, prefix_embeddings), prefix_rows), max_tokens) in inputs
                .iter()
                .zip(prefix_embeddings_per_request)
                .zip(prefix_rows_per_request)
                .zip(max_tokens_per_request)
            {
                if input.is_empty() {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal greedy generation requires non-empty prompts"
                    );
                }
                self.validate_prefix_embeddings(prefix_embeddings, *prefix_rows, dims.embed)?;
                let prompt_len = prefix_rows.checked_add(input.len()).context(
                    "CUDA lease-backed paged ragged multimodal prompt length overflows usize",
                )?;
                validate_batched_generation_context_budget(
                    "CUDA lease-backed paged ragged multimodal greedy generation",
                    prompt_len,
                    *max_tokens,
                    dims.context,
                )?;
                prompt_lens.push(prompt_len);
                token_capacity = token_capacity.max(prompt_len.checked_add(*max_tokens).context(
                    "CUDA lease-backed paged ragged multimodal greedy token capacity overflows usize",
                )?);
            }
            if token_capacity == 0 || token_capacity > dims.context {
                bail!(
                    "CUDA lease-backed paged ragged multimodal token capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);

            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;

            let mut initial_logits = Vec::with_capacity(batch_count);
            for idx in 0..batch_count {
                let token_embeddings = self.embed_tokens_device(&inputs[idx])?.copy_to_host()?;
                let prompt_len = prompt_lens[idx];
                let total_values = prompt_len.checked_mul(dims.embed).context(
                    "CUDA lease-backed paged ragged multimodal hidden value count overflows usize",
                )?;
                let mut hidden_host = Vec::with_capacity(total_values);
                hidden_host.extend_from_slice(&prefix_embeddings_per_request[idx]);
                hidden_host.extend_from_slice(&token_embeddings);
                if hidden_host.len() != total_values {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal hidden build produced {} values; expected {total_values}",
                        hidden_host.len()
                    );
                }
                let hidden = self.f32_tensor_from_host(
                    &hidden_host,
                    prompt_len,
                    dims.embed,
                    "CUDA lease-backed paged ragged multimodal prompt embeddings",
                )?;
                let request_page_tables = [page_tables[idx].clone()];
                let mut prefill_cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                    &dims,
                    1,
                    page_size,
                    token_capacity,
                    &request_page_tables,
                    pool,
                    &self.stream,
                )?;
                initial_logits.push(
                    self.full_context_logits_from_hidden_device_batched_paged_cache(
                        hidden,
                        1,
                        prompt_len,
                        &mut prefill_cache,
                    )?,
                );
            }

            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_positions = prompt_lens;
            let mut logits: Option<GpuF32Tensor> = None;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = if step == 0 {
                    let mut tokens = Vec::with_capacity(batch_count);
                    for logits in &initial_logits {
                        tokens.push(self.argmax_last_row(logits)?);
                    }
                    tokens
                } else {
                    let logits = logits
                        .as_ref()
                        .ok_or_else(|| anyhow!("CUDA ragged multimodal decode logits missing"))?;
                    self.argmax_batched_last_token(logits, batch_count, 1)?
                };
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                let decoded = self.decode_batch_logits_paged_device_with_positions(
                    &next_tokens,
                    &next_positions,
                    &mut cache,
                )?;
                for (idx, active) in active.iter().copied().enumerate() {
                    if active {
                        next_positions[idx] = next_positions[idx]
                            .checked_add(1)
                            .context("CUDA ragged multimodal decode position overflows usize")?;
                    }
                }
                logits = Some(decoded);
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows_per_request: &[usize],
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged ragged multimodal MRoPE greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count
                || prompt_rows_per_request.len() != batch_count
                || position_ids_per_request.len() != batch_count
                || next_rope_positions.len() != batch_count
                || max_tokens_per_request.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE greedy generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE greedy generation requires positive token limits"
                );
            }

            let mut prompt_lens = Vec::with_capacity(batch_count);
            let mut token_capacity = 0usize;
            for (((embeddings, prompt_rows), position_ids), max_tokens) in
                prompt_embeddings_per_request
                    .iter()
                    .zip(prompt_rows_per_request)
                    .zip(position_ids_per_request)
                    .zip(max_tokens_per_request)
            {
                if *prompt_rows == 0 {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal MRoPE greedy generation requires non-empty prompts"
                    );
                }
                self.validate_prompt_embeddings(embeddings, *prompt_rows, dims.embed)?;
                if position_ids.len() != *prompt_rows {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal MRoPE greedy generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                validate_batched_generation_context_budget(
                    "CUDA lease-backed paged ragged multimodal MRoPE greedy generation",
                    *prompt_rows,
                    *max_tokens,
                    dims.context,
                )?;
                prompt_lens.push(*prompt_rows);
                token_capacity = token_capacity.max(prompt_rows.checked_add(*max_tokens).context(
                    "CUDA lease-backed paged ragged multimodal MRoPE greedy token capacity overflows usize",
                )?);
            }
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }
            if token_capacity == 0 || token_capacity > dims.context {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE token capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);

            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;

            let mut initial_logits = Vec::with_capacity(batch_count);
            for idx in 0..batch_count {
                let hidden = self.f32_tensor_from_host(
                    &prompt_embeddings_per_request[idx],
                    prompt_lens[idx],
                    dims.embed,
                    "CUDA lease-backed paged ragged multimodal MRoPE prompt embeddings",
                )?;
                let request_page_tables = [page_tables[idx].clone()];
                let mut prefill_cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                    &dims,
                    1,
                    page_size,
                    token_capacity,
                    &request_page_tables,
                    pool,
                    &self.stream,
                )?;
                initial_logits.push(
                    self.full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                        hidden,
                        1,
                        prompt_lens[idx],
                        &position_ids_per_request[idx],
                        &mut prefill_cache,
                    )?,
                );
            }

            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_cache_positions = prompt_lens;
            let mut next_rope_positions = next_rope_positions.to_vec();
            let mut logits: Option<GpuF32Tensor> = None;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = if step == 0 {
                    let mut tokens = Vec::with_capacity(batch_count);
                    for logits in &initial_logits {
                        tokens.push(self.argmax_last_row(logits)?);
                    }
                    tokens
                } else {
                    let logits = logits.as_ref().ok_or_else(|| {
                        anyhow!("CUDA ragged multimodal MRoPE decode logits missing")
                    })?;
                    self.argmax_batched_last_token(logits, batch_count, 1)?
                };
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                let decoded = self
                    .decode_batch_logits_paged_device_with_cache_and_mrope_positions(
                        &next_tokens,
                        &next_cache_positions,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                for (idx, active) in active.iter().copied().enumerate() {
                    if active {
                        next_cache_positions[idx] =
                            next_cache_positions[idx].checked_add(1).context(
                                "CUDA ragged multimodal MRoPE decode position overflows usize",
                            )?;
                        next_rope_positions[idx] = next_rope_positions[idx]
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                }
                logits = Some(decoded);
            }

            Ok(generated)
        }

        #[cfg(feature = "native-cuda")]
        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched multimodal MRoPE greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            if position_ids_per_request.len() != batch_count
                || next_rope_positions.len() != batch_count
                || max_tokens_per_request.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 || prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched multimodal MRoPE greedy generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context(
                    "CUDA lease-backed paged batched multimodal MRoPE hidden value count overflows usize",
                )?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA lease-backed paged batched multimodal MRoPE greedy generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_rows).context(
                    "CUDA lease-backed paged batched multimodal MRoPE hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA lease-backed paged batched multimodal MRoPE prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_decode_steps)
                .context(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy token capacity overflows usize",
                )?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let logits = self
                .full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    &mut cache,
                )?;
            let mut next_rope_positions = next_rope_positions.to_vec();

            self.generate_batched_greedy_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_rows,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_cache_position| {
                    let decoded = self.decode_batch_logits_paged_device_with_rope_positions(
                        next_tokens,
                        next_cache_position,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                    for position in &mut next_rope_positions {
                        *position = position
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                    Ok(decoded)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched multimodal sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged batched multimodal sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched multimodal sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count
                || prefix_embeddings_per_request.len() != batch_count
                || max_tokens_per_request.len() != batch_count
                || seeds.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged batched multimodal sampled generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched multimodal sampled generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!(
                    "CUDA lease-backed paged batched multimodal sampled generation requires non-empty prompts"
                );
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!(
                    "CUDA lease-backed paged batched multimodal sampled generation requires equal prompt lengths"
                );
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows.checked_add(token_prompt_len).context(
                "CUDA lease-backed paged batched multimodal prompt length overflows usize",
            )?;
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched multimodal sampled generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let rngs = per_request_sample_rngs(seeds);
            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_len).context(
                    "CUDA lease-backed paged batched multimodal hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA lease-backed paged batched multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context(
                    "CUDA lease-backed paged batched multimodal sampled token capacity overflows usize",
                )?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let logits = self.full_context_logits_from_hidden_device_batched_paged_cache(
                hidden,
                batch_count,
                prompt_len,
                &mut cache,
            )?;

            self.generate_batched_sampled_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_len,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                rngs,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_position| {
                    self.decode_batch_logits_paged_device(next_tokens, next_position, &mut cache)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows_per_request: &[usize],
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged ragged multimodal sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count
                || prefix_embeddings_per_request.len() != batch_count
                || prefix_rows_per_request.len() != batch_count
                || max_tokens_per_request.len() != batch_count
                || seeds.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged ragged multimodal sampled generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged ragged multimodal sampled generation requires positive token limits"
                );
            }

            let mut prompt_lens = Vec::with_capacity(batch_count);
            let mut token_capacity = 0usize;
            for (((input, prefix_embeddings), prefix_rows), max_tokens) in inputs
                .iter()
                .zip(prefix_embeddings_per_request)
                .zip(prefix_rows_per_request)
                .zip(max_tokens_per_request)
            {
                if input.is_empty() {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal sampled generation requires non-empty prompts"
                    );
                }
                self.validate_prefix_embeddings(prefix_embeddings, *prefix_rows, dims.embed)?;
                let prompt_len = prefix_rows.checked_add(input.len()).context(
                    "CUDA lease-backed paged ragged multimodal prompt length overflows usize",
                )?;
                validate_batched_generation_context_budget(
                    "CUDA lease-backed paged ragged multimodal sampled generation",
                    prompt_len,
                    *max_tokens,
                    dims.context,
                )?;
                prompt_lens.push(prompt_len);
                token_capacity = token_capacity.max(prompt_len.checked_add(*max_tokens).context(
                    "CUDA lease-backed paged ragged multimodal sampled token capacity overflows usize",
                )?);
            }
            if token_capacity == 0 || token_capacity > dims.context {
                bail!(
                    "CUDA lease-backed paged ragged multimodal token capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let mut rngs = per_request_sample_rngs(seeds);

            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;

            let mut initial_logits = Vec::with_capacity(batch_count);
            for idx in 0..batch_count {
                let token_embeddings = self.embed_tokens_device(&inputs[idx])?.copy_to_host()?;
                let prompt_len = prompt_lens[idx];
                let total_values = prompt_len.checked_mul(dims.embed).context(
                    "CUDA lease-backed paged ragged multimodal hidden value count overflows usize",
                )?;
                let mut hidden_host = Vec::with_capacity(total_values);
                hidden_host.extend_from_slice(&prefix_embeddings_per_request[idx]);
                hidden_host.extend_from_slice(&token_embeddings);
                if hidden_host.len() != total_values {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal hidden build produced {} values; expected {total_values}",
                        hidden_host.len()
                    );
                }
                let hidden = self.f32_tensor_from_host(
                    &hidden_host,
                    prompt_len,
                    dims.embed,
                    "CUDA lease-backed paged ragged multimodal prompt embeddings",
                )?;
                let request_page_tables = [page_tables[idx].clone()];
                let mut prefill_cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                    &dims,
                    1,
                    page_size,
                    token_capacity,
                    &request_page_tables,
                    pool,
                    &self.stream,
                )?;
                initial_logits.push(
                    self.full_context_logits_from_hidden_device_batched_paged_cache(
                        hidden,
                        1,
                        prompt_len,
                        &mut prefill_cache,
                    )?,
                );
            }

            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_positions = prompt_lens;
            let mut logits: Option<GpuF32Tensor> = None;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = if step == 0 {
                    let mut tokens = vec![eos_token_id.unwrap_or(0); batch_count];
                    for idx in 0..batch_count {
                        if active[idx] {
                            tokens[idx] = self.sample_last_row(
                                &initial_logits[idx],
                                temperature,
                                top_p,
                                top_k,
                                &mut rngs[idx],
                            )?;
                        }
                    }
                    tokens
                } else {
                    let logits = logits
                        .as_ref()
                        .ok_or_else(|| anyhow!("CUDA ragged multimodal decode logits missing"))?;
                    let samples = rngs
                        .iter_mut()
                        .map(|rng| rng.gen_range(0.0f32..1.0f32))
                        .collect::<Vec<_>>();
                    self.sample_batched_last_token(
                        logits,
                        batch_count,
                        1,
                        temperature,
                        top_p,
                        top_k,
                        &samples,
                    )?
                };
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                let decoded = self.decode_batch_logits_paged_device_with_positions(
                    &next_tokens,
                    &next_positions,
                    &mut cache,
                )?;
                for (idx, active) in active.iter().copied().enumerate() {
                    if active {
                        next_positions[idx] = next_positions[idx]
                            .checked_add(1)
                            .context("CUDA ragged multimodal decode position overflows usize")?;
                    }
                }
                logits = Some(decoded);
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows_per_request: &[usize],
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged ragged multimodal MRoPE sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count
                || prompt_rows_per_request.len() != batch_count
                || position_ids_per_request.len() != batch_count
                || next_rope_positions.len() != batch_count
                || max_tokens_per_request.len() != batch_count
                || seeds.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE sampled generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE sampled generation requires positive token limits"
                );
            }

            let mut prompt_lens = Vec::with_capacity(batch_count);
            let mut token_capacity = 0usize;
            for (((embeddings, prompt_rows), position_ids), max_tokens) in
                prompt_embeddings_per_request
                    .iter()
                    .zip(prompt_rows_per_request)
                    .zip(position_ids_per_request)
                    .zip(max_tokens_per_request)
            {
                if *prompt_rows == 0 {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal MRoPE sampled generation requires non-empty prompts"
                    );
                }
                self.validate_prompt_embeddings(embeddings, *prompt_rows, dims.embed)?;
                if position_ids.len() != *prompt_rows {
                    bail!(
                        "CUDA lease-backed paged ragged multimodal MRoPE sampled generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                validate_batched_generation_context_budget(
                    "CUDA lease-backed paged ragged multimodal MRoPE sampled generation",
                    *prompt_rows,
                    *max_tokens,
                    dims.context,
                )?;
                prompt_lens.push(*prompt_rows);
                token_capacity = token_capacity.max(prompt_rows.checked_add(*max_tokens).context(
                    "CUDA lease-backed paged ragged multimodal MRoPE sampled token capacity overflows usize",
                )?);
            }
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }
            if token_capacity == 0 || token_capacity > dims.context {
                bail!(
                    "CUDA lease-backed paged ragged multimodal MRoPE token capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let mut rngs = per_request_sample_rngs(seeds);

            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;

            let mut initial_logits = Vec::with_capacity(batch_count);
            for idx in 0..batch_count {
                let hidden = self.f32_tensor_from_host(
                    &prompt_embeddings_per_request[idx],
                    prompt_lens[idx],
                    dims.embed,
                    "CUDA lease-backed paged ragged multimodal MRoPE prompt embeddings",
                )?;
                let request_page_tables = [page_tables[idx].clone()];
                let mut prefill_cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                    &dims,
                    1,
                    page_size,
                    token_capacity,
                    &request_page_tables,
                    pool,
                    &self.stream,
                )?;
                initial_logits.push(
                    self.full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                        hidden,
                        1,
                        prompt_lens[idx],
                        &position_ids_per_request[idx],
                        &mut prefill_cache,
                    )?,
                );
            }

            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_cache_positions = prompt_lens;
            let mut next_rope_positions = next_rope_positions.to_vec();
            let mut logits: Option<GpuF32Tensor> = None;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let batch_tokens = if step == 0 {
                    let mut tokens = vec![eos_token_id.unwrap_or(0); batch_count];
                    for idx in 0..batch_count {
                        if active[idx] {
                            tokens[idx] = self.sample_last_row(
                                &initial_logits[idx],
                                temperature,
                                top_p,
                                top_k,
                                &mut rngs[idx],
                            )?;
                        }
                    }
                    tokens
                } else {
                    let logits = logits.as_ref().ok_or_else(|| {
                        anyhow!("CUDA ragged multimodal MRoPE decode logits missing")
                    })?;
                    let samples = rngs
                        .iter_mut()
                        .map(|rng| rng.gen_range(0.0f32..1.0f32))
                        .collect::<Vec<_>>();
                    self.sample_batched_last_token(
                        logits,
                        batch_count,
                        1,
                        temperature,
                        top_p,
                        top_k,
                        &samples,
                    )?
                };
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                let decoded = self
                    .decode_batch_logits_paged_device_with_cache_and_mrope_positions(
                        &next_tokens,
                        &next_cache_positions,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                for (idx, active) in active.iter().copied().enumerate() {
                    if active {
                        next_cache_positions[idx] =
                            next_cache_positions[idx].checked_add(1).context(
                                "CUDA ragged multimodal MRoPE decode position overflows usize",
                            )?;
                        next_rope_positions[idx] = next_rope_positions[idx]
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                }
                logits = Some(decoded);
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE sampled generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE sampled generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched multimodal MRoPE sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count
                || position_ids_per_request.len() != batch_count
                || next_rope_positions.len() != batch_count
                || max_tokens_per_request.len() != batch_count
                || seeds.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE sampled generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE sampled generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 || prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched multimodal MRoPE sampled generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let rngs = per_request_sample_rngs(seeds);
            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context(
                    "CUDA lease-backed paged batched multimodal MRoPE hidden value count overflows usize",
                )?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA lease-backed paged batched multimodal MRoPE sampled generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_rows).context(
                    "CUDA lease-backed paged batched multimodal MRoPE hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA lease-backed paged batched multimodal MRoPE prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_decode_steps)
                .context(
                    "CUDA lease-backed paged batched multimodal MRoPE sampled token capacity overflows usize",
                )?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let logits = self
                .full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    &mut cache,
                )?;
            let mut next_rope_positions = next_rope_positions.to_vec();

            self.generate_batched_sampled_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_rows,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                rngs,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_cache_position| {
                    let decoded = self.decode_batch_logits_paged_device_with_rope_positions(
                        next_tokens,
                        next_cache_position,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                    for position in &mut next_rope_positions {
                        *position = position
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                    Ok(decoded)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_limits(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
        ) -> Result<Vec<Vec<u32>>> {
            self.generate_sampled_tokens_batch_paged_with_limits_and_cancellation(
                inputs,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seeds,
                page_size,
                &[],
                |_| false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_limits_and_cancellation<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!("CUDA paged batched sampled generation requires at least one request");
            }
            if self.config.recurrent_ssm_tensor_layout {
                self.validate_batched_text_generation_request(
                    "CUDA paged batched sampled generation",
                    inputs,
                    max_tokens_per_request,
                    stop_token_sequences_per_request,
                    Some(seeds),
                )?;
                return self.generate_recurrent_ssm_sampled_batch_full_context(
                    inputs,
                    max_tokens_per_request,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    seeds,
                    stop_token_sequences_per_request,
                    is_cancelled,
                );
            }
            if !self.supports_batched_text_generation() {
                bail!(
                    "CUDA paged batched sampled generation currently supports loaded decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA paged batched sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA paged batched sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA paged batched sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!("CUDA paged batched sampled generation requires positive token limits");
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("CUDA paged batched sampled generation requires non-empty prompts");
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("CUDA paged batched sampled generation requires equal prompt lengths");
            }
            validate_batched_generation_context_budget(
                "CUDA paged batched sampled generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let mut rngs = per_request_sample_rngs(seeds);
            let mut flat = Vec::with_capacity(batch_count * prompt_len);
            for input in inputs {
                flat.extend_from_slice(input);
            }
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context("CUDA paged batched sampled token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedBatchKvCache::new(
                self.config.block_count,
                &dims,
                batch_count,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_device_batched_paged_cache(
                &flat,
                batch_count,
                prompt_len,
                &mut cache,
                false,
            )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let samples = rngs
                    .iter_mut()
                    .map(|rng| rng.gen_range(0.0f32..1.0f32))
                    .collect::<Vec<_>>();
                let batch_tokens = self.sample_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { prompt_len } else { 1 },
                    temperature,
                    top_p,
                    top_k,
                    &samples,
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_paged_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_page_tables(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
        ) -> Result<Vec<Vec<u32>>> {
            self.generate_sampled_tokens_batch_paged_with_page_tables_and_cancellation(
                inputs,
                max_tokens_per_request,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seeds,
                page_size,
                page_tables,
                physical_page_count,
                &[],
                |_| false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_page_tables_and_cancellation<F>(
            &self,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seeds: &[Option<u64>],
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            mut is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched sampled generation requires at least one request"
                );
            }
            if self.config.recurrent_ssm_tensor_layout {
                let (dims, batch_count, prompt_len, max_decode_steps) = self
                    .validate_batched_text_generation_request(
                        "CUDA lease-backed paged batched sampled generation",
                        inputs,
                        max_tokens_per_request,
                        stop_token_sequences_per_request,
                        Some(seeds),
                    )?;
                let token_capacity = prompt_len
                    .checked_add(max_decode_steps)
                    .context(
                        "CUDA lease-backed paged batched sampled token capacity overflows usize",
                    )?
                    .min(dims.context);
                self.validate_page_tables_for_token_capacity(
                    "CUDA lease-backed paged batched sampled generation",
                    batch_count,
                    page_size,
                    token_capacity,
                    page_tables,
                    physical_page_count,
                    dims.context,
                )?;
                return self.generate_recurrent_ssm_sampled_batch_full_context(
                    inputs,
                    max_tokens_per_request,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    seeds,
                    stop_token_sequences_per_request,
                    is_cancelled,
                );
            }
            if !self.supports_batched_text_generation() {
                bail!(
                    "CUDA lease-backed paged batched sampled generation currently supports loaded decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched sampled generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched sampled generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if seeds.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched sampled generation got {} seeds for {batch_count} requests",
                    seeds.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched sampled generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!(
                    "CUDA lease-backed paged batched sampled generation requires non-empty prompts"
                );
            }
            if prompt_len > dims.context {
                bail!(
                    "input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!(
                    "CUDA lease-backed paged batched sampled generation requires equal prompt lengths"
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched sampled generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let mut rngs = per_request_sample_rngs(seeds);
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context("CUDA lease-backed paged batched sampled token capacity overflows usize")?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let (mut cache, mut logits, initial_logits_seq_len) = self
                .full_context_logits_device_batched_paged_cache_with_shared_prefix(
                    inputs,
                    page_size,
                    token_capacity,
                    page_tables,
                    pool,
                )?;
            let mut generated = vec![Vec::new(); batch_count];
            let mut active = vec![true; batch_count];
            let mut next_position = prompt_len;

            for step in 0..max_decode_steps {
                if !retain_uncancelled_batch_rows(&mut active, &mut is_cancelled) {
                    break;
                }
                let samples = rngs
                    .iter_mut()
                    .map(|rng| rng.gen_range(0.0f32..1.0f32))
                    .collect::<Vec<_>>();
                let batch_tokens = self.sample_batched_last_token(
                    &logits,
                    batch_count,
                    if step == 0 { initial_logits_seq_len } else { 1 },
                    temperature,
                    top_p,
                    top_k,
                    &samples,
                )?;
                let mut any_active = false;
                for (idx, token) in batch_tokens.iter().copied().enumerate() {
                    if !active[idx] {
                        continue;
                    }
                    generated[idx].push(token);
                    if is_row_stop_sequence(
                        token,
                        &generated[idx],
                        eos_token_id,
                        stop_token_sequences_per_request,
                        idx,
                    ) || generated[idx].len() >= max_tokens_per_request[idx]
                    {
                        active[idx] = false;
                    }
                    any_active |= active[idx];
                }
                if !any_active || step + 1 == max_decode_steps {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                let mut next_tokens = batch_tokens;
                for (idx, active) in active.iter().copied().enumerate() {
                    if !active {
                        next_tokens[idx] = eos_token_id.unwrap_or(next_tokens[idx]);
                    }
                }
                logits =
                    self.decode_batch_logits_paged_device(&next_tokens, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn generate_sampled_tokens(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                &[],
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_stop_ids(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            let stop_token_sequences = stop_sequences_from_stop_ids(stop_token_ids);
            self.generate_sampled_tokens_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                &stop_token_sequences,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_stop_sequences(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_with_stop_sequences_and_cancellation(
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                stop_token_sequences,
                || false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_stop_sequences_and_cancellation<F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            stop_token_sequences: &[Vec<u32>],
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_rng(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                    &mut is_cancelled,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_rng(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                    &mut is_cancelled,
                )
            }
        }

        pub fn generate_sampled_tokens_paged(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_paged_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                page_size,
                &[],
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged_with_stop_ids(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
            stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            let stop_token_sequences = stop_sequences_from_stop_ids(stop_token_ids);
            self.generate_sampled_tokens_paged_with_stop_sequences(
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                page_size,
                &stop_token_sequences,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged_with_stop_sequences(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_paged_with_stop_sequences_and_cancellation(
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                page_size,
                stop_token_sequences,
                || false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged_with_stop_sequences_and_cancellation<F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_paged_with_rng(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                    &mut is_cancelled,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_paged_with_rng(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                    &mut is_cancelled,
                )
            }
        }

        pub fn generate_sampled_tokens_with_prefix_embeddings(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_with_prefix_embeddings_and_stop_sequences(
                prefix_embeddings,
                prefix_rows,
                input_ids,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                &[],
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prefix_embeddings_and_stop_sequences(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_prefix_embeddings_and_rng(
                    prefix_embeddings,
                    prefix_rows,
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_prefix_embeddings_and_rng(
                    prefix_embeddings,
                    prefix_rows,
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                )
            }
        }

        pub fn generate_sampled_tokens_with_prompt_embeddings(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_with_prompt_embeddings_and_stop_sequences(
                prompt_embeddings,
                prompt_rows,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                &[],
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_prompt_embeddings_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_prompt_embeddings_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_positions(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            self.generate_sampled_tokens_with_prompt_embeddings_and_positions_and_stop_sequences(
                prompt_embeddings,
                prompt_rows,
                position_ids,
                next_rope_position,
                max_tokens,
                eos_token_id,
                temperature,
                top_p,
                top_k,
                seed,
                &[],
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_positions_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_prompt_embeddings_positions_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    position_ids,
                    next_rope_position,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_prompt_embeddings_positions_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    position_ids,
                    next_rope_position,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    &mut rng,
                )
            }
        }

        fn generate_sampled_tokens_with_rng<R, F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            R: Rng + ?Sized,
            F: FnMut() -> bool,
        {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens("CUDA sampled generation", max_tokens)?;
            if input_ids.is_empty() {
                bail!("CUDA sampled generation requires at least one input token");
            }
            if input_ids.len() > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    input_ids.len(),
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA sampled generation",
                input_ids.len(),
                max_tokens,
                dims.context,
            )?;

            if self.config.recurrent_ssm_tensor_layout {
                return self.generate_sampled_tokens_full_context_with_cancellation(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    rng,
                    is_cancelled,
                );
            }

            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits =
                self.full_context_logits_device_with_cache(input_ids, Some(&mut cache))?;
            let mut generated = Vec::new();
            let mut next_position = input_ids.len();

            for step in 0..max_tokens {
                if is_cancelled() {
                    break;
                }
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_full_context_with_cancellation<R, F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            R: Rng + ?Sized,
            F: FnMut() -> bool,
        {
            let dims = self.qwen_dims()?;
            let mut context = input_ids.to_vec();
            let mut generated = Vec::new();
            for _ in 0..max_tokens {
                if is_cancelled() {
                    break;
                }
                let logits = self.full_context_logits_device(&context)?;
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if context.len() >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        context.len() + 1,
                        dims.context
                    );
                }
                context.push(next);
            }
            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_paged_with_rng<R, F>(
            &self,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
            mut is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            R: Rng + ?Sized,
            F: FnMut() -> bool,
        {
            let dims = self.qwen_dims()?;
            let max_tokens =
                validate_generation_max_tokens("CUDA paged sampled generation", max_tokens)?;
            if input_ids.is_empty() {
                bail!("CUDA paged sampled generation requires at least one input token");
            }
            if input_ids.len() > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    input_ids.len(),
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged sampled generation",
                input_ids.len(),
                max_tokens,
                dims.context,
            )?;

            if self.config.recurrent_ssm_tensor_layout {
                return self.generate_sampled_tokens_full_context_with_cancellation(
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    stop_token_sequences,
                    rng,
                    is_cancelled,
                );
            }

            let token_capacity = input_ids
                .len()
                .checked_add(max_tokens)
                .context("CUDA paged sampled generation token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits =
                self.full_context_logits_device_with_paged_cache(input_ids, &mut cache)?;
            let mut generated = Vec::new();
            let mut next_position = input_ids.len();

            for step in 0..max_tokens {
                if is_cancelled() {
                    break;
                }
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_with_prompt_embeddings_and_rng<R: Rng + ?Sized>(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA sampled multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA sampled multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA multimodal prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits =
                self.full_context_logits_from_hidden_device(hidden, Some(&mut cache))?;
            let mut generated = Vec::new();
            let mut next_position = prompt_rows;

            for step in 0..max_tokens {
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_with_prompt_embeddings_positions_and_rng<R: Rng + ?Sized>(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA sampled multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            if next_rope_position > dims.context {
                bail!(
                    "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA sampled multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA multimodal prompt embeddings",
            )?;
            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits = self.full_context_logits_from_hidden_device_with_position_ids(
                hidden,
                position_ids,
                Some(&mut cache),
            )?;
            let mut generated = Vec::new();
            let mut next_cache_position = prompt_rows;
            let mut next_rope_position = next_rope_position;

            for step in 0..max_tokens {
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_cache_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_cache_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device_with_rope_position(
                    next,
                    next_cache_position,
                    next_rope_position,
                    &mut cache,
                )?;
                next_cache_position += 1;
                next_rope_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prefix_embeddings_paged_and_stop_sequences(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_prefix_embeddings_paged_and_rng(
                    prefix_embeddings,
                    prefix_rows,
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_prefix_embeddings_paged_and_rng(
                    prefix_embeddings,
                    prefix_rows,
                    input_ids,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_paged_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_prompt_embeddings_paged_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_prompt_embeddings_paged_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_positions_paged_and_stop_sequences(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            seed: Option<u64>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            if let Some(seed) = seed {
                let mut rng = StdRng::seed_from_u64(seed);
                self.generate_sampled_tokens_with_prompt_embeddings_positions_paged_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    position_ids,
                    next_rope_position,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                )
            } else {
                let mut rng = rand::thread_rng();
                self.generate_sampled_tokens_with_prompt_embeddings_positions_paged_and_rng(
                    prompt_embeddings,
                    prompt_rows,
                    position_ids,
                    next_rope_position,
                    max_tokens,
                    eos_token_id,
                    temperature,
                    top_p,
                    top_k,
                    page_size,
                    stop_token_sequences,
                    &mut rng,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_with_prefix_embeddings_paged_and_rng<R: Rng + ?Sized>(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA paged sampled multimodal generation",
                max_tokens,
            )?;
            if input_ids.is_empty() {
                bail!("CUDA paged sampled multimodal generation requires at least one input token");
            }
            let input_len = prefix_rows
                .checked_add(input_ids.len())
                .context("CUDA paged sampled multimodal input length overflows usize")?;
            if input_len > dims.context {
                bail!(
                    "multimodal input length {input_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            validate_batched_generation_context_budget(
                "CUDA paged sampled multimodal generation",
                input_len,
                max_tokens,
                dims.context,
            )?;

            let token_capacity = input_len
                .checked_add(max_tokens)
                .context("CUDA paged sampled multimodal token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits = self.full_context_logits_device_with_paged_prefix_cache(
                prefix_embeddings,
                prefix_rows,
                input_ids,
                &mut cache,
            )?;
            let mut generated = Vec::new();
            let mut next_position = input_len;

            for step in 0..max_tokens {
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_with_prompt_embeddings_paged_and_rng<R: Rng + ?Sized>(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA paged sampled multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged sampled multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA paged multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_tokens)
                .context("CUDA paged sampled multimodal prompt token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits =
                self.full_context_logits_from_hidden_device_with_paged_cache(hidden, &mut cache)?;
            let mut generated = Vec::new();
            let mut next_position = prompt_rows;

            for step in 0..max_tokens {
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_with_prompt_embeddings_positions_paged_and_rng<
            R: Rng + ?Sized,
        >(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            position_ids: &[[u32; 3]],
            next_rope_position: usize,
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            page_size: usize,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens = validate_generation_max_tokens(
                "CUDA paged sampled multimodal prompt generation",
                max_tokens,
            )?;
            self.validate_prompt_embeddings(prompt_embeddings, prompt_rows, dims.embed)?;
            if prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            if next_rope_position > dims.context {
                bail!(
                    "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA paged sampled multimodal prompt generation",
                prompt_rows,
                max_tokens,
                dims.context,
            )?;

            let hidden = self.f32_tensor_from_host(
                prompt_embeddings,
                prompt_rows,
                dims.embed,
                "CUDA paged multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_tokens)
                .context("CUDA paged sampled multimodal prompt token capacity overflows usize")?
                .min(dims.context);
            let mut cache = CudaPagedKvCache::new_for_token_capacity(
                self.config.block_count,
                &dims,
                page_size,
                token_capacity,
                &self.stream,
            )?;
            let mut logits = self
                .full_context_logits_from_hidden_device_with_position_ids_paged_cache(
                    hidden,
                    position_ids,
                    &mut cache,
                )?;
            let mut generated = Vec::new();
            let mut next_cache_position = prompt_rows;
            let mut next_rope_position = next_rope_position;

            for step in 0..max_tokens {
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_cache_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_cache_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_paged_device_with_rope_position(
                    next,
                    next_cache_position,
                    next_rope_position,
                    &mut cache,
                )?;
                next_cache_position += 1;
                next_rope_position += 1;
            }

            Ok(generated)
        }

        #[allow(clippy::too_many_arguments)]
        fn generate_sampled_tokens_with_prefix_embeddings_and_rng<R: Rng + ?Sized>(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            input_ids: &[u32],
            max_tokens: usize,
            eos_token_id: Option<u32>,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            stop_token_sequences: &[Vec<u32>],
            rng: &mut R,
        ) -> Result<Vec<u32>> {
            let dims = self.qwen_dims()?;
            let max_tokens =
                validate_generation_max_tokens("CUDA sampled multimodal generation", max_tokens)?;
            if input_ids.is_empty() {
                bail!("CUDA sampled multimodal generation requires at least one input token");
            }
            let input_len = prefix_rows
                .checked_add(input_ids.len())
                .context("CUDA multimodal input length overflows usize")?;
            if input_len > dims.context {
                bail!(
                    "multimodal input length {input_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            validate_batched_generation_context_budget(
                "CUDA sampled multimodal generation",
                input_len,
                max_tokens,
                dims.context,
            )?;

            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let mut logits = self.full_context_logits_device_with_prefix_cache(
                prefix_embeddings,
                prefix_rows,
                input_ids,
                Some(&mut cache),
            )?;
            let mut generated = Vec::new();
            let mut next_position = input_len;

            for step in 0..max_tokens {
                let next = self.sample_last_row(&logits, temperature, top_p, top_k, rng)?;
                generated.push(next);
                if is_stop_sequence(next, &generated, eos_token_id, stop_token_sequences) {
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                if next_position >= dims.context {
                    bail!(
                        "generation context length {} exceeds qwen context length {}",
                        next_position + 1,
                        dims.context
                    );
                }
                logits = self.decode_one_logits_device(next, next_position, &mut cache)?;
                next_position += 1;
            }

            Ok(generated)
        }

        pub fn kv_decode_logits_host(&self, prefix: &[u32], token_id: u32) -> Result<Vec<f32>> {
            let dims = self.qwen_dims()?;
            if prefix.is_empty() {
                bail!("CUDA KV decode parity requires a non-empty prefix");
            }
            if prefix.len() >= dims.context {
                bail!(
                    "prefix length {} leaves no room for a decode token in context length {}",
                    prefix.len(),
                    dims.context
                );
            }
            let mut cache = CudaKvCache::new(self.config.block_count, &dims, &self.stream)?;
            let _ = self.full_context_logits_device_with_cache(prefix, Some(&mut cache))?;
            let logits = self.decode_one_logits_device(token_id, prefix.len(), &mut cache)?;
            self.op_barrier()?;
            logits.copy_to_host()
        }

        pub fn paged_kv_decode_logits_host(
            &self,
            prefix: &[u32],
            token_id: u32,
            page_size: usize,
        ) -> Result<Vec<f32>> {
            let dims = self.qwen_dims()?;
            if prefix.is_empty() {
                bail!("CUDA paged KV decode parity requires a non-empty prefix");
            }
            if prefix.len() >= dims.context {
                bail!(
                    "prefix length {} leaves no room for a decode token in context length {}",
                    prefix.len(),
                    dims.context
                );
            }
            let mut cache =
                CudaPagedKvCache::new(self.config.block_count, &dims, page_size, &self.stream)?;
            let _ = self.full_context_logits_device_with_paged_cache(prefix, &mut cache)?;
            let logits = self.decode_one_logits_paged_device(token_id, prefix.len(), &mut cache)?;
            self.op_barrier()?;
            logits.copy_to_host()
        }

        fn argmax_last_row(&self, logits: &GpuF32Tensor) -> Result<u32> {
            let token = DeviceBuffer::alloc(std::mem::size_of::<u32>())
                .context("allocating CUDA argmax token")?;
            crate::kernels::launch_argmax_last_row(
                &logits.buffer,
                &token,
                logits.rows,
                logits.cols,
                &self.stream,
            )?;
            self.op_barrier()?;
            token
                .copy_to_host::<u32>(1)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("CUDA argmax returned no token"))
        }

        fn sample_last_row<R: Rng + ?Sized>(
            &self,
            logits: &GpuF32Tensor,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            rng: &mut R,
        ) -> Result<u32> {
            let sample = rng.gen_range(0.0f32..1.0f32);
            if sampled_selection_needs_host_rank(temperature, top_p, top_k) {
                if logits.rows == 0 || logits.cols == 0 {
                    bail!(
                        "CUDA sampled token selection requires non-empty logits, got {}x{}",
                        logits.rows,
                        logits.cols
                    );
                }
                let row = self.copy_row_f32_device(logits, logits.rows - 1)?;
                let row = row.copy_to_host()?;
                return sample_host_ranked_logits_with_uniform(
                    &row,
                    temperature,
                    top_p,
                    top_k,
                    sample,
                );
            }
            let token = DeviceBuffer::alloc(std::mem::size_of::<u32>())
                .context("allocating CUDA sampled token")?;
            crate::kernels::launch_sample_last_row(
                &logits.buffer,
                &token,
                logits.rows,
                logits.cols,
                temperature,
                top_p,
                top_k,
                sample,
                &self.stream,
            )?;
            self.op_barrier()?;
            token
                .copy_to_host::<u32>(1)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("CUDA sampled token kernel returned no token"))
        }

        fn sample_last_row_with_sample(
            &self,
            logits: &GpuF32Tensor,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            sample: f32,
        ) -> Result<u32> {
            if sampled_selection_needs_host_rank(temperature, top_p, top_k) {
                if logits.rows == 0 || logits.cols == 0 {
                    bail!(
                        "CUDA sampled token selection requires non-empty logits, got {}x{}",
                        logits.rows,
                        logits.cols
                    );
                }
                let row = self.copy_row_f32_device(logits, logits.rows - 1)?;
                let row = row.copy_to_host()?;
                return sample_host_ranked_logits_with_uniform(
                    &row,
                    temperature,
                    top_p,
                    top_k,
                    sample,
                );
            }
            let token = DeviceBuffer::alloc(std::mem::size_of::<u32>())
                .context("allocating CUDA sampled token")?;
            crate::kernels::launch_sample_last_row(
                &logits.buffer,
                &token,
                logits.rows,
                logits.cols,
                temperature,
                top_p,
                top_k,
                sample,
                &self.stream,
            )?;
            self.op_barrier()?;
            token
                .copy_to_host::<u32>(1)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("CUDA sampled token kernel returned no token"))
        }

        fn argmax_batched_last_token(
            &self,
            logits: &GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
        ) -> Result<Vec<u32>> {
            if logits.rows != batch_count * seq_len {
                bail!(
                    "CUDA batched argmax logits rows {} do not match batch {batch_count} x seq {seq_len}",
                    logits.rows
                );
            }
            let tokens = DeviceBuffer::alloc(batch_count * std::mem::size_of::<u32>())
                .context("allocating CUDA batched argmax tokens")?;
            crate::kernels::launch_argmax_batched_last_token(
                &logits.buffer,
                &tokens,
                batch_count,
                seq_len,
                logits.cols,
                &self.stream,
            )?;
            self.op_barrier()?;
            tokens.copy_to_host(batch_count)
        }

        fn sample_batched_last_token(
            &self,
            logits: &GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            temperature: f32,
            top_p: f32,
            top_k: Option<u32>,
            samples: &[f32],
        ) -> Result<Vec<u32>> {
            if logits.rows != batch_count * seq_len {
                bail!(
                    "CUDA batched sampler logits rows {} do not match batch {batch_count} x seq {seq_len}",
                    logits.rows
                );
            }
            if samples.len() != batch_count {
                bail!(
                    "CUDA batched sampler got {} random samples for {batch_count} requests",
                    samples.len()
                );
            }
            if batch_count == 0 || seq_len == 0 || logits.cols == 0 {
                bail!(
                    "CUDA batched sampled token selection requires non-empty logits, got batch={batch_count}, seq_len={seq_len}, cols={}",
                    logits.cols
                );
            }
            if sampled_selection_needs_host_rank(temperature, top_p, top_k) {
                let mut tokens = Vec::with_capacity(batch_count);
                for (batch, sample) in samples.iter().copied().enumerate() {
                    let row_idx = batch
                        .checked_mul(seq_len)
                        .and_then(|offset| offset.checked_add(seq_len - 1))
                        .context("CUDA batched sampler row index overflows usize")?;
                    let row = self.copy_row_f32_device(logits, row_idx)?;
                    let row = row.copy_to_host()?;
                    tokens.push(sample_host_ranked_logits_with_uniform(
                        &row,
                        temperature,
                        top_p,
                        top_k,
                        sample,
                    )?);
                }
                return Ok(tokens);
            }
            let tokens = DeviceBuffer::alloc(batch_count * std::mem::size_of::<u32>())
                .context("allocating CUDA batched sampled tokens")?;
            let sample_buffer = DeviceBuffer::alloc(batch_count * std::mem::size_of::<f32>())
                .context("allocating CUDA batched sampler samples")?;
            sample_buffer
                .copy_from_host(samples)
                .context("copying CUDA batched sampler samples")?;
            crate::kernels::launch_sample_batched_last_token(
                &logits.buffer,
                &tokens,
                &sample_buffer,
                batch_count,
                seq_len,
                logits.cols,
                temperature,
                top_p,
                top_k,
                &self.stream,
            )?;
            self.op_barrier()?;
            tokens.copy_to_host(batch_count)
        }

        fn full_context_logits_device(&self, token_ids: &[u32]) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            self.full_context_logits_device_with_cache(token_ids, None)
        }

        fn f32_tensor_from_host(
            &self,
            values: &[f32],
            rows: usize,
            cols: usize,
            operation: &str,
        ) -> Result<GpuF32Tensor> {
            let expected = rows
                .checked_mul(cols)
                .ok_or_else(|| anyhow!("{operation} element count overflows usize"))?;
            if values.len() != expected {
                bail!(
                    "{operation} got {} f32 values; expected {rows} x {cols} = {expected}",
                    values.len()
                );
            }
            let byte_len = expected
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| anyhow!("{operation} byte count overflows usize"))?;
            let buffer =
                DeviceBuffer::alloc(byte_len).with_context(|| format!("allocating {operation}"))?;
            buffer
                .copy_from_host(values)
                .with_context(|| format!("copying {operation}"))?;
            Ok(GpuF32Tensor { rows, cols, buffer })
        }

        fn validate_prefix_embeddings(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            embed_dim: usize,
        ) -> Result<()> {
            if prefix_rows == 0 {
                bail!("CUDA multimodal prefix generation requires at least one prefix row");
            }
            let expected = prefix_rows
                .checked_mul(embed_dim)
                .context("CUDA multimodal prefix embedding element count overflows usize")?;
            if prefix_embeddings.len() != expected {
                bail!(
                    "CUDA multimodal prefix has {} values; expected {prefix_rows} row(s) x embedding dim {embed_dim} = {expected}",
                    prefix_embeddings.len()
                );
            }
            Ok(())
        }

        fn validate_prompt_embeddings(
            &self,
            prompt_embeddings: &[f32],
            prompt_rows: usize,
            embed_dim: usize,
        ) -> Result<()> {
            if prompt_rows == 0 {
                bail!("CUDA multimodal prompt generation requires at least one prompt row");
            }
            let expected = prompt_rows
                .checked_mul(embed_dim)
                .context("CUDA multimodal prompt embedding element count overflows usize")?;
            if prompt_embeddings.len() != expected {
                bail!(
                    "CUDA multimodal prompt has {} values; expected {prompt_rows} row(s) x embedding dim {embed_dim} = {expected}",
                    prompt_embeddings.len()
                );
            }
            Ok(())
        }

        fn mrope_sections(&self, head_dim: usize) -> Result<Option<[usize; 4]>> {
            let Some(sections) = self.config.rope_dimension_sections else {
                return Ok(None);
            };
            let sections = [
                usize::try_from(sections[0])
                    .context("MRoPE temporal section does not fit usize")?,
                usize::try_from(sections[1]).context("MRoPE height section does not fit usize")?,
                usize::try_from(sections[2]).context("MRoPE width section does not fit usize")?,
                usize::try_from(sections[3]).context("MRoPE extra section does not fit usize")?,
            ];
            let section_sum = sections
                .iter()
                .try_fold(0usize, |acc, value| acc.checked_add(*value))
                .context("MRoPE section sum overflows usize")?;
            if section_sum == 0 {
                bail!("MRoPE dimension sections must not all be zero");
            }
            let rotary_pairs = head_dim / 2;
            if section_sum > rotary_pairs {
                bail!(
                    "MRoPE dimension sections sum {section_sum} exceeds rotary pair count {rotary_pairs}"
                );
            }
            Ok(Some(sections))
        }

        fn mrope_positions_device(
            &self,
            position_ids: &[[u32; 3]],
            expected_rows: usize,
        ) -> Result<MropePositionsDevice> {
            if position_ids.len() != expected_rows {
                bail!(
                    "CUDA MRoPE got {} position rows; expected {expected_rows}",
                    position_ids.len()
                );
            }
            let mut t = Vec::with_capacity(position_ids.len());
            let mut h = Vec::with_capacity(position_ids.len());
            let mut w = Vec::with_capacity(position_ids.len());
            for position in position_ids {
                t.push(position[0]);
                h.push(position[1]);
                w.push(position[2]);
            }
            let bytes = position_ids
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .context("CUDA MRoPE position byte count overflows usize")?;
            let pos_t = DeviceBuffer::alloc(bytes).context("allocating CUDA MRoPE t positions")?;
            let pos_h = DeviceBuffer::alloc(bytes).context("allocating CUDA MRoPE h positions")?;
            let pos_w = DeviceBuffer::alloc(bytes).context("allocating CUDA MRoPE w positions")?;
            pos_t
                .copy_from_host(&t)
                .context("copying CUDA MRoPE t positions")?;
            pos_h
                .copy_from_host(&h)
                .context("copying CUDA MRoPE h positions")?;
            pos_w
                .copy_from_host(&w)
                .context("copying CUDA MRoPE w positions")?;
            Ok(MropePositionsDevice {
                rows: position_ids.len(),
                t: pos_t,
                h: pos_h,
                w: pos_w,
                host: position_ids.to_vec(),
            })
        }

        fn full_context_logits_device_with_prefix_cache(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            token_ids: &[u32],
            cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let cache = cache.map(|cache| cache as &mut dyn CudaSingleKvCacheWrite);
            self.full_context_logits_device_with_prefix_cache_writer(
                prefix_embeddings,
                prefix_rows,
                token_ids,
                cache,
            )
        }

        fn full_context_logits_device_with_paged_prefix_cache(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            token_ids: &[u32],
            cache: &mut CudaPagedKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            self.full_context_logits_device_with_prefix_cache_writer(
                prefix_embeddings,
                prefix_rows,
                token_ids,
                Some(cache as &mut dyn CudaSingleKvCacheWrite),
            )
        }

        fn full_context_logits_device_with_prefix_cache_writer(
            &self,
            prefix_embeddings: &[f32],
            prefix_rows: usize,
            token_ids: &[u32],
            cache: Option<&mut dyn CudaSingleKvCacheWrite>,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if token_ids.is_empty() {
                bail!("CUDA multimodal Qwen forward requires at least one token");
            }
            self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            let total_rows = prefix_rows
                .checked_add(token_ids.len())
                .context("CUDA multimodal context row count overflows usize")?;
            if total_rows > dims.context {
                bail!(
                    "multimodal input length {total_rows} exceeds qwen context length {}",
                    dims.context
                );
            }

            let token_embeddings = self.embed_tokens_device(token_ids)?;
            let token_embeddings = token_embeddings.copy_to_host()?;
            let mut combined = Vec::with_capacity(
                total_rows
                    .checked_mul(dims.embed)
                    .context("CUDA multimodal combined embedding count overflows usize")?,
            );
            combined.extend_from_slice(prefix_embeddings);
            combined.extend_from_slice(&token_embeddings);
            let hidden = self.f32_tensor_from_host(
                &combined,
                total_rows,
                dims.embed,
                "CUDA multimodal combined embeddings",
            )?;
            self.full_context_logits_from_hidden_device_with_mrope(hidden, None, cache)
        }

        fn full_context_logits_from_hidden_device(
            &self,
            hidden: GpuF32Tensor,
            cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let cache = cache.map(|cache| cache as &mut dyn CudaSingleKvCacheWrite);
            self.full_context_logits_from_hidden_device_with_mrope(hidden, None, cache)
        }

        fn full_context_logits_from_hidden_device_with_position_ids(
            &self,
            hidden: GpuF32Tensor,
            position_ids: &[[u32; 3]],
            cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let positions = self.mrope_positions_device(position_ids, hidden.rows)?;
            let cache = cache.map(|cache| cache as &mut dyn CudaSingleKvCacheWrite);
            self.full_context_logits_from_hidden_device_with_mrope(hidden, Some(&positions), cache)
        }

        fn full_context_logits_from_hidden_device_with_paged_cache(
            &self,
            hidden: GpuF32Tensor,
            cache: &mut CudaPagedKvCache,
        ) -> Result<GpuF32Tensor> {
            self.full_context_logits_from_hidden_device_with_mrope(
                hidden,
                None,
                Some(cache as &mut dyn CudaSingleKvCacheWrite),
            )
        }

        fn full_context_logits_from_hidden_device_with_position_ids_paged_cache(
            &self,
            hidden: GpuF32Tensor,
            position_ids: &[[u32; 3]],
            cache: &mut CudaPagedKvCache,
        ) -> Result<GpuF32Tensor> {
            let positions = self.mrope_positions_device(position_ids, hidden.rows)?;
            self.full_context_logits_from_hidden_device_with_mrope(
                hidden,
                Some(&positions),
                Some(cache as &mut dyn CudaSingleKvCacheWrite),
            )
        }

        fn full_context_logits_from_hidden_device_with_mrope(
            &self,
            mut hidden: GpuF32Tensor,
            mrope_positions: Option<&MropePositionsDevice>,
            mut cache: Option<&mut dyn CudaSingleKvCacheWrite>,
        ) -> Result<GpuF32Tensor> {
            let dims = self.qwen_dims()?;
            if hidden.rows == 0 {
                bail!("CUDA Qwen forward requires at least one hidden row");
            }
            if hidden.cols != dims.embed {
                bail!(
                    "CUDA Qwen hidden dim {} does not match embedding dim {}",
                    hidden.cols,
                    dims.embed
                );
            }
            if hidden.rows > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    hidden.rows,
                    dims.context
                );
            }
            let seq_len = hidden.rows;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);
            let mrope_sections = if mrope_positions.is_some() {
                let head_dim = if self.config.attention_mla_tensor_layout {
                    qwen_mla_dims(&self.config)?
                        .map(|mla| mla.qk_rope_head_dim)
                        .unwrap_or(dims.head_dim)
                } else {
                    dims.head_dim
                };
                Some(self.mrope_sections(head_dim)?.ok_or_else(|| {
                    anyhow!(
                        "CUDA multimodal MRoPE positions require GGUF rope.dimension_sections metadata"
                    )
                })?)
            } else {
                None
            };

            for layer in 0..self.config.block_count {
                let prefix = format!("blk.{layer}");
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                if self.layer_uses_recurrent_ssm(&prefix) {
                    let ssm_out = self.recurrent_ssm_f32_device(&prefix, &attn_input, eps)?;
                    hidden = self.add_f32_device(&hidden, &ssm_out)?;
                } else {
                    self.ensure_layer_runtime_supported(&prefix)?;
                    let (q, k, v, gate) = if self.layer_uses_mla_attention(&prefix) {
                        let rope = match (mrope_positions, mrope_sections) {
                            (Some(positions), Some(sections)) => QwenRopeLayout::Mrope {
                                positions,
                                sections,
                            },
                            _ => QwenRopeLayout::Single {
                                seq_len,
                                position_offset: 0,
                            },
                        };
                        self.attention_qkv_f32_device(
                            &prefix,
                            &attn_input,
                            &dims,
                            eps,
                            rope_base,
                            rope_scale,
                            rope,
                        )?
                    } else {
                        let (q, gate) =
                            self.dense_attention_q_f32_device(&prefix, &attn_input, &dims, eps)?;
                        if let (Some(positions), Some(sections)) = (mrope_positions, mrope_sections)
                        {
                            self.apply_mrope_f32_device(
                                &q,
                                positions,
                                dims.heads,
                                dims.head_dim,
                                rope_base,
                                rope_scale,
                                sections,
                                true,
                            )?;
                        } else {
                            self.apply_rope_f32_device(
                                &q,
                                seq_len,
                                dims.heads,
                                dims.head_dim,
                                rope_base,
                                rope_scale,
                                0,
                                true,
                            )?;
                        }

                        let k = self
                            .project_f32_device(&format!("{prefix}.attn_k.weight"), &attn_input)?;
                        let k = self
                            .add_optional_rowwise_f32_device(k, &format!("{prefix}.attn_k.bias"))?;
                        let k = self.optional_head_rms_norm_f32_device(
                            k,
                            &format!("{prefix}.attn_k_norm.weight"),
                            seq_len,
                            dims.kv_heads,
                            dims.head_dim,
                            eps,
                        )?;
                        if let (Some(positions), Some(sections)) = (mrope_positions, mrope_sections)
                        {
                            self.apply_mrope_f32_device(
                                &k,
                                positions,
                                dims.kv_heads,
                                dims.head_dim,
                                rope_base,
                                rope_scale,
                                sections,
                                true,
                            )?;
                        } else {
                            self.apply_rope_f32_device(
                                &k,
                                seq_len,
                                dims.kv_heads,
                                dims.head_dim,
                                rope_base,
                                rope_scale,
                                0,
                                true,
                            )?;
                        }

                        let v = self
                            .project_f32_device(&format!("{prefix}.attn_v.weight"), &attn_input)?;
                        let v = self
                            .add_optional_rowwise_f32_device(v, &format!("{prefix}.attn_v.bias"))?;
                        (q, k, v, gate)
                    };
                    if let Some(cache) = cache.as_deref_mut() {
                        cache.write_layer(
                            usize::try_from(layer)
                                .context("qwen layer index does not fit usize")?,
                            &k,
                            &v,
                            0,
                            dims.kv_heads,
                            dims.head_dim,
                            dims.v_head_dim,
                            &self.stream,
                        )?;
                    }
                    let attn = self.causal_attention_f32_device(
                        &q,
                        &k,
                        &v,
                        seq_len,
                        dims.heads,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        self.layer_attention_window(&prefix),
                    )?;
                    let attn_out =
                        self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                    hidden = self.add_f32_device(&hidden, &attn_out)?;
                }

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn full_context_logits_device_batched(
            &self,
            token_ids: &[u32],
            batch_count: usize,
            seq_len: usize,
            mut cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if batch_count == 0 || seq_len == 0 {
                bail!("CUDA batched Qwen forward requires non-empty batch and sequence");
            }
            if token_ids.len() != batch_count * seq_len {
                bail!(
                    "CUDA batched Qwen forward got {} tokens; expected batch {batch_count} x seq {seq_len}",
                    token_ids.len()
                );
            }
            if seq_len > dims.context {
                bail!(
                    "input length {seq_len} exceeds qwen context length {}",
                    dims.context
                );
            }

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::Batched {
                        batch_count,
                        seq_len,
                        position_offset: 0,
                    },
                )?;
                if let Some(cache) = cache.as_deref_mut() {
                    cache.write_layer_batched(
                        usize::try_from(layer).context("qwen layer index does not fit usize")?,
                        &k,
                        &v,
                        batch_count,
                        seq_len,
                        0,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        &self.stream,
                    )?;
                }
                let attn = self.causal_attention_batched_f32_device(
                    &q,
                    &k,
                    &v,
                    batch_count,
                    seq_len,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn batched_prefix_prompt_embeddings_host(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            dims: &QwenDims,
        ) -> Result<Vec<f32>> {
            let batch_count = inputs.len();
            if batch_count == 0 {
                bail!("CUDA batched multimodal prompt embedding build requires a non-empty batch");
            }
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!("CUDA batched multimodal prompt embedding build requires non-empty prompts");
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!(
                    "CUDA batched multimodal prompt embedding build requires equal prompt lengths"
                );
            }
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA batched multimodal prompt embedding build got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }

            let mut flat_tokens = Vec::with_capacity(batch_count * token_prompt_len);
            for input in inputs {
                flat_tokens.extend_from_slice(input);
            }
            let token_embeddings = self.embed_tokens_device(&flat_tokens)?.copy_to_host()?;
            let prompt_len = prefix_rows
                .checked_add(token_prompt_len)
                .context("CUDA batched multimodal prompt length overflows usize")?;
            let total_values = batch_count
                .checked_mul(prompt_len)
                .and_then(|value| value.checked_mul(dims.embed))
                .context("CUDA batched multimodal hidden value count overflows usize")?;
            let mut hidden = Vec::with_capacity(total_values);
            for (idx, prefix_embeddings) in prefix_embeddings_per_request.iter().enumerate() {
                hidden.extend_from_slice(prefix_embeddings);
                let token_start = idx
                    .checked_mul(token_prompt_len)
                    .and_then(|value| value.checked_mul(dims.embed))
                    .context("CUDA batched token embedding offset overflows usize")?;
                let token_len = token_prompt_len
                    .checked_mul(dims.embed)
                    .context("CUDA batched token embedding length overflows usize")?;
                let token_end = token_start
                    .checked_add(token_len)
                    .context("CUDA batched token embedding end overflows usize")?;
                hidden.extend_from_slice(
                    token_embeddings
                        .get(token_start..token_end)
                        .ok_or_else(|| anyhow!("CUDA batched token embeddings ended early"))?,
                );
            }
            if hidden.len() != total_values {
                bail!(
                    "CUDA batched multimodal hidden build produced {} values; expected {total_values}",
                    hidden.len()
                );
            }
            Ok(hidden)
        }

        fn full_context_logits_from_hidden_device_batched(
            &self,
            hidden: GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let cache = cache.map(|cache| cache as &mut dyn CudaBatchedKvCacheWrite);
            self.full_context_logits_from_hidden_device_batched_with_mrope(
                hidden,
                batch_count,
                seq_len,
                None,
                cache,
            )
        }

        fn full_context_logits_from_hidden_device_batched_with_position_ids(
            &self,
            hidden: GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            position_ids: &[[u32; 3]],
            cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let positions = self.mrope_positions_device(position_ids, hidden.rows)?;
            let cache = cache.map(|cache| cache as &mut dyn CudaBatchedKvCacheWrite);
            self.full_context_logits_from_hidden_device_batched_with_mrope(
                hidden,
                batch_count,
                seq_len,
                Some(&positions),
                cache,
            )
        }

        fn full_context_logits_from_hidden_device_batched_paged_cache(
            &self,
            hidden: GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            cache: &mut CudaPagedBatchKvCache,
        ) -> Result<GpuF32Tensor> {
            self.full_context_logits_from_hidden_device_batched_with_mrope(
                hidden,
                batch_count,
                seq_len,
                None,
                Some(cache as &mut dyn CudaBatchedKvCacheWrite),
            )
        }

        fn full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
            &self,
            hidden: GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            position_ids: &[[u32; 3]],
            cache: &mut CudaPagedBatchKvCache,
        ) -> Result<GpuF32Tensor> {
            let positions = self.mrope_positions_device(position_ids, hidden.rows)?;
            self.full_context_logits_from_hidden_device_batched_with_mrope(
                hidden,
                batch_count,
                seq_len,
                Some(&positions),
                Some(cache as &mut dyn CudaBatchedKvCacheWrite),
            )
        }

        fn full_context_logits_from_hidden_device_batched_with_mrope(
            &self,
            mut hidden: GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            mrope_positions: Option<&MropePositionsDevice>,
            mut cache: Option<&mut dyn CudaBatchedKvCacheWrite>,
        ) -> Result<GpuF32Tensor> {
            let dims = self.qwen_dims()?;
            if batch_count == 0 || seq_len == 0 {
                bail!("CUDA batched Qwen hidden forward requires non-empty batch and sequence");
            }
            if hidden.rows != batch_count * seq_len || hidden.cols != dims.embed {
                bail!(
                    "CUDA batched Qwen hidden forward got hidden shape {}x{}; expected {}x{}",
                    hidden.rows,
                    hidden.cols,
                    batch_count * seq_len,
                    dims.embed
                );
            }
            if seq_len > dims.context {
                bail!(
                    "input length {seq_len} exceeds qwen context length {}",
                    dims.context
                );
            }

            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);
            let mrope_sections = if mrope_positions.is_some() {
                let head_dim = if self.config.attention_mla_tensor_layout {
                    qwen_mla_dims(&self.config)?
                        .map(|mla| mla.qk_rope_head_dim)
                        .unwrap_or(dims.head_dim)
                } else {
                    dims.head_dim
                };
                Some(self.mrope_sections(head_dim)?.ok_or_else(|| {
                    anyhow!(
                        "CUDA batched multimodal MRoPE positions require GGUF rope.dimension_sections metadata"
                    )
                })?)
            } else {
                None
            };

            for layer in 0..self.config.block_count {
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = if self.layer_uses_mla_attention(&prefix) {
                    let rope = match (mrope_positions, mrope_sections) {
                        (Some(positions), Some(sections)) => QwenRopeLayout::Mrope {
                            positions,
                            sections,
                        },
                        _ => QwenRopeLayout::Batched {
                            batch_count,
                            seq_len,
                            position_offset: 0,
                        },
                    };
                    self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        rope,
                    )?
                } else {
                    let (q, gate) =
                        self.dense_attention_q_f32_device(&prefix, &attn_input, &dims, eps)?;
                    if let (Some(positions), Some(sections)) = (mrope_positions, mrope_sections) {
                        self.apply_mrope_f32_device(
                            &q,
                            positions,
                            dims.heads,
                            dims.head_dim,
                            rope_base,
                            rope_scale,
                            sections,
                            true,
                        )?;
                    } else {
                        self.apply_rope_batched_f32_device(
                            &q,
                            batch_count,
                            seq_len,
                            dims.heads,
                            dims.head_dim,
                            rope_base,
                            rope_scale,
                            0,
                            true,
                        )?;
                    }

                    let k =
                        self.project_f32_device(&format!("{prefix}.attn_k.weight"), &attn_input)?;
                    let k =
                        self.add_optional_rowwise_f32_device(k, &format!("{prefix}.attn_k.bias"))?;
                    let k = self.optional_head_rms_norm_f32_device(
                        k,
                        &format!("{prefix}.attn_k_norm.weight"),
                        batch_count * seq_len,
                        dims.kv_heads,
                        dims.head_dim,
                        eps,
                    )?;
                    if let (Some(positions), Some(sections)) = (mrope_positions, mrope_sections) {
                        self.apply_mrope_f32_device(
                            &k,
                            positions,
                            dims.kv_heads,
                            dims.head_dim,
                            rope_base,
                            rope_scale,
                            sections,
                            true,
                        )?;
                    } else {
                        self.apply_rope_batched_f32_device(
                            &k,
                            batch_count,
                            seq_len,
                            dims.kv_heads,
                            dims.head_dim,
                            rope_base,
                            rope_scale,
                            0,
                            true,
                        )?;
                    }

                    let v =
                        self.project_f32_device(&format!("{prefix}.attn_v.weight"), &attn_input)?;
                    let v =
                        self.add_optional_rowwise_f32_device(v, &format!("{prefix}.attn_v.bias"))?;
                    (q, k, v, gate)
                };
                if let Some(cache) = cache.as_deref_mut() {
                    cache.write_layer_batched(
                        usize::try_from(layer).context("qwen layer index does not fit usize")?,
                        &k,
                        &v,
                        batch_count,
                        seq_len,
                        0,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        &self.stream,
                    )?;
                }
                let attn = self.causal_attention_batched_f32_device(
                    &q,
                    &k,
                    &v,
                    batch_count,
                    seq_len,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn full_context_logits_device_batched_paged_cache_with_shared_prefix<'a>(
            &self,
            inputs: &[Vec<u32>],
            page_size: usize,
            token_capacity: usize,
            page_tables: &[Vec<usize>],
            pool: &'a CudaPagedBatchDevicePool,
        ) -> Result<(CudaPagedBatchKvCache<'a>, GpuF32Tensor, usize)> {
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            if batch_count == 0 {
                bail!("CUDA paged batched shared-prefix prefill requires a non-empty batch");
            }
            let prompt_len = inputs[0].len();
            if prompt_len == 0 {
                bail!("CUDA paged batched shared-prefix prefill requires non-empty prompts");
            }
            if inputs.iter().any(|input| input.len() != prompt_len) {
                bail!("CUDA paged batched shared-prefix prefill requires equal prompt lengths");
            }

            let shared_prefix_len = shared_prefix_token_len(inputs);
            if shared_prefix_len == 0 || shared_prefix_len >= prompt_len {
                let mut flat = Vec::with_capacity(batch_count * prompt_len);
                for input in inputs {
                    flat.extend_from_slice(input);
                }
                let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                    &dims,
                    batch_count,
                    page_size,
                    token_capacity,
                    page_tables,
                    pool,
                    &self.stream,
                )?;
                // Prefill only needs the last position's logits to pick the
                // first generated token; project just that row (returned as
                // logits_seq_len = 1) to avoid a vocab-wide buffer per prompt
                // token.
                let logits = self.full_context_logits_device_batched_paged_cache(
                    &flat,
                    batch_count,
                    prompt_len,
                    &mut cache,
                    true,
                )?;
                return Ok((cache, logits, 1));
            }

            let prefix_page_tables = vec![page_tables[0].clone()];
            {
                let mut prefix_cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                    &dims,
                    1,
                    page_size,
                    token_capacity,
                    &prefix_page_tables,
                    pool,
                    &self.stream,
                )?;
                // Only the KV matters here (logits discarded), so project the
                // cheap last row.
                let _ = self.full_context_logits_device_batched_paged_cache(
                    &inputs[0][..shared_prefix_len],
                    1,
                    shared_prefix_len,
                    &mut prefix_cache,
                    true,
                )?;
            }

            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            cache.copy_prefix_from_first_batch(
                shared_prefix_len,
                dims.kv_heads,
                dims.head_dim,
                dims.v_head_dim,
                &self.stream,
            )?;

            let mut logits = None;
            for position in shared_prefix_len..prompt_len {
                let tokens = inputs
                    .iter()
                    .map(|input| input[position])
                    .collect::<Vec<_>>();
                logits =
                    Some(self.decode_batch_logits_paged_device(&tokens, position, &mut cache)?);
            }
            let logits = logits.ok_or_else(|| {
                anyhow!("CUDA paged batched shared-prefix prefill produced no suffix logits")
            })?;
            Ok((cache, logits, 1))
        }

        fn full_context_logits_device_batched_paged_cache(
            &self,
            token_ids: &[u32],
            batch_count: usize,
            seq_len: usize,
            cache: &mut CudaPagedBatchKvCache,
            last_row_only: bool,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if batch_count == 0 || seq_len == 0 {
                bail!("CUDA paged batched Qwen forward requires non-empty batch and sequence");
            }
            if token_ids.len() != batch_count * seq_len {
                bail!(
                    "CUDA paged batched Qwen forward got {} tokens; expected batch {batch_count} x seq {seq_len}",
                    token_ids.len()
                );
            }
            if seq_len > dims.context {
                bail!(
                    "input length {seq_len} exceeds qwen context length {}",
                    dims.context
                );
            }

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::Batched {
                        batch_count,
                        seq_len,
                        position_offset: 0,
                    },
                )?;
                cache.write_layer_batched(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    seq_len,
                    0,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.causal_attention_batched_f32_device(
                    &q,
                    &k,
                    &v,
                    batch_count,
                    seq_len,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            if last_row_only {
                self.output_logits_last_row_f32_device(&normed, batch_count, seq_len)
            } else {
                self.output_logits_f32_device(&normed)
            }
        }

        fn full_context_logits_device_with_cache(
            &self,
            token_ids: &[u32],
            mut cache: Option<&mut CudaKvCache>,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if token_ids.is_empty() {
                bail!("CUDA Qwen forward requires at least one token");
            }
            if token_ids.len() > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    token_ids.len(),
                    dims.context
                );
            }

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let prefix = format!("blk.{layer}");
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                if self.layer_uses_recurrent_ssm(&prefix) {
                    let ssm_out = self.recurrent_ssm_f32_device(&prefix, &attn_input, eps)?;
                    hidden = self.add_f32_device(&hidden, &ssm_out)?;
                } else {
                    self.ensure_layer_runtime_supported(&prefix)?;
                    let (q, k, v, gate) = self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        QwenRopeLayout::Single {
                            seq_len: token_ids.len(),
                            position_offset: 0,
                        },
                    )?;
                    if let Some(cache) = cache.as_deref_mut() {
                        cache.write_layer(
                            usize::try_from(layer)
                                .context("qwen layer index does not fit usize")?,
                            &k,
                            &v,
                            0,
                            dims.kv_heads,
                            dims.head_dim,
                            dims.v_head_dim,
                            &self.stream,
                        )?;
                    }
                    let attn = self.causal_attention_f32_device(
                        &q,
                        &k,
                        &v,
                        token_ids.len(),
                        dims.heads,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        self.layer_attention_window(&prefix),
                    )?;
                    let attn_out =
                        self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                    hidden = self.add_f32_device(&hidden, &attn_out)?;
                }

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn full_context_logits_device_with_paged_cache(
            &self,
            token_ids: &[u32],
            cache: &mut CudaPagedKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if token_ids.is_empty() {
                bail!("CUDA paged Qwen forward requires at least one token");
            }
            if token_ids.len() > dims.context {
                bail!(
                    "input length {} exceeds qwen context length {}",
                    token_ids.len(),
                    dims.context
                );
            }

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                if self.layer_uses_recurrent_ssm(&prefix) {
                    let ssm_out = self.recurrent_ssm_f32_device(&prefix, &attn_input, eps)?;
                    hidden = self.add_f32_device(&hidden, &ssm_out)?;
                } else {
                    self.ensure_layer_runtime_supported(&prefix)?;
                    let (q, k, v, gate) = self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        QwenRopeLayout::Single {
                            seq_len: token_ids.len(),
                            position_offset: 0,
                        },
                    )?;
                    cache.write_layer(
                        layer_idx,
                        &k,
                        &v,
                        0,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        &self.stream,
                    )?;
                    let attn = self.causal_attention_f32_device(
                        &q,
                        &k,
                        &v,
                        token_ids.len(),
                        dims.heads,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        self.layer_attention_window(&prefix),
                    )?;
                    let attn_out =
                        self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                    hidden = self.add_f32_device(&hidden, &attn_out)?;
                }

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_one_logits_device(
            &self,
            token_id: u32,
            position: usize,
            cache: &mut CudaKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            self.decode_one_logits_device_with_rope_position(token_id, position, position, cache)
        }

        fn decode_one_logits_device_with_rope_position(
            &self,
            token_id: u32,
            cache_position: usize,
            rope_position: usize,
            cache: &mut CudaKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if cache_position >= dims.context {
                bail!(
                    "decode position {cache_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            if rope_position > dims.context {
                bail!(
                    "decode RoPE position {rope_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let mut hidden = self.embed_tokens_device(&[token_id])?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::Single {
                        seq_len: 1,
                        position_offset: rope_position,
                    },
                )?;
                cache.write_layer(
                    layer_idx,
                    &k,
                    &v,
                    cache_position,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.cached_decode_attention_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    cache_position,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_one_logits_paged_device(
            &self,
            token_id: u32,
            position: usize,
            cache: &mut CudaPagedKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            self.decode_one_logits_paged_device_with_rope_position(
                token_id, position, position, cache,
            )
        }

        fn decode_one_logits_paged_device_with_rope_position(
            &self,
            token_id: u32,
            cache_position: usize,
            rope_position: usize,
            cache: &mut CudaPagedKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            if cache_position >= dims.context {
                bail!(
                    "decode position {cache_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            if rope_position > dims.context {
                bail!(
                    "decode RoPE position {rope_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let mut hidden = self.embed_tokens_device(&[token_id])?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::Single {
                        seq_len: 1,
                        position_offset: rope_position,
                    },
                )?;
                cache.write_layer(
                    layer_idx,
                    &k,
                    &v,
                    cache_position,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.paged_decode_attention_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    cache_position,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_batch_logits_device(
            &self,
            token_ids: &[u32],
            position: usize,
            cache: &mut CudaKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("CUDA batched decode requires at least one token");
            }
            if position >= dims.context {
                bail!(
                    "decode position {position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::Batched {
                        batch_count,
                        seq_len: 1,
                        position_offset: position,
                    },
                )?;
                cache.write_layer_batched(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    1,
                    position,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.cached_decode_attention_batched_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    batch_count,
                    position,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_batch_logits_device_with_rope_positions(
            &self,
            token_ids: &[u32],
            cache_position: usize,
            rope_positions: &[usize],
            cache: &mut CudaKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("CUDA batched MRoPE decode requires at least one token");
            }
            if rope_positions.len() != batch_count {
                bail!(
                    "CUDA batched MRoPE decode got {} RoPE positions for {batch_count} tokens",
                    rope_positions.len()
                );
            }
            if cache_position >= dims.context {
                bail!(
                    "decode position {cache_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let mut position_ids = Vec::with_capacity(batch_count);
            for position in rope_positions {
                if *position > dims.context {
                    bail!(
                        "decode RoPE position {position} exceeds qwen context length {}",
                        dims.context
                    );
                }
                let position = u32::try_from(*position)
                    .context("CUDA batched MRoPE decode position does not fit u32")?;
                position_ids.push([position, position, position]);
            }
            let mrope_positions = self.mrope_positions_device(&position_ids, batch_count)?;
            let mrope_head_dim = if self.config.attention_mla_tensor_layout {
                qwen_mla_dims(&self.config)?
                    .map(|mla| mla.qk_rope_head_dim)
                    .unwrap_or(dims.head_dim)
            } else {
                dims.head_dim
            };
            let mrope_sections = self.mrope_sections(mrope_head_dim)?.ok_or_else(|| {
                anyhow!("CUDA batched MRoPE decode requires GGUF rope.dimension_sections metadata")
            })?;

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = if self.layer_uses_mla_attention(&prefix) {
                    self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        QwenRopeLayout::Mrope {
                            positions: &mrope_positions,
                            sections: mrope_sections,
                        },
                    )?
                } else {
                    let (q, gate) =
                        self.dense_attention_q_f32_device(&prefix, &attn_input, &dims, eps)?;
                    self.apply_mrope_f32_device(
                        &q,
                        &mrope_positions,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        mrope_sections,
                        true,
                    )?;

                    let k =
                        self.project_f32_device(&format!("{prefix}.attn_k.weight"), &attn_input)?;
                    let k =
                        self.add_optional_rowwise_f32_device(k, &format!("{prefix}.attn_k.bias"))?;
                    let k = self.optional_head_rms_norm_f32_device(
                        k,
                        &format!("{prefix}.attn_k_norm.weight"),
                        batch_count,
                        dims.kv_heads,
                        dims.head_dim,
                        eps,
                    )?;
                    self.apply_mrope_f32_device(
                        &k,
                        &mrope_positions,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        mrope_sections,
                        true,
                    )?;

                    let v =
                        self.project_f32_device(&format!("{prefix}.attn_v.weight"), &attn_input)?;
                    let v =
                        self.add_optional_rowwise_f32_device(v, &format!("{prefix}.attn_v.bias"))?;
                    (q, k, v, gate)
                };
                cache.write_layer_batched(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    1,
                    cache_position,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.cached_decode_attention_batched_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    batch_count,
                    cache_position,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_batch_logits_paged_device_with_rope_positions(
            &self,
            token_ids: &[u32],
            cache_position: usize,
            rope_positions: &[usize],
            cache: &mut CudaPagedBatchKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("CUDA paged batched MRoPE decode requires at least one token");
            }
            if rope_positions.len() != batch_count {
                bail!(
                    "CUDA paged batched MRoPE decode got {} RoPE positions for {batch_count} tokens",
                    rope_positions.len()
                );
            }
            if cache_position >= dims.context {
                bail!(
                    "decode position {cache_position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let mut position_ids = Vec::with_capacity(batch_count);
            for position in rope_positions {
                if *position > dims.context {
                    bail!(
                        "decode RoPE position {position} exceeds qwen context length {}",
                        dims.context
                    );
                }
                let position = u32::try_from(*position)
                    .context("CUDA paged batched MRoPE decode position does not fit u32")?;
                position_ids.push([position, position, position]);
            }
            let mrope_positions = self.mrope_positions_device(&position_ids, batch_count)?;
            let mrope_head_dim = if self.config.attention_mla_tensor_layout {
                qwen_mla_dims(&self.config)?
                    .map(|mla| mla.qk_rope_head_dim)
                    .unwrap_or(dims.head_dim)
            } else {
                dims.head_dim
            };
            let mrope_sections = self.mrope_sections(mrope_head_dim)?.ok_or_else(|| {
                anyhow!(
                    "CUDA paged batched MRoPE decode requires GGUF rope.dimension_sections metadata"
                )
            })?;

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = if self.layer_uses_mla_attention(&prefix) {
                    self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        QwenRopeLayout::Mrope {
                            positions: &mrope_positions,
                            sections: mrope_sections,
                        },
                    )?
                } else {
                    let (q, gate) =
                        self.dense_attention_q_f32_device(&prefix, &attn_input, &dims, eps)?;
                    self.apply_mrope_f32_device(
                        &q,
                        &mrope_positions,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        mrope_sections,
                        true,
                    )?;

                    let k =
                        self.project_f32_device(&format!("{prefix}.attn_k.weight"), &attn_input)?;
                    let k =
                        self.add_optional_rowwise_f32_device(k, &format!("{prefix}.attn_k.bias"))?;
                    let k = self.optional_head_rms_norm_f32_device(
                        k,
                        &format!("{prefix}.attn_k_norm.weight"),
                        batch_count,
                        dims.kv_heads,
                        dims.head_dim,
                        eps,
                    )?;
                    self.apply_mrope_f32_device(
                        &k,
                        &mrope_positions,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        mrope_sections,
                        true,
                    )?;

                    let v =
                        self.project_f32_device(&format!("{prefix}.attn_v.weight"), &attn_input)?;
                    let v =
                        self.add_optional_rowwise_f32_device(v, &format!("{prefix}.attn_v.bias"))?;
                    (q, k, v, gate)
                };
                cache.write_layer_batched(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    1,
                    cache_position,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.paged_decode_attention_batched_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    batch_count,
                    cache_position,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_batch_logits_paged_device_with_cache_and_mrope_positions(
            &self,
            token_ids: &[u32],
            cache_positions: &[usize],
            rope_positions: &[usize],
            cache: &mut CudaPagedBatchKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("CUDA paged ragged MRoPE decode requires at least one token");
            }
            if cache_positions.len() != batch_count || rope_positions.len() != batch_count {
                bail!(
                    "CUDA paged ragged MRoPE decode got {} cache position(s) and {} RoPE position(s) for {batch_count} token(s)",
                    cache_positions.len(),
                    rope_positions.len()
                );
            }
            let cache_max_seq = cache.layer(0)?.max_seq;
            let mut cache_positions_u32 = Vec::with_capacity(batch_count);
            let mut position_ids = Vec::with_capacity(batch_count);
            for (cache_position, rope_position) in cache_positions.iter().zip(rope_positions) {
                if *cache_position >= dims.context {
                    bail!(
                        "decode position {cache_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
                if *cache_position >= cache_max_seq {
                    bail!(
                        "decode position {cache_position} exceeds paged KV cache sequence capacity {cache_max_seq}"
                    );
                }
                if *rope_position > dims.context {
                    bail!(
                        "decode RoPE position {rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
                cache_positions_u32.push(
                    u32::try_from(*cache_position)
                        .context("CUDA paged ragged MRoPE cache position does not fit u32")?,
                );
                let rope_position = u32::try_from(*rope_position)
                    .context("CUDA paged ragged MRoPE decode position does not fit u32")?;
                position_ids.push([rope_position, rope_position, rope_position]);
            }
            let position_bytes = cache_positions_u32
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .context("CUDA paged ragged MRoPE cache position byte count overflows usize")?;
            let cache_positions_device = DeviceBuffer::alloc(position_bytes)
                .context("allocating CUDA paged ragged MRoPE cache positions")?;
            cache_positions_device
                .copy_from_host(&cache_positions_u32)
                .context("copying CUDA paged ragged MRoPE cache positions")?;
            let mrope_positions = self.mrope_positions_device(&position_ids, batch_count)?;
            let mrope_head_dim = if self.config.attention_mla_tensor_layout {
                qwen_mla_dims(&self.config)?
                    .map(|mla| mla.qk_rope_head_dim)
                    .unwrap_or(dims.head_dim)
            } else {
                dims.head_dim
            };
            let mrope_sections = self.mrope_sections(mrope_head_dim)?.ok_or_else(|| {
                anyhow!(
                    "CUDA paged ragged MRoPE decode requires GGUF rope.dimension_sections metadata"
                )
            })?;

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = if self.layer_uses_mla_attention(&prefix) {
                    self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        QwenRopeLayout::Mrope {
                            positions: &mrope_positions,
                            sections: mrope_sections,
                        },
                    )?
                } else {
                    let (q, gate) =
                        self.dense_attention_q_f32_device(&prefix, &attn_input, &dims, eps)?;
                    self.apply_mrope_f32_device(
                        &q,
                        &mrope_positions,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        mrope_sections,
                        true,
                    )?;

                    let k =
                        self.project_f32_device(&format!("{prefix}.attn_k.weight"), &attn_input)?;
                    let k =
                        self.add_optional_rowwise_f32_device(k, &format!("{prefix}.attn_k.bias"))?;
                    let k = self.optional_head_rms_norm_f32_device(
                        k,
                        &format!("{prefix}.attn_k_norm.weight"),
                        batch_count,
                        dims.kv_heads,
                        dims.head_dim,
                        eps,
                    )?;
                    self.apply_mrope_f32_device(
                        &k,
                        &mrope_positions,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        mrope_sections,
                        true,
                    )?;

                    let v =
                        self.project_f32_device(&format!("{prefix}.attn_v.weight"), &attn_input)?;
                    let v =
                        self.add_optional_rowwise_f32_device(v, &format!("{prefix}.attn_v.bias"))?;
                    (q, k, v, gate)
                };
                cache.write_layer_batched_positions(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    1,
                    &cache_positions_device,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.paged_decode_attention_batched_positions_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    &cache_positions_device,
                    batch_count,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_batch_logits_paged_device(
            &self,
            token_ids: &[u32],
            position: usize,
            cache: &mut CudaPagedBatchKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("CUDA paged batched decode requires at least one token");
            }
            if position >= dims.context {
                bail!(
                    "decode position {position} exceeds qwen context length {}",
                    dims.context
                );
            }
            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::Batched {
                        batch_count,
                        seq_len: 1,
                        position_offset: position,
                    },
                )?;
                cache.write_layer_batched(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    1,
                    position,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.paged_decode_attention_batched_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    batch_count,
                    position,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn decode_batch_logits_paged_device_with_positions(
            &self,
            token_ids: &[u32],
            positions: &[usize],
            cache: &mut CudaPagedBatchKvCache,
        ) -> Result<GpuF32Tensor> {
            let _t = self.forward_timer();
            let dims = self.qwen_dims()?;
            let batch_count = token_ids.len();
            if batch_count == 0 {
                bail!("CUDA paged ragged batched decode requires at least one token");
            }
            if positions.len() != batch_count {
                bail!(
                    "CUDA paged ragged batched decode got {} positions for {batch_count} token(s)",
                    positions.len()
                );
            }
            let cache_max_seq = cache.layer(0)?.max_seq;
            let mut positions_u32 = Vec::with_capacity(batch_count);
            for position in positions {
                if *position >= dims.context {
                    bail!(
                        "decode position {position} exceeds qwen context length {}",
                        dims.context
                    );
                }
                if *position >= cache_max_seq {
                    bail!(
                        "decode position {position} exceeds paged KV cache sequence capacity {cache_max_seq}"
                    );
                }
                positions_u32.push(
                    u32::try_from(*position)
                        .context("CUDA paged ragged decode position does not fit u32")?,
                );
            }
            let position_bytes = positions_u32
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .context("CUDA paged ragged decode position byte count overflows usize")?;
            let positions_device = DeviceBuffer::alloc(position_bytes)
                .context("allocating CUDA paged ragged decode positions")?;
            positions_device
                .copy_from_host(&positions_u32)
                .context("copying CUDA paged ragged decode positions")?;

            let mut hidden = self.embed_tokens_device(token_ids)?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                self.ensure_layer_runtime_supported(&prefix)?;
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                let (q, k, v, gate) = self.attention_qkv_f32_device(
                    &prefix,
                    &attn_input,
                    &dims,
                    eps,
                    rope_base,
                    rope_scale,
                    QwenRopeLayout::BatchedPositions {
                        positions: &positions_device,
                        host_positions: positions,
                        batch_count,
                        seq_len: 1,
                    },
                )?;
                cache.write_layer_batched_positions(
                    layer_idx,
                    &k,
                    &v,
                    batch_count,
                    1,
                    &positions_device,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    &self.stream,
                )?;
                let attn = self.paged_decode_attention_batched_positions_f32_device(
                    &q,
                    cache.layer(layer_idx)?,
                    &positions_device,
                    batch_count,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.v_head_dim,
                    self.layer_attention_window(&prefix),
                )?;
                let attn_out =
                    self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                hidden = self.add_f32_device(&hidden, &attn_out)?;

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn qwen_dims(&self) -> Result<QwenDims> {
            let context = usize::try_from(self.config.context_length)
                .context("qwen context_length does not fit usize")?;
            let embed = usize::try_from(self.config.embedding_length)
                .context("qwen embedding_length does not fit usize")?;
            let heads = usize::try_from(self.config.attention_head_count)
                .context("qwen attention_head_count does not fit usize")?;
            let metadata_kv_heads = usize::try_from(self.config.attention_head_count_kv)
                .context("qwen attention_head_count_kv does not fit usize")?;
            if heads == 0 || metadata_kv_heads == 0 {
                bail!("qwen attention heads and kv heads must be non-zero");
            }
            let kv_heads = if self.config.attention_mla_tensor_layout {
                heads
            } else {
                metadata_kv_heads
            };
            if heads % kv_heads != 0 {
                bail!(
                    "qwen attention heads {heads} must be a non-zero multiple of kv heads {kv_heads}"
                );
            }
            let head_dim = self
                .config
                .attention_key_head_dim()
                .map(usize::try_from)
                .transpose()
                .context("qwen attention key head dimension does not fit usize")?
                .ok_or_else(|| {
                    anyhow!(
                        "qwen attention key length is incompatible with embedding length {embed} and attention heads {heads}"
                    )
                })?;
            let v_head_dim = self
                .config
                .attention_value_head_dim()
                .map(usize::try_from)
                .transpose()
                .context("qwen attention value head dimension does not fit usize")?
                .ok_or_else(|| {
                    anyhow!(
                        "qwen attention value length is incompatible with embedding length {embed} and attention heads {heads}"
                    )
                })?;
            if head_dim == 0 || head_dim % 2 != 0 {
                bail!("qwen CUDA RoPE requires an even non-zero head dimension, got {head_dim}");
            }
            if v_head_dim == 0 {
                bail!("qwen CUDA attention requires a non-zero value head dimension");
            }
            Ok(QwenDims {
                context,
                embed,
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
            })
        }

        fn layer_uses_mla_attention(&self, prefix: &str) -> bool {
            self.config.attention_mla_tensor_layout
                && self.has_matrix(&format!("{prefix}.attn_q_a.weight"))
        }

        fn layer_uses_recurrent_ssm(&self, prefix: &str) -> bool {
            self.config.recurrent_ssm_tensor_layout
                && (self.has_matrix(&format!("{prefix}.ssm_in.weight"))
                    || self.has_matrix(&format!("{prefix}.attn_qkv.weight")))
        }

        fn ensure_layer_runtime_supported(&self, prefix: &str) -> Result<()> {
            if self.layer_uses_recurrent_ssm(prefix) {
                bail!(
                    "CUDA Qwen recurrent SSM KV-cache decode is not implemented yet for {prefix}; full-context generation is supported, but batched/paged decode requires a recurrent-state cache for ssm_conv1d/ssm_dt/ssm_a/ssm_ba/ssm_norm/ssm_out"
                );
            }
            Ok(())
        }

        fn attention_qkv_f32_device(
            &self,
            prefix: &str,
            attn_input: &GpuF32Tensor,
            dims: &QwenDims,
            eps: f32,
            rope_base: f32,
            rope_scale: f32,
            rope: QwenRopeLayout<'_>,
        ) -> Result<(
            GpuF32Tensor,
            GpuF32Tensor,
            GpuF32Tensor,
            Option<GpuF32Tensor>,
        )> {
            self.ensure_layer_runtime_supported(prefix)?;
            if attn_input.rows != rope.rows() {
                bail!(
                    "CUDA attention input rows {} do not match rope layout rows {}",
                    attn_input.rows,
                    rope.rows()
                );
            }
            // Gemma-3 uses a per-layer RoPE base (local layers 10000, global layers
            // rope.freq_base); every other model keeps the single base passed in.
            let rope_base = self.layer_rope_base(prefix, rope_base);
            if !self.layer_uses_mla_attention(prefix) {
                let (q, gate) = self.dense_attention_q_f32_device(prefix, attn_input, dims, eps)?;
                match rope {
                    QwenRopeLayout::Single {
                        seq_len,
                        position_offset,
                    } => self.apply_rope_f32_device(
                        &q,
                        seq_len,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        position_offset,
                        true,
                    )?,
                    QwenRopeLayout::Batched {
                        batch_count,
                        seq_len,
                        position_offset,
                    } => self.apply_rope_batched_f32_device(
                        &q,
                        batch_count,
                        seq_len,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        position_offset,
                        true,
                    )?,
                    QwenRopeLayout::BatchedPositions {
                        positions,
                        batch_count,
                        seq_len,
                        ..
                    } => self.apply_rope_batched_positions_f32_device(
                        &q,
                        positions,
                        batch_count,
                        seq_len,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        true,
                    )?,
                    QwenRopeLayout::Mrope {
                        positions,
                        sections,
                    } => self.apply_mrope_f32_device(
                        &q,
                        positions,
                        dims.heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        sections,
                        true,
                    )?,
                }

                let k = self.project_f32_device(&format!("{prefix}.attn_k.weight"), attn_input)?;
                let k =
                    self.add_optional_rowwise_f32_device(k, &format!("{prefix}.attn_k.bias"))?;
                let k = self.optional_head_rms_norm_f32_device(
                    k,
                    &format!("{prefix}.attn_k_norm.weight"),
                    attn_input.rows,
                    dims.kv_heads,
                    dims.head_dim,
                    eps,
                )?;
                match rope {
                    QwenRopeLayout::Single {
                        seq_len,
                        position_offset,
                    } => self.apply_rope_f32_device(
                        &k,
                        seq_len,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        position_offset,
                        true,
                    )?,
                    QwenRopeLayout::Batched {
                        batch_count,
                        seq_len,
                        position_offset,
                    } => self.apply_rope_batched_f32_device(
                        &k,
                        batch_count,
                        seq_len,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        position_offset,
                        true,
                    )?,
                    QwenRopeLayout::BatchedPositions {
                        positions,
                        batch_count,
                        seq_len,
                        ..
                    } => self.apply_rope_batched_positions_f32_device(
                        &k,
                        positions,
                        batch_count,
                        seq_len,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        true,
                    )?,
                    QwenRopeLayout::Mrope {
                        positions,
                        sections,
                    } => self.apply_mrope_f32_device(
                        &k,
                        positions,
                        dims.kv_heads,
                        dims.head_dim,
                        rope_base,
                        rope_scale,
                        sections,
                        true,
                    )?,
                }

                let v = self.project_f32_device(&format!("{prefix}.attn_v.weight"), attn_input)?;
                let v =
                    self.add_optional_rowwise_f32_device(v, &format!("{prefix}.attn_v.bias"))?;
                return Ok((q, k, v, gate));
            }

            let mla = qwen_mla_dims(&self.config)?
                .ok_or_else(|| anyhow!("MLA tensor layout missing MLA metadata"))?;
            if dims.head_dim != mla.qk_head_dim || dims.v_head_dim != mla.v_head_dim {
                bail!(
                    "CUDA MLA dims disagree with Qwen dims: qk {}/{}, v {}/{}",
                    mla.qk_head_dim,
                    dims.head_dim,
                    mla.v_head_dim,
                    dims.v_head_dim
                );
            }
            if dims.kv_heads != dims.heads {
                bail!("CUDA MLA attention requires expanded kv_heads to equal heads");
            }

            let q_latent =
                self.project_f32_device(&format!("{prefix}.attn_q_a.weight"), attn_input)?;
            let q_latent = self.rms_norm_f32_device(
                &format!("{prefix}.attn_q_a_norm.weight"),
                &q_latent,
                eps,
            )?;
            let q = self.project_f32_device(&format!("{prefix}.attn_q_b.weight"), &q_latent)?;
            let mut q_host = q.copy_to_host()?;
            for row in 0..attn_input.rows {
                for head in 0..dims.heads {
                    let head_start = row
                        .checked_mul(q.cols)
                        .and_then(|value| value.checked_add(head * dims.head_dim))
                        .context("CUDA MLA q host offset overflows usize")?;
                    let rope_start = head_start + mla.qk_nope_head_dim;
                    let rope_end = rope_start + mla.qk_rope_head_dim;
                    apply_rope_layout_host(
                        &mut q_host[rope_start..rope_end],
                        row,
                        rope,
                        rope_base,
                        rope_scale,
                        true,
                    )?;
                }
            }
            let q = self.f32_tensor_from_host(&q_host, q.rows, q.cols, "CUDA MLA q compose")?;

            let kv_a =
                self.project_f32_device(&format!("{prefix}.attn_kv_a_mqa.weight"), attn_input)?;
            let kv_a_host = kv_a.copy_to_host()?;
            let mut kv_latent_host = vec![0.0; attn_input.rows * mla.kv_lora_rank];
            let mut k_pe_host = vec![0.0; attn_input.rows * mla.qk_rope_head_dim];
            for row in 0..attn_input.rows {
                let source_start = row * kv_a.cols;
                let latent_start = row * mla.kv_lora_rank;
                let pe_start = row * mla.qk_rope_head_dim;
                kv_latent_host[latent_start..latent_start + mla.kv_lora_rank]
                    .copy_from_slice(&kv_a_host[source_start..source_start + mla.kv_lora_rank]);
                k_pe_host[pe_start..pe_start + mla.qk_rope_head_dim].copy_from_slice(
                    &kv_a_host[source_start + mla.kv_lora_rank
                        ..source_start + mla.kv_lora_rank + mla.qk_rope_head_dim],
                );
            }
            let kv_latent = self.f32_tensor_from_host(
                &kv_latent_host,
                attn_input.rows,
                mla.kv_lora_rank,
                "CUDA MLA kv latent",
            )?;
            let kv_latent = self.rms_norm_f32_device(
                &format!("{prefix}.attn_kv_a_norm.weight"),
                &kv_latent,
                eps,
            )?;
            let kv_b =
                self.project_f32_device(&format!("{prefix}.attn_kv_b.weight"), &kv_latent)?;
            let kv_b_host = kv_b.copy_to_host()?;
            let mut k_host = vec![0.0; attn_input.rows * dims.heads * dims.head_dim];
            let mut v_host = vec![0.0; attn_input.rows * dims.heads * dims.v_head_dim];
            let kv_b_head_dim = mla
                .qk_nope_head_dim
                .checked_add(dims.v_head_dim)
                .context("CUDA MLA kv_b per-head dimension overflows usize")?;
            for row in 0..attn_input.rows {
                let mut k_pe = k_pe_host
                    [row * mla.qk_rope_head_dim..(row + 1) * mla.qk_rope_head_dim]
                    .to_vec();
                apply_rope_layout_host(&mut k_pe, row, rope, rope_base, rope_scale, true)?;
                for head in 0..dims.heads {
                    let kv_b_start = row * kv_b.cols + head * kv_b_head_dim;
                    let k_start = row * dims.heads * dims.head_dim + head * dims.head_dim;
                    let v_start = row * dims.heads * dims.v_head_dim + head * dims.v_head_dim;
                    k_host[k_start..k_start + mla.qk_nope_head_dim]
                        .copy_from_slice(&kv_b_host[kv_b_start..kv_b_start + mla.qk_nope_head_dim]);
                    k_host[k_start + mla.qk_nope_head_dim
                        ..k_start + mla.qk_nope_head_dim + mla.qk_rope_head_dim]
                        .copy_from_slice(&k_pe);
                    v_host[v_start..v_start + dims.v_head_dim].copy_from_slice(
                        &kv_b_host[kv_b_start + mla.qk_nope_head_dim
                            ..kv_b_start + mla.qk_nope_head_dim + dims.v_head_dim],
                    );
                }
            }
            let k = self.f32_tensor_from_host(
                &k_host,
                attn_input.rows,
                dims.heads * dims.head_dim,
                "CUDA MLA k compose",
            )?;
            let v = self.f32_tensor_from_host(
                &v_host,
                attn_input.rows,
                dims.heads * dims.v_head_dim,
                "CUDA MLA v compose",
            )?;
            Ok((q, k, v, None))
        }

        fn prefill_recurrent_ssm_state(
            &self,
            input: &[u32],
            state: &mut RecurrentSsmRequestState,
            mut cache: Option<&mut CudaPagedBatchKvCache<'_>>,
            label: &str,
        ) -> Result<()> {
            if input.is_empty() {
                bail!("{label} requires non-empty recurrent SSM prompts");
            }
            if state.seq_len != 0 || !state.tokens.is_empty() || !state.layers.is_empty() {
                bail!("{label} recurrent SSM state must be empty before prefill");
            }
            for token_id in input.iter().copied() {
                let _ = self.recurrent_ssm_decode_one_logits_with_state(
                    token_id,
                    state,
                    cache.as_deref_mut(),
                    label,
                )?;
            }
            Ok(())
        }

        fn recurrent_ssm_decode_one_logits_with_state(
            &self,
            token_id: u32,
            state: &mut RecurrentSsmRequestState,
            mut cache: Option<&mut CudaPagedBatchKvCache<'_>>,
            label: &str,
        ) -> Result<GpuF32Tensor> {
            let dims = self.qwen_dims()?;
            if state.seq_len >= dims.context {
                bail!(
                    "{label} recurrent SSM state length {} exceeds qwen context length {}",
                    state.seq_len + 1,
                    dims.context
                );
            }
            let ssm = qwen_ssm_dims(&self.config)?
                .ok_or_else(|| anyhow!("recurrent SSM tensor layout missing SSM metadata"))?;
            let mut hidden = self.embed_tokens_device(&[token_id])?;
            let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
            let rope_base = self
                .config
                .rope_freq_base
                .unwrap_or_else(|| self.config.default_rope_freq_base());
            let rope_scale = self.config.rope_freq_scale.unwrap_or(1.0);

            for layer in 0..self.config.block_count {
                let layer_idx =
                    usize::try_from(layer).context("qwen layer index does not fit usize")?;
                let prefix = format!("blk.{layer}");
                let attn_input =
                    self.rms_norm_f32_device(&format!("{prefix}.attn_norm.weight"), &hidden, eps)?;

                if self.layer_uses_recurrent_ssm(&prefix) {
                    if !state.layers.contains_key(&layer_idx) {
                        state
                            .layers
                            .insert(layer_idx, RecurrentSsmLayerState::new(&ssm, &self.stream)?);
                    }
                    let layer_state = state
                        .layers
                        .get_mut(&layer_idx)
                        .ok_or_else(|| anyhow!("{label} recurrent SSM layer state is missing"))?;
                    let ssm_out =
                        self.recurrent_ssm_step_f32_device(&prefix, &attn_input, eps, layer_state)?;
                    hidden = self.add_f32_device(&hidden, &ssm_out)?;
                } else {
                    self.ensure_layer_runtime_supported(&prefix)?;
                    let cache_position = state.seq_len;
                    let cache = cache.as_deref_mut().ok_or_else(|| {
                        anyhow!(
                            "{label} persistent recurrent SSM decode requires paged KV state for attention layer {prefix}"
                        )
                    })?;
                    let (q, k, v, gate) = self.attention_qkv_f32_device(
                        &prefix,
                        &attn_input,
                        &dims,
                        eps,
                        rope_base,
                        rope_scale,
                        QwenRopeLayout::Single {
                            seq_len: 1,
                            position_offset: cache_position,
                        },
                    )?;
                    cache.write_layer_batched(
                        layer_idx,
                        &k,
                        &v,
                        1,
                        1,
                        cache_position,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        &self.stream,
                    )?;
                    let attn = self.paged_decode_attention_batched_f32_device(
                        &q,
                        cache.layer(layer_idx)?,
                        1,
                        cache_position,
                        dims.heads,
                        dims.kv_heads,
                        dims.head_dim,
                        dims.v_head_dim,
                        self.layer_attention_window(&prefix),
                    )?;
                    let attn_out =
                        self.attention_output_projection_f32_device(&prefix, &attn, gate.as_ref())?;
                    hidden = self.add_f32_device(&hidden, &attn_out)?;
                }

                let mlp_input =
                    self.rms_norm_f32_device(&format!("{prefix}.ffn_norm.weight"), &hidden, eps)?;
                let mlp_out = self.ffn_f32_device(&prefix, &mlp_input)?;
                hidden = self.add_f32_device(&hidden, &mlp_out)?;
            }

            state.tokens.push(token_id);
            state.seq_len = state.seq_len.saturating_add(1);
            let normed = self.rms_norm_f32_device("output_norm.weight", &hidden, eps)?;
            self.output_logits_f32_device(&normed)
        }

        fn recurrent_ssm_step_f32_device(
            &self,
            prefix: &str,
            attn_input: &GpuF32Tensor,
            eps: f32,
            state: &mut RecurrentSsmLayerState,
        ) -> Result<GpuF32Tensor> {
            let dims = self.qwen_dims()?;
            let ssm = qwen_ssm_dims(&self.config)?
                .ok_or_else(|| anyhow!("recurrent SSM tensor layout missing SSM metadata"))?;
            if attn_input.rows != 1 {
                bail!(
                    "CUDA recurrent SSM state update for {prefix} expects one row, got {}",
                    attn_input.rows
                );
            }
            if attn_input.cols != dims.embed {
                bail!(
                    "CUDA recurrent SSM input cols {} do not match embedding dim {} for {prefix}",
                    attn_input.cols,
                    dims.embed
                );
            }

            let (qkv, gate, packed_qkvz) = if self.has_matrix(&format!("{prefix}.ssm_in.weight")) {
                let qkvz =
                    self.project_f32_device(&format!("{prefix}.ssm_in.weight"), attn_input)?;
                if qkvz.cols != ssm.qkvz_dim {
                    bail!(
                        "CUDA recurrent SSM projection {prefix}.ssm_in.weight produced {} columns; expected {}",
                        qkvz.cols,
                        ssm.qkvz_dim
                    );
                }
                (qkvz, None, true)
            } else {
                let qkv =
                    self.project_f32_device(&format!("{prefix}.attn_qkv.weight"), attn_input)?;
                if qkv.cols != ssm.conv_dim {
                    bail!(
                        "CUDA recurrent SSM projection {prefix}.attn_qkv.weight produced {} columns; expected {}",
                        qkv.cols,
                        ssm.conv_dim
                    );
                }
                let gate =
                    self.project_f32_device(&format!("{prefix}.attn_gate.weight"), attn_input)?;
                if gate.cols != ssm.value_dim {
                    bail!(
                        "CUDA recurrent SSM projection {prefix}.attn_gate.weight produced {} columns; expected {}",
                        gate.cols,
                        ssm.value_dim
                    );
                }
                (qkv, Some(gate), false)
            };

            let conv_weight_name = format!("{prefix}.ssm_conv1d.weight");
            let conv_weight_matrix = self
                .matrix(&conv_weight_name)
                .ok_or_else(|| anyhow!("CUDA matrix {conv_weight_name} is missing"))?;
            if conv_weight_matrix.rows != ssm.conv_dim || conv_weight_matrix.cols != ssm.conv_kernel
            {
                bail!(
                    "CUDA recurrent SSM conv weight {conv_weight_name} has shape {}x{}; expected {}x{}",
                    conv_weight_matrix.rows,
                    conv_weight_matrix.cols,
                    ssm.conv_dim,
                    ssm.conv_kernel
                );
            }
            let conv_weight_owned =
                self.matrix_f32_device_owned(&conv_weight_name, conv_weight_matrix)?;
            let conv_weight = conv_weight_owned
                .as_ref()
                .unwrap_or(&conv_weight_matrix.buffer);

            let ba = self.project_f32_device(&format!("{prefix}.ssm_ba.weight"), attn_input)?;
            if ba.cols != ssm.ba_dim {
                bail!(
                    "CUDA recurrent SSM projection {prefix}.ssm_ba.weight produced {} columns; expected {}",
                    ba.cols,
                    ssm.ba_dim
                );
            }
            let dt_bias = self
                .vector(&format!("{prefix}.ssm_dt.bias"))
                .ok_or_else(|| anyhow!("CUDA vector {prefix}.ssm_dt.bias is missing"))?;
            if dt_bias.len != ssm.time_step_rank {
                bail!(
                    "CUDA recurrent SSM {prefix}.ssm_dt.bias has length {}; expected {}",
                    dt_bias.len,
                    ssm.time_step_rank
                );
            }
            let a_log = self
                .vector(&format!("{prefix}.ssm_a"))
                .ok_or_else(|| anyhow!("CUDA vector {prefix}.ssm_a is missing"))?;
            if a_log.len != ssm.time_step_rank {
                bail!(
                    "CUDA recurrent SSM {prefix}.ssm_a has length {}; expected {}",
                    a_log.len,
                    ssm.time_step_rank
                );
            }
            let norm_weight = self
                .vector(&format!("{prefix}.ssm_norm.weight"))
                .ok_or_else(|| anyhow!("CUDA vector {prefix}.ssm_norm.weight is missing"))?;
            if norm_weight.len != ssm.head_v_dim {
                bail!(
                    "CUDA recurrent SSM {prefix}.ssm_norm.weight has length {}; expected {}",
                    norm_weight.len,
                    ssm.head_v_dim
                );
            }
            let expected_conv_elements = ssm
                .conv_kernel
                .checked_mul(ssm.conv_dim)
                .context("CUDA recurrent SSM conv state element count overflows usize")?;
            if state.conv_elements != expected_conv_elements {
                bail!(
                    "CUDA recurrent SSM conv state for {prefix} has {} values; expected {expected_conv_elements}",
                    state.conv_elements
                );
            }
            let expected_recurrent_elements = ssm
                .time_step_rank
                .checked_mul(ssm.state_size)
                .and_then(|value| value.checked_mul(ssm.head_v_dim))
                .context("CUDA recurrent SSM matrix state element count overflows usize")?;
            if state.recurrent_elements != expected_recurrent_elements {
                bail!(
                    "CUDA recurrent SSM matrix state for {prefix} has {} values; expected {expected_recurrent_elements}",
                    state.recurrent_elements
                );
            }
            if state.conv_next >= ssm.conv_kernel {
                bail!(
                    "CUDA recurrent SSM streaming conv write cursor {} exceeds kernel {}",
                    state.conv_next,
                    ssm.conv_kernel
                );
            }
            if state.conv_len > ssm.conv_kernel {
                bail!(
                    "CUDA recurrent SSM streaming conv length {} exceeds kernel {}",
                    state.conv_len,
                    ssm.conv_kernel
                );
            }

            let scratch_elements = ssm
                .conv_dim
                .checked_add(
                    ssm.key_dim
                        .checked_mul(2)
                        .context("CUDA recurrent SSM scratch key dimension overflows usize")?,
                )
                .context("CUDA recurrent SSM scratch element count overflows usize")?;
            let scratch = DeviceBuffer::alloc(scratch_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA recurrent SSM step scratch")?;
            let normed_buffer = DeviceBuffer::alloc(ssm.value_dim * std::mem::size_of::<f32>())
                .context("allocating CUDA recurrent SSM state gated RMSNorm")?;
            crate::kernels::launch_qwen_ssm_streaming_step(
                &qkv.buffer,
                gate.as_ref().map(|tensor| &tensor.buffer),
                conv_weight,
                &ba.buffer,
                &dt_bias.buffer,
                &a_log.buffer,
                &norm_weight.buffer,
                &state.conv_ring,
                &state.recurrent,
                &scratch,
                &normed_buffer,
                state.conv_next,
                state.conv_len,
                ssm.conv_kernel,
                ssm.conv_dim,
                ssm.state_size,
                ssm.time_step_rank,
                ssm.group_count,
                ssm.head_v_dim,
                packed_qkvz,
                eps,
                &self.stream,
            )?;
            self.op_barrier()?;
            state.conv_len = state.conv_len.saturating_add(1).min(ssm.conv_kernel);
            state.conv_next = (state.conv_next + 1) % ssm.conv_kernel;
            let normed = GpuF32Tensor {
                rows: 1,
                cols: ssm.value_dim,
                buffer: normed_buffer,
            };
            self.project_f32_device(&format!("{prefix}.ssm_out.weight"), &normed)
        }

        fn recurrent_ssm_f32_device(
            &self,
            prefix: &str,
            attn_input: &GpuF32Tensor,
            eps: f32,
        ) -> Result<GpuF32Tensor> {
            let dims = self.qwen_dims()?;
            let ssm = qwen_ssm_dims(&self.config)?
                .ok_or_else(|| anyhow!("recurrent SSM tensor layout missing SSM metadata"))?;
            if attn_input.rows == 0 {
                bail!("CUDA recurrent SSM requires at least one row");
            }
            if attn_input.cols != dims.embed {
                bail!(
                    "CUDA recurrent SSM input cols {} do not match embedding dim {} for {prefix}",
                    attn_input.cols,
                    dims.embed
                );
            }

            let (mixed_qkv_host, z_host) = if self.has_matrix(&format!("{prefix}.ssm_in.weight")) {
                let qkvz =
                    self.project_f32_device(&format!("{prefix}.ssm_in.weight"), attn_input)?;
                if qkvz.cols != ssm.qkvz_dim {
                    bail!(
                        "CUDA recurrent SSM projection {prefix}.ssm_in.weight produced {} columns; expected {}",
                        qkvz.cols,
                        ssm.qkvz_dim
                    );
                }
                let qkvz_host = qkvz.copy_to_host()?;
                split_qwen_ssm_qkvz_host(&qkvz_host, attn_input.rows, &ssm)?
            } else {
                let qkv =
                    self.project_f32_device(&format!("{prefix}.attn_qkv.weight"), attn_input)?;
                if qkv.cols != ssm.conv_dim {
                    bail!(
                        "CUDA recurrent SSM projection {prefix}.attn_qkv.weight produced {} columns; expected {}",
                        qkv.cols,
                        ssm.conv_dim
                    );
                }
                let gate =
                    self.project_f32_device(&format!("{prefix}.attn_gate.weight"), attn_input)?;
                if gate.cols != ssm.value_dim {
                    bail!(
                        "CUDA recurrent SSM projection {prefix}.attn_gate.weight produced {} columns; expected {}",
                        gate.cols,
                        ssm.value_dim
                    );
                }
                (qkv.copy_to_host()?, gate.copy_to_host()?)
            };

            let conv_weight_name = format!("{prefix}.ssm_conv1d.weight");
            let conv_weight = self.matrix_f32_host(&conv_weight_name)?;
            let conv = qwen_ssm_depthwise_conv_host(
                &mixed_qkv_host,
                &conv_weight,
                attn_input.rows,
                &ssm,
                &conv_weight_name,
            )?;

            let ba = self.project_f32_device(&format!("{prefix}.ssm_ba.weight"), attn_input)?;
            if ba.cols != ssm.ba_dim {
                bail!(
                    "CUDA recurrent SSM projection {prefix}.ssm_ba.weight produced {} columns; expected {}",
                    ba.cols,
                    ssm.ba_dim
                );
            }
            let ba_host = ba.copy_to_host()?;
            let dt_bias = self
                .vector(&format!("{prefix}.ssm_dt.bias"))
                .ok_or_else(|| anyhow!("CUDA vector {prefix}.ssm_dt.bias is missing"))?
                .copy_to_host_f32()?;
            let a_log = self
                .vector(&format!("{prefix}.ssm_a"))
                .ok_or_else(|| anyhow!("CUDA vector {prefix}.ssm_a is missing"))?
                .copy_to_host_f32()?;
            let norm_weight = self
                .vector(&format!("{prefix}.ssm_norm.weight"))
                .ok_or_else(|| anyhow!("CUDA vector {prefix}.ssm_norm.weight is missing"))?
                .copy_to_host_f32()?;

            let core = qwen_ssm_gated_delta_host(
                &conv,
                &ba_host,
                &dt_bias,
                &a_log,
                attn_input.rows,
                &ssm,
                prefix,
            )?;
            let normed = qwen_ssm_gated_rms_norm_host(
                &core,
                &z_host,
                &norm_weight,
                attn_input.rows,
                &ssm,
                eps,
                prefix,
            )?;
            let normed = self.f32_tensor_from_host(
                &normed,
                attn_input.rows,
                ssm.value_dim,
                "CUDA recurrent SSM gated RMSNorm",
            )?;
            self.project_f32_device(&format!("{prefix}.ssm_out.weight"), &normed)
        }

        fn dense_attention_q_f32_device(
            &self,
            prefix: &str,
            attn_input: &GpuF32Tensor,
            dims: &QwenDims,
            eps: f32,
        ) -> Result<(GpuF32Tensor, Option<GpuF32Tensor>)> {
            let (q, gate) = if self.has_matrix(&format!("{prefix}.attn_q_gated.weight")) {
                let q_gate =
                    self.project_f32_device(&format!("{prefix}.attn_q_gated.weight"), attn_input)?;
                let q_gate = self.add_optional_rowwise_f32_device(
                    q_gate,
                    &format!("{prefix}.attn_q_gated.bias"),
                )?;
                let (q, gate) =
                    self.split_gated_q_projection_f32_device(&q_gate, dims.heads, dims.head_dim)?;
                (q, Some(gate))
            } else {
                let q = self.project_f32_device(&format!("{prefix}.attn_q.weight"), attn_input)?;
                let q =
                    self.add_optional_rowwise_f32_device(q, &format!("{prefix}.attn_q.bias"))?;
                (q, None)
            };
            let q = self.optional_head_rms_norm_f32_device(
                q,
                &format!("{prefix}.attn_q_norm.weight"),
                attn_input.rows,
                dims.heads,
                dims.head_dim,
                eps,
            )?;
            Ok((q, gate))
        }

        fn split_gated_q_projection_f32_device(
            &self,
            projected: &GpuF32Tensor,
            heads: usize,
            head_dim: usize,
        ) -> Result<(GpuF32Tensor, GpuF32Tensor)> {
            let q_cols = heads
                .checked_mul(head_dim)
                .context("CUDA gated q columns overflow usize")?;
            let expected_cols = q_cols
                .checked_mul(2)
                .context("CUDA gated q projection columns overflow usize")?;
            if projected.cols != expected_cols {
                bail!(
                    "CUDA gated q projection shape {}x{} does not match expected columns {expected_cols}",
                    projected.rows,
                    projected.cols
                );
            }
            let projected_host = projected.copy_to_host()?;
            let mut q = vec![0.0; projected.rows * q_cols];
            let mut gate = vec![0.0; projected.rows * q_cols];
            for row in 0..projected.rows {
                let source_row = row
                    .checked_mul(projected.cols)
                    .context("CUDA gated q source row offset overflows usize")?;
                let dest_row = row
                    .checked_mul(q_cols)
                    .context("CUDA gated q destination row offset overflows usize")?;
                for head in 0..heads {
                    let source = source_row
                        .checked_add(head * head_dim * 2)
                        .context("CUDA gated q source offset overflows usize")?;
                    let dest = dest_row
                        .checked_add(head * head_dim)
                        .context("CUDA gated q destination offset overflows usize")?;
                    q[dest..dest + head_dim]
                        .copy_from_slice(&projected_host[source..source + head_dim]);
                    gate[dest..dest + head_dim]
                        .copy_from_slice(&projected_host[source + head_dim..source + 2 * head_dim]);
                }
            }
            Ok((
                self.f32_tensor_from_host(&q, projected.rows, q_cols, "CUDA gated q split")?,
                self.f32_tensor_from_host(
                    &gate,
                    projected.rows,
                    q_cols,
                    "CUDA gated attention gate split",
                )?,
            ))
        }

        fn attention_output_projection_f32_device(
            &self,
            prefix: &str,
            attn: &GpuF32Tensor,
            gate: Option<&GpuF32Tensor>,
        ) -> Result<GpuF32Tensor> {
            let projected = if let Some(gate) = gate {
                gate.ensure_same_shape(attn, "CUDA gated attention output")?;
                let gate_host = gate.copy_to_host()?;
                let mut attn_host = attn.copy_to_host()?;
                for (value, gate) in attn_host.iter_mut().zip(gate_host) {
                    *value *= sigmoid(gate);
                }
                let gated = self.f32_tensor_from_host(
                    &attn_host,
                    attn.rows,
                    attn.cols,
                    "CUDA gated attention output",
                )?;
                self.project_f32_device(&format!("{prefix}.attn_output.weight"), &gated)?
            } else {
                self.project_f32_device(&format!("{prefix}.attn_output.weight"), attn)?
            };
            let output = self.add_optional_rowwise_f32_device(
                projected,
                &format!("{prefix}.attn_output.bias"),
            )?;
            self.apply_gemma_post_attn_norm(prefix, output)
        }

        fn output_logits_f32_device(&self, normed: &GpuF32Tensor) -> Result<GpuF32Tensor> {
            let head = if self.has_matrix("output.weight") {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            let logits = self.project_f32_device(head, normed)?;
            let logits = self.add_optional_rowwise_f32_device(logits, "output.bias")?;
            self.apply_final_logit_softcap(logits)
        }

        /// Project only the final position of each sequence through the lm_head,
        /// returning `[batch_count, vocab]` logits. Prefill only needs the last
        /// position to pick the first generated token, so projecting all
        /// `batch_count * seq_len` rows wastes a huge output buffer (vocab-wide
        /// per prompt token — e.g. ~4 GB for a 6.7k-token prompt) and can OOM.
        /// This gathers the `seq_len-1` row of each sequence first, then
        /// projects `batch_count` rows.
        fn output_logits_last_row_f32_device(
            &self,
            normed: &GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
        ) -> Result<GpuF32Tensor> {
            if batch_count == 0 || seq_len == 0 {
                bail!("CUDA last-row logits require a non-empty batch and sequence");
            }
            let expected_rows = batch_count
                .checked_mul(seq_len)
                .context("CUDA last-row logits row count overflows usize")?;
            if normed.rows != expected_rows {
                bail!(
                    "CUDA last-row logits got {} rows; expected batch {batch_count} x seq {seq_len}",
                    normed.rows
                );
            }
            // Fast path: a single-sequence prompt's last row is already the tail
            // of `normed`; still gather so the projection sees exactly one row.
            let row_ids = (0..batch_count)
                .map(|batch| {
                    let last = batch
                        .checked_mul(seq_len)
                        .and_then(|base| base.checked_add(seq_len - 1))
                        .context("CUDA last-row index overflows usize")?;
                    u32::try_from(last).context("CUDA last-row index does not fit u32")
                })
                .collect::<Result<Vec<u32>>>()?;
            let cols = normed.cols;
            let ids = DeviceBuffer::alloc(std::mem::size_of_val(row_ids.as_slice()))
                .context("allocating CUDA last-row gather ids")?;
            ids.copy_from_host(&row_ids)
                .context("copying CUDA last-row gather ids")?;
            let output_elements = batch_count
                .checked_mul(cols)
                .context("CUDA last-row gather output overflows usize")?;
            let last = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA last-row gather output")?;
            // launch_gather_rows_f32_to_f32(matrix, ids, output, row_count, cols,
            // matrix_rows): row_count is the number of *output* rows (one per
            // sequence) and matrix_rows bounds the source.
            crate::kernels::launch_gather_rows_f32_to_f32(
                &normed.buffer,
                &ids,
                &last,
                batch_count,
                cols,
                normed.rows,
                &self.stream,
            )?;
            self.op_barrier()?;
            let last_rows = GpuF32Tensor {
                rows: batch_count,
                cols,
                buffer: last,
            };
            self.output_logits_f32_device(&last_rows)
        }

        fn add_f32_device(
            &self,
            left: &GpuF32Tensor,
            right: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            left.ensure_same_shape(right, "CUDA residual add")?;
            let elements = left.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA add output")?;
            crate::kernels::launch_add(
                &left.buffer,
                &right.buffer,
                &output,
                elements,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: left.rows,
                cols: left.cols,
                buffer: output,
            })
        }

        fn add_optional_rowwise_f32_device(
            &self,
            input: GpuF32Tensor,
            vector_name: &str,
        ) -> Result<GpuF32Tensor> {
            let Some(bias) = self.vector(vector_name) else {
                return Ok(input);
            };
            if input.cols != bias.len {
                bail!(
                    "CUDA bias vector {vector_name} has length {}; expected {}",
                    bias.len,
                    input.cols
                );
            }
            let elements = input.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .with_context(|| format!("allocating CUDA rowwise add output for {vector_name}"))?;
            crate::kernels::launch_add_rowwise(
                &input.buffer,
                &bias.buffer,
                &output,
                input.rows,
                input.cols,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn ffn_f32_device(&self, prefix: &str, input: &GpuF32Tensor) -> Result<GpuF32Tensor> {
            let output = if self.has_matrix(&format!("{prefix}.ffn_gate_inp.weight")) {
                self.moe_f32_device(prefix, input)?
            } else {
                let gate = self.project_f32_device(&format!("{prefix}.ffn_gate.weight"), input)?;
                let gate =
                    self.add_optional_rowwise_f32_device(gate, &format!("{prefix}.ffn_gate.bias"))?;
                let up = self.project_f32_device(&format!("{prefix}.ffn_up.weight"), input)?;
                let up =
                    self.add_optional_rowwise_f32_device(up, &format!("{prefix}.ffn_up.bias"))?;
                // Gemma uses GeGLU (gelu(gate) * up); other dense models use SwiGLU.
                let activated = if self.config.is_gemma() {
                    self.gelu_mul_f32_device(&gate, &up)?
                } else {
                    self.silu_mul_f32_device(&gate, &up)?
                };
                let down =
                    self.project_f32_device(&format!("{prefix}.ffn_down.weight"), &activated)?;
                self.add_optional_rowwise_f32_device(down, &format!("{prefix}.ffn_down.bias"))?
            };
            self.apply_gemma_post_ffn_norm(prefix, output)
        }

        fn moe_f32_device(&self, prefix: &str, input: &GpuF32Tensor) -> Result<GpuF32Tensor> {
            let router =
                self.project_f32_device(&format!("{prefix}.ffn_gate_inp.weight"), input)?;
            let router = self
                .add_optional_rowwise_f32_device(router, &format!("{prefix}.ffn_gate_inp.bias"))?;
            let experts = router.cols;
            let top_k = self
                .config
                .expert_used_count
                .map(usize::try_from)
                .transpose()
                .context("qwen expert_used_count does not fit usize")?
                .ok_or_else(|| anyhow!("qwen MoE metadata missing expert_used_count"))?;
            if experts == 0 || top_k == 0 || top_k > experts {
                bail!(
                    "qwen MoE expert counts are invalid: expert_count={experts}, expert_used_count={top_k}"
                );
            }
            let routes = self.moe_routes_device(&router, top_k, self.config.expert_weights_norm)?;

            let output = DeviceBuffer::alloc(input.element_count()? * std::mem::size_of::<f32>())
                .context("allocating CUDA MoE output")?;
            output.memset_zero_async(&self.stream)?;
            for row in 0..input.rows {
                let token = self.copy_row_f32_device(input, row)?;
                for (expert, score) in routes[row].iter().copied() {
                    let gate = self.project_f32_device(
                        &format!("{prefix}.ffn_gate_exps.{expert}.weight"),
                        &token,
                    )?;
                    let gate = self.add_optional_rowwise_f32_device(
                        gate,
                        &format!("{prefix}.ffn_gate_exps.{expert}.bias"),
                    )?;
                    let up = self.project_f32_device(
                        &format!("{prefix}.ffn_up_exps.{expert}.weight"),
                        &token,
                    )?;
                    let up = self.add_optional_rowwise_f32_device(
                        up,
                        &format!("{prefix}.ffn_up_exps.{expert}.bias"),
                    )?;
                    let activated = self.silu_mul_f32_device(&gate, &up)?;
                    let down = self.project_f32_device(
                        &format!("{prefix}.ffn_down_exps.{expert}.weight"),
                        &activated,
                    )?;
                    let down = self.add_optional_rowwise_f32_device(
                        down,
                        &format!("{prefix}.ffn_down_exps.{expert}.bias"),
                    )?;
                    self.add_scaled_row_in_place_device(
                        &output, &down, row, input.rows, input.cols, score,
                    )?;
                }
                if self.has_matrix(&format!("{prefix}.ffn_gate_shexp.weight")) {
                    let shared_scale = if self
                        .has_matrix(&format!("{prefix}.ffn_gate_inp_shexp.weight"))
                    {
                        let shared_gate = self.project_f32_device(
                            &format!("{prefix}.ffn_gate_inp_shexp.weight"),
                            &token,
                        )?;
                        let shared_gate = self.add_optional_rowwise_f32_device(
                            shared_gate,
                            &format!("{prefix}.ffn_gate_inp_shexp.bias"),
                        )?;
                        if shared_gate.rows != 1 || shared_gate.cols != 1 {
                            bail!(
                                "CUDA MoE shared expert gate shape {}x{} does not match expected scalar",
                                shared_gate.rows,
                                shared_gate.cols
                            );
                        }
                        sigmoid(shared_gate.copy_to_host()?[0])
                    } else {
                        1.0
                    };
                    let gate = self
                        .project_f32_device(&format!("{prefix}.ffn_gate_shexp.weight"), &token)?;
                    let gate = self.add_optional_rowwise_f32_device(
                        gate,
                        &format!("{prefix}.ffn_gate_shexp.bias"),
                    )?;
                    let up =
                        self.project_f32_device(&format!("{prefix}.ffn_up_shexp.weight"), &token)?;
                    let up = self.add_optional_rowwise_f32_device(
                        up,
                        &format!("{prefix}.ffn_up_shexp.bias"),
                    )?;
                    let activated = self.silu_mul_f32_device(&gate, &up)?;
                    let down = self.project_f32_device(
                        &format!("{prefix}.ffn_down_shexp.weight"),
                        &activated,
                    )?;
                    let down = self.add_optional_rowwise_f32_device(
                        down,
                        &format!("{prefix}.ffn_down_shexp.bias"),
                    )?;
                    self.add_scaled_row_in_place_device(
                        &output,
                        &down,
                        row,
                        input.rows,
                        input.cols,
                        shared_scale,
                    )?;
                }
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn moe_routes_device(
            &self,
            router: &GpuF32Tensor,
            top_k: usize,
            norm_topk: bool,
        ) -> Result<Vec<Vec<(usize, f32)>>> {
            if router.rows == 0 || router.cols == 0 || top_k == 0 || top_k > router.cols {
                bail!(
                    "CUDA MoE route selection got invalid shape {}x{} and top_k {top_k}",
                    router.rows,
                    router.cols
                );
            }
            let route_count = router
                .rows
                .checked_mul(top_k)
                .context("CUDA MoE route count overflows usize")?;
            let ids = DeviceBuffer::alloc(route_count * std::mem::size_of::<u32>())
                .context("allocating CUDA MoE route ids")?;
            let weights = DeviceBuffer::alloc(route_count * std::mem::size_of::<f32>())
                .context("allocating CUDA MoE route weights")?;
            crate::kernels::launch_moe_topk_router(
                &router.buffer,
                &ids,
                &weights,
                router.rows,
                router.cols,
                top_k,
                norm_topk,
                &self.stream,
            )?;
            self.op_barrier()?;
            let ids = ids.copy_to_host::<u32>(route_count)?;
            let weights = weights.copy_to_host::<f32>(route_count)?;
            let mut routes = Vec::with_capacity(router.rows);
            for row in 0..router.rows {
                let mut row_routes = Vec::with_capacity(top_k);
                for rank in 0..top_k {
                    let idx = row * top_k + rank;
                    let expert = usize::try_from(ids[idx])
                        .context("CUDA MoE route expert id does not fit usize")?;
                    if expert >= router.cols {
                        bail!(
                            "CUDA MoE route expert {expert} is outside expert count {}",
                            router.cols
                        );
                    }
                    row_routes.push((expert, weights[idx]));
                }
                routes.push(row_routes);
            }
            Ok(routes)
        }

        fn optional_head_rms_norm_f32_device(
            &self,
            input: GpuF32Tensor,
            vector_name: &str,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            eps: f32,
        ) -> Result<GpuF32Tensor> {
            let Some(weight) = self.vector(vector_name) else {
                return Ok(input);
            };
            if input.rows != seq_len || input.cols != heads * head_dim {
                bail!(
                    "CUDA head RMSNorm input shape {}x{} does not match expected {}x{} for {vector_name}",
                    input.rows,
                    input.cols,
                    seq_len,
                    heads * head_dim
                );
            }
            if weight.len != head_dim {
                bail!(
                    "CUDA head RMSNorm vector {vector_name} has length {}; expected {head_dim}",
                    weight.len
                );
            }
            let output = DeviceBuffer::alloc(input.element_count()? * std::mem::size_of::<f32>())
                .with_context(|| {
                format!("allocating CUDA head RMSNorm output for {vector_name}")
            })?;
            crate::kernels::launch_rms_norm(
                &input.buffer,
                &weight.buffer,
                &output,
                seq_len * heads,
                head_dim,
                eps,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: input.rows,
                cols: input.cols,
                buffer: output,
            })
        }

        fn apply_rope_f32_device(
            &self,
            input: &GpuF32Tensor,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            base: f32,
            scale: f32,
            position_offset: usize,
            split_half: bool,
        ) -> Result<()> {
            if input.rows != seq_len || input.cols != heads * head_dim {
                bail!(
                    "CUDA RoPE input shape {}x{} does not match expected {}x{}",
                    input.rows,
                    input.cols,
                    seq_len,
                    heads * head_dim
                );
            }
            crate::kernels::launch_rope_with_offset(
                &input.buffer,
                seq_len,
                heads,
                head_dim,
                base,
                scale,
                position_offset,
                split_half,
                &self.stream,
            )?;
            self.op_barrier()
        }

        fn apply_mrope_f32_device(
            &self,
            input: &GpuF32Tensor,
            positions: &MropePositionsDevice,
            heads: usize,
            head_dim: usize,
            base: f32,
            scale: f32,
            sections: [usize; 4],
            split_half: bool,
        ) -> Result<()> {
            if input.rows != positions.rows || input.cols != heads * head_dim {
                bail!(
                    "CUDA MRoPE input shape {}x{} does not match expected {}x{}",
                    input.rows,
                    input.cols,
                    positions.rows,
                    heads * head_dim
                );
            }
            crate::kernels::launch_mrope(
                &input.buffer,
                &positions.t,
                &positions.h,
                &positions.w,
                positions.rows,
                heads,
                head_dim,
                base,
                scale,
                sections,
                split_half,
                &self.stream,
            )?;
            self.op_barrier()
        }

        fn apply_rope_batched_f32_device(
            &self,
            input: &GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            base: f32,
            scale: f32,
            position_offset: usize,
            split_half: bool,
        ) -> Result<()> {
            if input.rows != batch_count * seq_len || input.cols != heads * head_dim {
                bail!(
                    "CUDA batched RoPE input shape {}x{} does not match expected {}x{}",
                    input.rows,
                    input.cols,
                    batch_count * seq_len,
                    heads * head_dim
                );
            }
            crate::kernels::launch_rope_batched_with_offset(
                &input.buffer,
                batch_count,
                seq_len,
                heads,
                head_dim,
                base,
                scale,
                position_offset,
                split_half,
                &self.stream,
            )?;
            self.op_barrier()
        }

        #[allow(clippy::too_many_arguments)]
        fn apply_rope_batched_positions_f32_device(
            &self,
            input: &GpuF32Tensor,
            positions: &DeviceBuffer,
            batch_count: usize,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            base: f32,
            scale: f32,
            split_half: bool,
        ) -> Result<()> {
            if input.rows != batch_count * seq_len || input.cols != heads * head_dim {
                bail!(
                    "CUDA batched positioned RoPE input shape {}x{} does not match expected {}x{}",
                    input.rows,
                    input.cols,
                    batch_count * seq_len,
                    heads * head_dim
                );
            }
            crate::kernels::launch_rope_batched_positions(
                &input.buffer,
                positions,
                batch_count,
                seq_len,
                heads,
                head_dim,
                base,
                scale,
                split_half,
                &self.stream,
            )?;
            self.op_barrier()
        }

        fn causal_attention_f32_device(
            &self,
            q: &GpuF32Tensor,
            k: &GpuF32Tensor,
            v: &GpuF32Tensor,
            seq_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            window: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != seq_len || q.cols != heads * head_dim {
                bail!("CUDA attention q shape {}x{} is invalid", q.rows, q.cols);
            }
            if k.rows != seq_len || k.cols != kv_heads * head_dim {
                bail!("CUDA attention k shape {}x{} is invalid", k.rows, k.cols);
            }
            if v.rows != seq_len || v.cols != kv_heads * v_head_dim {
                bail!("CUDA attention v shape {}x{} is invalid", v.rows, v.cols);
            }
            let output_elements = seq_len
                .checked_mul(heads)
                .and_then(|value| value.checked_mul(v_head_dim))
                .context("CUDA attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA attention output")?;
            if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_tiled_causal_attention(
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &output,
                    seq_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    window,
                    &self.stream,
                )?;
            } else {
                // Only the tiled kernel implements the sliding window; the wide
                // head-dim fallback is never used by windowed (Gemma-3) models.
                if window > 0 {
                    bail!(
                        "CUDA sliding-window attention requires head_dim <= {FLASH_ONLINE_MAX_HEAD_DIM}, got {head_dim}"
                    );
                }
                crate::kernels::launch_causal_attention(
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &output,
                    seq_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: seq_len,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn causal_attention_batched_f32_device(
            &self,
            q: &GpuF32Tensor,
            k: &GpuF32Tensor,
            v: &GpuF32Tensor,
            batch_count: usize,
            seq_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,

            window: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != batch_count * seq_len || q.cols != heads * head_dim {
                bail!(
                    "CUDA batched attention q shape {}x{} is invalid",
                    q.rows,
                    q.cols
                );
            }
            if k.rows != batch_count * seq_len || k.cols != kv_heads * head_dim {
                bail!(
                    "CUDA batched attention k shape {}x{} is invalid",
                    k.rows,
                    k.cols
                );
            }
            if v.rows != batch_count * seq_len || v.cols != kv_heads * v_head_dim {
                bail!(
                    "CUDA batched attention v shape {}x{} is invalid",
                    v.rows,
                    v.cols
                );
            }
            let output_elements = batch_count
                .checked_mul(seq_len)
                .and_then(|value| value.checked_mul(heads))
                .and_then(|value| value.checked_mul(v_head_dim))
                .context("CUDA batched attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA batched attention output")?;
            if window == 0
                && head_dim <= FLASH_TILE_MAX_HEAD_DIM
                && v_head_dim <= FLASH_TILE_MAX_HEAD_DIM
            {
                // Tensor-core (WMMA) prefill attention defaults on for quantized
                // models (weights kept quantized == large models, where prefill is
                // slow and the f16 attention is a fine trade); f16-stored small models
                // stay on the f32 flash kernel unless HI_CUDA_WMMA_ATTN opts them in.
                let use_wmma = (self.info.quantized_matrix_count > 0 || wmma_attn_forced_on())
                    && !wmma_attn_forced_off()
                    && head_dim % 16 == 0
                    && v_head_dim == head_dim;
                if use_wmma {
                    // Tensor-core prefill attention: cast q/k/v to f16 and run WMMA.
                    let q16 = DeviceBuffer::alloc(q.rows * q.cols * std::mem::size_of::<u16>())
                        .context("alloc wmma q16")?;
                    let k16 = DeviceBuffer::alloc(k.rows * k.cols * std::mem::size_of::<u16>())
                        .context("alloc wmma k16")?;
                    let v16 = DeviceBuffer::alloc(v.rows * v.cols * std::mem::size_of::<u16>())
                        .context("alloc wmma v16")?;
                    crate::kernels::launch_cast_f32_to_f16(
                        &q.buffer,
                        &q16,
                        q.rows * q.cols,
                        &self.stream,
                    )?;
                    crate::kernels::launch_cast_f32_to_f16(
                        &k.buffer,
                        &k16,
                        k.rows * k.cols,
                        &self.stream,
                    )?;
                    crate::kernels::launch_cast_f32_to_f16(
                        &v.buffer,
                        &v16,
                        v.rows * v.cols,
                        &self.stream,
                    )?;
                    crate::kernels::launch_wmma_causal_attention_batched(
                        &q16,
                        &k16,
                        &v16,
                        &output,
                        batch_count,
                        seq_len,
                        heads,
                        kv_heads,
                        head_dim,
                        &self.stream,
                    )?;
                } else {
                    crate::kernels::launch_flashtile_causal_attention_batched(
                        &q.buffer,
                        &k.buffer,
                        &v.buffer,
                        &output,
                        batch_count,
                        seq_len,
                        heads,
                        kv_heads,
                        head_dim,
                        v_head_dim,
                        &self.stream,
                    )?;
                }
            } else if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_tiled_causal_attention_batched(
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &output,
                    batch_count,
                    seq_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    window,
                    &self.stream,
                )?;
            } else {
                if window > 0 {
                    bail!(
                        "CUDA sliding-window attention requires the tiled kernel (head_dim <= {FLASH_ONLINE_MAX_HEAD_DIM})"
                    );
                }
                crate::kernels::launch_causal_attention_batched(
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &output,
                    batch_count,
                    seq_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: batch_count * seq_len,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn cached_decode_attention_f32_device(
            &self,
            q: &GpuF32Tensor,
            cache: &CudaLayerKvCache,
            position: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != 1 || q.cols != heads * head_dim {
                bail!(
                    "CUDA cached attention q shape {}x{} is invalid",
                    q.rows,
                    q.cols
                );
            }
            let output_elements = heads
                .checked_mul(v_head_dim)
                .context("CUDA cached attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA cached attention output")?;
            if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_flash_cached_decode_attention(
                    &q.buffer,
                    &cache.key,
                    &cache.value,
                    &output,
                    position,
                    cache.max_seq,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            } else {
                crate::kernels::launch_cached_decode_attention(
                    &q.buffer,
                    &cache.key,
                    &cache.value,
                    &output,
                    position,
                    cache.max_seq,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: 1,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn paged_decode_attention_f32_device(
            &self,
            q: &GpuF32Tensor,
            cache: &CudaPagedLayerKvCache,
            position: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            window: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != 1 || q.cols != heads * head_dim {
                bail!(
                    "CUDA paged attention q shape {}x{} is invalid",
                    q.rows,
                    q.cols
                );
            }
            let output_elements = heads
                .checked_mul(v_head_dim)
                .context("CUDA paged attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA paged attention output")?;
            if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_tiled_paged_decode_attention(
                    &q.buffer,
                    &cache.key_pages,
                    &cache.value_pages,
                    &cache.page_table,
                    &output,
                    position,
                    cache.page_size,
                    cache.page_table_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    window,
                    &self.stream,
                )?;
            } else {
                // Only the tiled kernel implements the sliding window; the wide
                // head-dim fallback is never used by windowed (Gemma-3) models.
                if window > 0 {
                    bail!(
                        "CUDA sliding-window decode requires head_dim <= {FLASH_ONLINE_MAX_HEAD_DIM}, got {head_dim}"
                    );
                }
                crate::kernels::launch_paged_decode_attention(
                    &q.buffer,
                    &cache.key_pages,
                    &cache.value_pages,
                    &cache.page_table,
                    &output,
                    position,
                    cache.page_size,
                    cache.page_table_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: 1,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn paged_decode_attention_batched_f32_device(
            &self,
            q: &GpuF32Tensor,
            cache: &CudaPagedBatchLayerKvCache,
            batch_count: usize,
            position: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,

            window: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != batch_count || q.cols != heads * head_dim {
                bail!(
                    "CUDA paged batched attention q shape {}x{} is invalid",
                    q.rows,
                    q.cols
                );
            }
            if cache.batch_count != batch_count {
                bail!(
                    "CUDA paged batched attention cache batch {} does not match q batch {batch_count}",
                    cache.batch_count
                );
            }
            let output_elements = batch_count
                .checked_mul(heads)
                .and_then(|value| value.checked_mul(v_head_dim))
                .context("CUDA paged batched attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA paged batched attention output")?;
            if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_tiled_paged_decode_attention_batched(
                    &q.buffer,
                    cache.key_pages.as_buffer(),
                    cache.value_pages.as_buffer(),
                    &cache.page_table,
                    &output,
                    batch_count,
                    position,
                    cache.page_size,
                    cache.page_table_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    window,
                    &self.stream,
                )?;
            } else {
                if window > 0 {
                    bail!(
                        "CUDA sliding-window attention requires the tiled kernel (head_dim <= {FLASH_ONLINE_MAX_HEAD_DIM})"
                    );
                }
                crate::kernels::launch_paged_decode_attention_batched(
                    &q.buffer,
                    cache.key_pages.as_buffer(),
                    cache.value_pages.as_buffer(),
                    &cache.page_table,
                    &output,
                    batch_count,
                    position,
                    cache.page_size,
                    cache.page_table_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: batch_count,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn paged_decode_attention_batched_positions_f32_device(
            &self,
            q: &GpuF32Tensor,
            cache: &CudaPagedBatchLayerKvCache,
            positions: &DeviceBuffer,
            batch_count: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,

            window: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != batch_count || q.cols != heads * head_dim {
                bail!(
                    "CUDA paged batched positioned attention q shape {}x{} is invalid",
                    q.rows,
                    q.cols
                );
            }
            if cache.batch_count != batch_count {
                bail!(
                    "CUDA paged batched positioned attention cache batch {} does not match q batch {batch_count}",
                    cache.batch_count
                );
            }
            let output_elements = batch_count
                .checked_mul(heads)
                .and_then(|value| value.checked_mul(v_head_dim))
                .context(
                    "CUDA paged batched positioned attention output element count overflows usize",
                )?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA paged batched positioned attention output")?;
            if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_tiled_paged_decode_attention_batched_positions(
                    &q.buffer,
                    cache.key_pages.as_buffer(),
                    cache.value_pages.as_buffer(),
                    &cache.page_table,
                    positions,
                    &output,
                    batch_count,
                    cache.page_size,
                    cache.page_table_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    window,
                    &self.stream,
                )?;
            } else {
                if window > 0 {
                    bail!(
                        "CUDA sliding-window attention requires the tiled kernel (head_dim <= {FLASH_ONLINE_MAX_HEAD_DIM})"
                    );
                }
                crate::kernels::launch_paged_decode_attention_batched_positions(
                    &q.buffer,
                    cache.key_pages.as_buffer(),
                    cache.value_pages.as_buffer(),
                    &cache.page_table,
                    positions,
                    &output,
                    batch_count,
                    cache.page_size,
                    cache.page_table_len,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: batch_count,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn cached_decode_attention_batched_f32_device(
            &self,
            q: &GpuF32Tensor,
            cache: &CudaLayerKvCache,
            batch_count: usize,
            position: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
        ) -> Result<GpuF32Tensor> {
            if q.rows != batch_count || q.cols != heads * head_dim {
                bail!(
                    "CUDA batched cached attention q shape {}x{} is invalid",
                    q.rows,
                    q.cols
                );
            }
            if cache.batch_count != batch_count {
                bail!(
                    "CUDA batched cached attention cache batch {} does not match q batch {batch_count}",
                    cache.batch_count
                );
            }
            let output_elements = batch_count
                .checked_mul(heads)
                .and_then(|value| value.checked_mul(v_head_dim))
                .context("CUDA batched cached attention output element count overflows usize")?;
            let output = DeviceBuffer::alloc(output_elements * std::mem::size_of::<f32>())
                .context("allocating CUDA batched cached attention output")?;
            if head_dim <= FLASH_ONLINE_MAX_HEAD_DIM {
                crate::kernels::launch_flash_cached_decode_attention_batched(
                    &q.buffer,
                    &cache.key,
                    &cache.value,
                    &output,
                    batch_count,
                    position,
                    cache.max_seq,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            } else {
                crate::kernels::launch_cached_decode_attention_batched(
                    &q.buffer,
                    &cache.key,
                    &cache.value,
                    &output,
                    batch_count,
                    position,
                    cache.max_seq,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    &self.stream,
                )?;
            }
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: batch_count,
                cols: heads * v_head_dim,
                buffer: output,
            })
        }

        fn silu_mul_f32_device(
            &self,
            gate: &GpuF32Tensor,
            up: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            gate.ensure_same_shape(up, "CUDA SwiGLU")?;
            let elements = gate.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA SwiGLU output")?;
            crate::kernels::launch_silu_mul(
                &gate.buffer,
                &up.buffer,
                &output,
                elements,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: gate.rows,
                cols: gate.cols,
                buffer: output,
            })
        }

        /// GeGLU: `gelu(gate) * up` (Gemma's MLP activation), the tanh-gelu
        /// counterpart of `silu_mul_f32_device`.
        fn gelu_mul_f32_device(
            &self,
            gate: &GpuF32Tensor,
            up: &GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            gate.ensure_same_shape(up, "CUDA GeGLU")?;
            let elements = gate.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA GeGLU output")?;
            crate::kernels::launch_gelu_mul(
                &gate.buffer,
                &up.buffer,
                &output,
                elements,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: gate.rows,
                cols: gate.cols,
                buffer: output,
            })
        }

        /// Apply Gemma logit soft-capping (`cap * tanh(x/cap)`) to the final
        /// logits when the model declares `final_logit_softcapping`. Monotonic, so
        /// greedy argmax is unchanged; it shapes the sampling distribution.
        fn apply_final_logit_softcap(&self, logits: GpuF32Tensor) -> Result<GpuF32Tensor> {
            let Some(cap) = self.config.final_logit_softcapping else {
                return Ok(logits);
            };
            if !(cap > 0.0) {
                return Ok(logits);
            }
            let elements = logits.element_count()?;
            let output = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating CUDA final logit softcap output")?;
            crate::kernels::launch_softcap(&logits.buffer, &output, elements, cap, &self.stream)?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: logits.rows,
                cols: logits.cols,
                buffer: output,
            })
        }

        /// First present Gemma post-norm vector among `aliases` (each is a
        /// `{prefix}.{alias}.weight` name), or None.
        fn gemma_post_norm_name(&self, prefix: &str, aliases: &[&str]) -> Option<String> {
            aliases
                .iter()
                .map(|alias| format!("{prefix}.{alias}.weight"))
                .find(|name| self.has_vector(name))
        }

        /// Gemma-2 post-attention norm, applied to the attention sub-layer output
        /// before the residual add (`residual + post_norm(attn(x))`). No-op for
        /// non-Gemma models and Gemma-1 (which lack the tensor).
        fn apply_gemma_post_attn_norm(
            &self,
            prefix: &str,
            output: GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            if !self.config.is_gemma() {
                return Ok(output);
            }
            match self.gemma_post_norm_name(
                prefix,
                &[
                    "post_attention_norm",
                    "attn_post_norm",
                    "post_attention_layernorm",
                    "post_attention_layer_norm",
                ],
            ) {
                Some(name) => {
                    let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
                    self.rms_norm_f32_device(&name, &output, eps)
                }
                None => Ok(output),
            }
        }

        /// Gemma-2 post-FFN norm, applied to the MLP sub-layer output before the
        /// residual add (`residual + post_norm(mlp(x))`).
        fn apply_gemma_post_ffn_norm(
            &self,
            prefix: &str,
            output: GpuF32Tensor,
        ) -> Result<GpuF32Tensor> {
            if !self.config.is_gemma() {
                return Ok(output);
            }
            match self.gemma_post_norm_name(
                prefix,
                &[
                    "post_ffw_norm",
                    "post_feedforward_norm",
                    "post_feedforward_layernorm",
                    "post_feed_forward_norm",
                    "ffn_post_norm",
                ],
            ) {
                Some(name) => {
                    let eps = self.config.rms_norm_eps.unwrap_or(1.0e-6);
                    self.rms_norm_f32_device(&name, &output, eps)
                }
                None => Ok(output),
            }
        }

        /// RoPE base for the layer named by `prefix` (`blk.{i}`). Gemma-3 interleaves
        /// local (sliding) and global (full) attention layers with different RoPE
        /// bases: every 6th layer (index % 6 == 5) is global and uses the model's
        /// `rope.freq_base`; the other five-of-six local layers use base 10000. All
        /// other architectures use one base for every layer.
        fn layer_rope_base(&self, prefix: &str, default_base: f32) -> f32 {
            const GEMMA3_SLIDING_PATTERN: usize = 6;
            const GEMMA3_LOCAL_ROPE_BASE: f32 = 10000.0;
            if !self.config.is_gemma3() {
                return default_base;
            }
            let Some(layer) = prefix
                .strip_prefix("blk.")
                .and_then(|index| index.parse::<usize>().ok())
            else {
                return default_base;
            };
            if layer % GEMMA3_SLIDING_PATTERN != GEMMA3_SLIDING_PATTERN - 1 {
                GEMMA3_LOCAL_ROPE_BASE
            } else {
                default_base
            }
        }

        /// Sliding-window size for the layer named by `prefix`. Gemma-3 local
        /// layers (index % 6 != 5) attend only to the last `attention.sliding_window`
        /// tokens; global layers and every other architecture use 0 (unlimited).
        fn layer_attention_window(&self, prefix: &str) -> usize {
            const GEMMA3_SLIDING_PATTERN: usize = 6;
            if !self.config.is_gemma3() {
                return 0;
            }
            let Some(window) = self.config.attention_sliding_window else {
                return 0;
            };
            let Some(layer) = prefix
                .strip_prefix("blk.")
                .and_then(|index| index.parse::<usize>().ok())
            else {
                return 0;
            };
            if layer % GEMMA3_SLIDING_PATTERN != GEMMA3_SLIDING_PATTERN - 1 {
                window as usize
            } else {
                0
            }
        }

        fn copy_row_f32_device(&self, input: &GpuF32Tensor, row: usize) -> Result<GpuF32Tensor> {
            if row >= input.rows {
                bail!(
                    "CUDA row copy index {row} is outside tensor row count {}",
                    input.rows
                );
            }
            let output = DeviceBuffer::alloc(input.cols * std::mem::size_of::<f32>())
                .context("allocating CUDA row copy output")?;
            crate::kernels::launch_copy_row_f32(
                &input.buffer,
                &output,
                row,
                input.rows,
                input.cols,
                &self.stream,
            )?;
            self.op_barrier()?;
            Ok(GpuF32Tensor {
                rows: 1,
                cols: input.cols,
                buffer: output,
            })
        }

        fn add_scaled_row_in_place_device(
            &self,
            output: &DeviceBuffer,
            row_values: &GpuF32Tensor,
            row: usize,
            rows: usize,
            cols: usize,
            scale: f32,
        ) -> Result<()> {
            if row_values.rows != 1 || row_values.cols != cols {
                bail!(
                    "CUDA scaled row add input shape {}x{} does not match expected 1x{cols}",
                    row_values.rows,
                    row_values.cols
                );
            }
            let expected_bytes = rows
                .checked_mul(cols)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .context("CUDA scaled row add output byte count overflows usize")?;
            if output.bytes() < expected_bytes {
                bail!(
                    "CUDA scaled row add output has {} bytes; expected at least {expected_bytes}",
                    output.bytes()
                );
            }
            crate::kernels::launch_add_scaled_row_in_place(
                output,
                &row_values.buffer,
                row,
                rows,
                cols,
                scale,
                &self.stream,
            )?;
            self.op_barrier()
        }
    }

    impl fmt::Debug for CudaQwenGpuModel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CudaQwenGpuModel")
                .field("info", &self.info)
                .field("tensor_names", &self.tensors.keys().collect::<Vec<_>>())
                .field("matrix_names", &self.matrices.keys().collect::<Vec<_>>())
                .field("vector_names", &self.vectors.keys().collect::<Vec<_>>())
                .finish_non_exhaustive()
        }
    }

    struct CudaKvCache {
        layers: Vec<CudaLayerKvCache>,
    }

    impl CudaKvCache {
        fn new(layer_count: u32, dims: &QwenDims, stream: &Stream) -> Result<Self> {
            Self::new_batched(layer_count, dims, 1, stream)
        }

        fn new_batched(
            layer_count: u32,
            dims: &QwenDims,
            batch_count: usize,
            stream: &Stream,
        ) -> Result<Self> {
            if batch_count == 0 {
                bail!("CUDA KV cache batch_count must be greater than zero");
            }
            let layer_count =
                usize::try_from(layer_count).context("qwen block_count does not fit usize")?;
            let key_elements = dims
                .kv_heads
                .checked_mul(batch_count)
                .and_then(|value| value.checked_mul(dims.context))
                .and_then(|value| value.checked_mul(dims.head_dim))
                .context("CUDA KV key cache element count overflows usize")?;
            let value_elements = dims
                .kv_heads
                .checked_mul(batch_count)
                .and_then(|value| value.checked_mul(dims.context))
                .and_then(|value| value.checked_mul(dims.v_head_dim))
                .context("CUDA KV value cache element count overflows usize")?;
            let key_bytes = key_elements
                .checked_mul(std::mem::size_of::<f32>())
                .context("CUDA KV key cache byte count overflows usize")?;
            let value_bytes = value_elements
                .checked_mul(std::mem::size_of::<f32>())
                .context("CUDA KV value cache byte count overflows usize")?;
            let mut layers = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let key = DeviceBuffer::alloc(key_bytes).context("allocating CUDA key cache")?;
                let value =
                    DeviceBuffer::alloc(value_bytes).context("allocating CUDA value cache")?;
                key.memset_zero_async(stream)?;
                value.memset_zero_async(stream)?;
                layers.push(CudaLayerKvCache {
                    key,
                    value,
                    max_seq: dims.context,
                    batch_count,
                });
            }
            stream.synchronize()?;
            Ok(Self { layers })
        }

        fn layer(&self, idx: usize) -> Result<&CudaLayerKvCache> {
            self.layers
                .get(idx)
                .ok_or_else(|| anyhow!("CUDA KV cache layer {idx} is missing"))
        }

        fn write_layer(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            let layer = self
                .layers
                .get_mut(idx)
                .ok_or_else(|| anyhow!("CUDA KV cache layer {idx} is missing"))?;
            layer.write(
                key, value, start_pos, kv_heads, head_dim, v_head_dim, stream,
            )
        }

        fn write_layer_batched(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            let layer = self
                .layers
                .get_mut(idx)
                .ok_or_else(|| anyhow!("CUDA KV cache layer {idx} is missing"))?;
            layer.write_batched(
                key,
                value,
                batch_count,
                row_count,
                start_pos,
                kv_heads,
                head_dim,
                v_head_dim,
                stream,
            )
        }
    }

    struct CudaPagedKvCache {
        layers: Vec<CudaPagedLayerKvCache>,
    }

    impl CudaPagedKvCache {
        fn new(
            layer_count: u32,
            dims: &QwenDims,
            page_size: usize,
            stream: &Stream,
        ) -> Result<Self> {
            Self::new_for_token_capacity(layer_count, dims, page_size, dims.context, stream)
        }

        fn new_for_token_capacity(
            layer_count: u32,
            dims: &QwenDims,
            page_size: usize,
            token_capacity: usize,
            stream: &Stream,
        ) -> Result<Self> {
            if page_size == 0 {
                bail!("CUDA paged KV cache page_size must be greater than zero");
            }
            if token_capacity == 0 {
                bail!("CUDA paged KV cache token_capacity must be greater than zero");
            }
            if token_capacity > dims.context {
                bail!(
                    "CUDA paged KV cache token_capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            let layer_count =
                usize::try_from(layer_count).context("qwen block_count does not fit usize")?;
            let page_table_len = token_capacity.div_ceil(page_size);
            if page_table_len == 0 {
                bail!("CUDA paged KV cache page table must contain at least one page");
            }
            let key_elements = page_table_len
                .checked_mul(dims.kv_heads)
                .and_then(|value| value.checked_mul(page_size))
                .and_then(|value| value.checked_mul(dims.head_dim))
                .context("CUDA paged KV key cache element count overflows usize")?;
            let value_elements = page_table_len
                .checked_mul(dims.kv_heads)
                .and_then(|value| value.checked_mul(page_size))
                .and_then(|value| value.checked_mul(dims.v_head_dim))
                .context("CUDA paged KV value cache element count overflows usize")?;
            let key_bytes = key_elements
                .checked_mul(std::mem::size_of::<u16>()) // f16 KV pages (kv_t = __half)
                .context("CUDA paged KV key cache byte count overflows usize")?;
            let value_bytes = value_elements
                .checked_mul(std::mem::size_of::<u16>()) // f16 KV pages (kv_t = __half)
                .context("CUDA paged KV value cache byte count overflows usize")?;
            let page_table = (0..page_table_len)
                .map(|page| u32::try_from(page).context("CUDA KV page index does not fit u32"))
                .collect::<Result<Vec<_>>>()?;
            let page_table_bytes = page_table
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .context("CUDA paged KV page table byte count overflows usize")?;
            let mut layers = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let key_pages =
                    DeviceBuffer::alloc(key_bytes).context("allocating CUDA paged key cache")?;
                let value_pages = DeviceBuffer::alloc(value_bytes)
                    .context("allocating CUDA paged value cache")?;
                let page_table_device = DeviceBuffer::alloc(page_table_bytes)
                    .context("allocating CUDA paged KV page table")?;
                key_pages.memset_zero_async(stream)?;
                value_pages.memset_zero_async(stream)?;
                page_table_device
                    .copy_from_host(&page_table)
                    .context("copying CUDA paged KV page table")?;
                layers.push(CudaPagedLayerKvCache {
                    key_pages,
                    value_pages,
                    page_table: page_table_device,
                    page_size,
                    page_table_len,
                    max_seq: token_capacity,
                });
            }
            stream.synchronize()?;
            Ok(Self { layers })
        }

        fn layer(&self, idx: usize) -> Result<&CudaPagedLayerKvCache> {
            self.layers
                .get(idx)
                .ok_or_else(|| anyhow!("CUDA paged KV cache layer {idx} is missing"))
        }

        fn write_layer(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            let layer = self
                .layers
                .get_mut(idx)
                .ok_or_else(|| anyhow!("CUDA paged KV cache layer {idx} is missing"))?;
            layer.write(
                key, value, start_pos, kv_heads, head_dim, v_head_dim, stream,
            )
        }
    }

    struct CudaPagedBatchDevicePool {
        layer_count: usize,
        page_size: usize,
        physical_page_count: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        layers: Vec<CudaPagedBatchDevicePoolLayer>,
    }

    struct CudaPagedBatchDevicePoolLayer {
        key_pages: DeviceBuffer,
        value_pages: DeviceBuffer,
    }

    impl CudaPagedBatchDevicePool {
        fn new(
            layer_count: u32,
            dims: &QwenDims,
            page_size: usize,
            physical_page_count: usize,
            stream: &Stream,
        ) -> Result<Self> {
            if page_size == 0 {
                bail!("CUDA paged batch device pool page_size must be greater than zero");
            }
            if physical_page_count == 0 {
                bail!("CUDA paged batch device pool physical_page_count must be greater than zero");
            }
            let layer_count =
                usize::try_from(layer_count).context("qwen block_count does not fit usize")?;
            let key_elements = physical_page_count
                .checked_mul(dims.kv_heads)
                .and_then(|value| value.checked_mul(page_size))
                .and_then(|value| value.checked_mul(dims.head_dim))
                .context("CUDA paged batch device key pool element count overflows usize")?;
            let value_elements = physical_page_count
                .checked_mul(dims.kv_heads)
                .and_then(|value| value.checked_mul(page_size))
                .and_then(|value| value.checked_mul(dims.v_head_dim))
                .context("CUDA paged batch device value pool element count overflows usize")?;
            let key_bytes = key_elements
                .checked_mul(std::mem::size_of::<u16>()) // f16 KV pages (kv_t = __half)
                .context("CUDA paged batch device key pool byte count overflows usize")?;
            let value_bytes = value_elements
                .checked_mul(std::mem::size_of::<u16>()) // f16 KV pages (kv_t = __half)
                .context("CUDA paged batch device value pool byte count overflows usize")?;
            let mut layers = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let key_pages = DeviceBuffer::alloc(key_bytes)
                    .context("allocating CUDA paged batch key pool")?;
                let value_pages = DeviceBuffer::alloc(value_bytes)
                    .context("allocating CUDA paged batch value pool")?;
                key_pages.memset_zero_async(stream)?;
                value_pages.memset_zero_async(stream)?;
                layers.push(CudaPagedBatchDevicePoolLayer {
                    key_pages,
                    value_pages,
                });
            }
            stream.synchronize()?;
            Ok(Self {
                layer_count,
                page_size,
                physical_page_count,
                kv_heads: dims.kv_heads,
                head_dim: dims.head_dim,
                v_head_dim: dims.v_head_dim,
                layers,
            })
        }

        fn matches(&self, dims: &QwenDims, page_size: usize) -> bool {
            self.layer_count == self.layers.len()
                && self.page_size == page_size
                && self.kv_heads == dims.kv_heads
                && self.head_dim == dims.head_dim
                && self.v_head_dim == dims.v_head_dim
        }

        fn can_cover(&self, dims: &QwenDims, page_size: usize, physical_page_count: usize) -> bool {
            self.matches(dims, page_size) && self.physical_page_count >= physical_page_count
        }

        fn zero(&self, stream: &Stream) -> Result<()> {
            for layer in &self.layers {
                layer.key_pages.memset_zero_async(stream)?;
                layer.value_pages.memset_zero_async(stream)?;
            }
            stream.synchronize()
        }
    }

    struct CudaPagedBatchKvCache<'a> {
        layers: Vec<CudaPagedBatchLayerKvCache<'a>>,
    }

    impl<'a> CudaPagedBatchKvCache<'a> {
        fn new(
            layer_count: u32,
            dims: &QwenDims,
            batch_count: usize,
            page_size: usize,
            token_capacity: usize,
            stream: &Stream,
        ) -> Result<Self> {
            if batch_count == 0 {
                bail!("CUDA paged batch KV cache batch_count must be greater than zero");
            }
            if page_size == 0 {
                bail!("CUDA paged batch KV cache page_size must be greater than zero");
            }
            if token_capacity == 0 {
                bail!("CUDA paged batch KV cache token_capacity must be greater than zero");
            }
            if token_capacity > dims.context {
                bail!(
                    "CUDA paged batch KV cache token_capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            let layer_count =
                usize::try_from(layer_count).context("qwen block_count does not fit usize")?;
            let page_table_len = token_capacity.div_ceil(page_size);
            if page_table_len == 0 {
                bail!("CUDA paged batch KV cache page table must contain at least one page");
            }
            let physical_pages = batch_count
                .checked_mul(page_table_len)
                .context("CUDA paged batch KV physical page count overflows usize")?;
            let key_elements = physical_pages
                .checked_mul(dims.kv_heads)
                .and_then(|value| value.checked_mul(page_size))
                .and_then(|value| value.checked_mul(dims.head_dim))
                .context("CUDA paged batch KV key cache element count overflows usize")?;
            let value_elements = physical_pages
                .checked_mul(dims.kv_heads)
                .and_then(|value| value.checked_mul(page_size))
                .and_then(|value| value.checked_mul(dims.v_head_dim))
                .context("CUDA paged batch KV value cache element count overflows usize")?;
            let key_bytes = key_elements
                .checked_mul(std::mem::size_of::<u16>()) // f16 KV pages (kv_t = __half)
                .context("CUDA paged batch KV key cache byte count overflows usize")?;
            let value_bytes = value_elements
                .checked_mul(std::mem::size_of::<u16>()) // f16 KV pages (kv_t = __half)
                .context("CUDA paged batch KV value cache byte count overflows usize")?;
            let mut page_table = Vec::with_capacity(physical_pages);
            for batch in 0..batch_count {
                for page in 0..page_table_len {
                    let physical_page = batch
                        .checked_mul(page_table_len)
                        .and_then(|value| value.checked_add(page))
                        .context("CUDA paged batch KV page table index overflows usize")?;
                    page_table.push(
                        u32::try_from(physical_page)
                            .context("CUDA paged batch KV page index does not fit u32")?,
                    );
                }
            }
            let page_table_bytes = page_table
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .context("CUDA paged batch KV page table byte count overflows usize")?;
            let mut layers = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let key_pages = DeviceBuffer::alloc(key_bytes)
                    .context("allocating CUDA paged batch key cache")?;
                let value_pages = DeviceBuffer::alloc(value_bytes)
                    .context("allocating CUDA paged batch value cache")?;
                let page_table_device = DeviceBuffer::alloc(page_table_bytes)
                    .context("allocating CUDA paged batch KV page table")?;
                key_pages.memset_zero_async(stream)?;
                value_pages.memset_zero_async(stream)?;
                page_table_device
                    .copy_from_host(&page_table)
                    .context("copying CUDA paged batch KV page table")?;
                layers.push(CudaPagedBatchLayerKvCache {
                    key_pages: CudaPagedBatchBuffer::Owned(key_pages),
                    value_pages: CudaPagedBatchBuffer::Owned(value_pages),
                    page_table: page_table_device,
                    batch_count,
                    page_size,
                    page_table_len,
                    max_seq: token_capacity,
                });
            }
            stream.synchronize()?;
            Ok(Self { layers })
        }

        fn new_with_page_tables_from_pool(
            dims: &QwenDims,
            batch_count: usize,
            page_size: usize,
            token_capacity: usize,
            page_tables: &[Vec<usize>],
            pool: &'a CudaPagedBatchDevicePool,
            stream: &Stream,
        ) -> Result<Self> {
            if batch_count == 0 {
                bail!("CUDA paged batch KV cache batch_count must be greater than zero");
            }
            if page_tables.len() != batch_count {
                bail!(
                    "CUDA paged batch KV cache got {} page tables for batch {batch_count}",
                    page_tables.len()
                );
            }
            if page_size == 0 {
                bail!("CUDA paged batch KV cache page_size must be greater than zero");
            }
            if token_capacity == 0 {
                bail!("CUDA paged batch KV cache token_capacity must be greater than zero");
            }
            if token_capacity > dims.context {
                bail!(
                    "CUDA paged batch KV cache token_capacity {token_capacity} exceeds qwen context length {}",
                    dims.context
                );
            }
            if !pool.matches(dims, page_size) {
                bail!("CUDA paged batch KV device pool shape does not match requested cache");
            }
            let page_table_len = token_capacity.div_ceil(page_size);
            if page_table_len == 0 {
                bail!("CUDA paged batch KV cache page table must contain at least one page");
            }
            let mut page_table = Vec::with_capacity(batch_count * page_table_len);
            for (batch_idx, pages) in page_tables.iter().enumerate() {
                if pages.len() < page_table_len {
                    bail!(
                        "CUDA paged batch KV cache page table {batch_idx} has {} page(s), expected at least {page_table_len}",
                        pages.len()
                    );
                }
                for page in pages.iter().take(page_table_len).copied() {
                    if page >= pool.physical_page_count {
                        bail!(
                            "CUDA paged batch KV cache page index {page} exceeds physical page count {}",
                            pool.physical_page_count
                        );
                    }
                    page_table.push(
                        u32::try_from(page)
                            .context("CUDA paged batch KV page index does not fit u32")?,
                    );
                }
            }
            let page_table_bytes = page_table
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .context("CUDA paged batch KV page table byte count overflows usize")?;
            let mut layers = Vec::with_capacity(pool.layers.len());
            for pool_layer in &pool.layers {
                let page_table_device = DeviceBuffer::alloc(page_table_bytes)
                    .context("allocating CUDA paged batch KV page table")?;
                page_table_device
                    .copy_from_host(&page_table)
                    .context("copying CUDA paged batch KV page table")?;
                layers.push(CudaPagedBatchLayerKvCache {
                    key_pages: CudaPagedBatchBuffer::Borrowed(&pool_layer.key_pages),
                    value_pages: CudaPagedBatchBuffer::Borrowed(&pool_layer.value_pages),
                    page_table: page_table_device,
                    batch_count,
                    page_size,
                    page_table_len,
                    max_seq: token_capacity,
                });
            }
            stream.synchronize()?;
            Ok(Self { layers })
        }

        fn layer(&self, idx: usize) -> Result<&CudaPagedBatchLayerKvCache<'a>> {
            self.layers
                .get(idx)
                .ok_or_else(|| anyhow!("CUDA paged batch KV cache layer {idx} is missing"))
        }

        fn write_layer_batched(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            let layer = self
                .layers
                .get_mut(idx)
                .ok_or_else(|| anyhow!("CUDA paged batch KV cache layer {idx} is missing"))?;
            layer.write_batched(
                key,
                value,
                batch_count,
                row_count,
                start_pos,
                kv_heads,
                head_dim,
                v_head_dim,
                stream,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn write_layer_batched_positions(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            positions: &DeviceBuffer,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            let layer = self
                .layers
                .get_mut(idx)
                .ok_or_else(|| anyhow!("CUDA paged batch KV cache layer {idx} is missing"))?;
            layer.write_batched_positions(
                key,
                value,
                batch_count,
                row_count,
                positions,
                kv_heads,
                head_dim,
                v_head_dim,
                stream,
            )
        }

        fn copy_prefix_from_first_batch(
            &self,
            token_count: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            for layer in &self.layers {
                layer.copy_prefix_from_first_batch(
                    token_count,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    impl CudaBatchedKvCacheWrite for CudaKvCache {
        fn write_layer_batched(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            CudaKvCache::write_layer_batched(
                self,
                idx,
                key,
                value,
                batch_count,
                row_count,
                start_pos,
                kv_heads,
                head_dim,
                v_head_dim,
                stream,
            )
        }
    }

    impl CudaSingleKvCacheWrite for CudaKvCache {
        fn write_layer(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            CudaKvCache::write_layer(
                self, idx, key, value, start_pos, kv_heads, head_dim, v_head_dim, stream,
            )
        }
    }

    impl CudaSingleKvCacheWrite for CudaPagedKvCache {
        fn write_layer(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            CudaPagedKvCache::write_layer(
                self, idx, key, value, start_pos, kv_heads, head_dim, v_head_dim, stream,
            )
        }
    }

    impl CudaBatchedKvCacheWrite for CudaPagedBatchKvCache<'_> {
        fn write_layer_batched(
            &mut self,
            idx: usize,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            CudaPagedBatchKvCache::write_layer_batched(
                self,
                idx,
                key,
                value,
                batch_count,
                row_count,
                start_pos,
                kv_heads,
                head_dim,
                v_head_dim,
                stream,
            )
        }
    }

    struct CudaPagedLayerKvCache {
        key_pages: DeviceBuffer,
        value_pages: DeviceBuffer,
        page_table: DeviceBuffer,
        page_size: usize,
        page_table_len: usize,
        max_seq: usize,
    }

    impl CudaPagedLayerKvCache {
        fn write(
            &mut self,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            if key.rows != value.rows {
                bail!(
                    "CUDA paged KV cache write row mismatch: key {}, value {}",
                    key.rows,
                    value.rows
                );
            }
            if key.cols != kv_heads * head_dim {
                bail!(
                    "CUDA paged KV cache key tensor shape {}x{} does not match kv shape {}x{}",
                    key.rows,
                    key.cols,
                    kv_heads,
                    head_dim
                );
            }
            if value.cols != kv_heads * v_head_dim {
                bail!(
                    "CUDA paged KV cache value tensor shape {}x{} does not match kv shape {}x{}",
                    value.rows,
                    value.cols,
                    kv_heads,
                    v_head_dim
                );
            }
            if start_pos
                .checked_add(key.rows)
                .is_none_or(|end| end > self.max_seq)
            {
                bail!(
                    "CUDA paged KV cache write range {}..{} exceeds max sequence {}",
                    start_pos,
                    start_pos.saturating_add(key.rows),
                    self.max_seq
                );
            }
            crate::kernels::launch_write_paged_kv_cache(
                &key.buffer,
                &self.key_pages,
                &self.page_table,
                key.rows,
                kv_heads,
                head_dim,
                self.page_size,
                self.page_table_len,
                start_pos,
                stream,
            )?;
            crate::kernels::launch_write_paged_kv_cache(
                &value.buffer,
                &self.value_pages,
                &self.page_table,
                value.rows,
                kv_heads,
                v_head_dim,
                self.page_size,
                self.page_table_len,
                start_pos,
                stream,
            )?;
            stream.synchronize()
        }
    }

    enum CudaPagedBatchBuffer<'a> {
        Owned(DeviceBuffer),
        Borrowed(&'a DeviceBuffer),
    }

    impl CudaPagedBatchBuffer<'_> {
        fn as_buffer(&self) -> &DeviceBuffer {
            match self {
                Self::Owned(buffer) => buffer,
                Self::Borrowed(buffer) => buffer,
            }
        }
    }

    struct CudaPagedBatchLayerKvCache<'a> {
        key_pages: CudaPagedBatchBuffer<'a>,
        value_pages: CudaPagedBatchBuffer<'a>,
        page_table: DeviceBuffer,
        batch_count: usize,
        page_size: usize,
        page_table_len: usize,
        max_seq: usize,
    }

    impl CudaPagedBatchLayerKvCache<'_> {
        fn write_batched(
            &mut self,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            if key.rows != value.rows {
                bail!(
                    "CUDA paged batch KV cache write row mismatch: key {}, value {}",
                    key.rows,
                    value.rows
                );
            }
            if batch_count != self.batch_count {
                bail!(
                    "CUDA paged batch KV cache write got batch {batch_count}; expected {}",
                    self.batch_count
                );
            }
            if key.rows != batch_count * row_count || key.cols != kv_heads * head_dim {
                bail!(
                    "CUDA paged batch KV cache key tensor shape {}x{} does not match batch {batch_count} x rows {row_count} x kv shape {}x{}",
                    key.rows,
                    key.cols,
                    kv_heads,
                    head_dim
                );
            }
            if value.rows != batch_count * row_count || value.cols != kv_heads * v_head_dim {
                bail!(
                    "CUDA paged batch KV cache value tensor shape {}x{} does not match batch {batch_count} x rows {row_count} x kv shape {}x{}",
                    value.rows,
                    value.cols,
                    kv_heads,
                    v_head_dim
                );
            }
            if start_pos
                .checked_add(row_count)
                .is_none_or(|end| end > self.max_seq)
            {
                bail!(
                    "CUDA paged batch KV cache write range {}..{} exceeds max sequence {}",
                    start_pos,
                    start_pos.saturating_add(row_count),
                    self.max_seq
                );
            }
            crate::kernels::launch_write_paged_kv_cache_batched(
                &key.buffer,
                self.key_pages.as_buffer(),
                &self.page_table,
                batch_count,
                row_count,
                kv_heads,
                head_dim,
                self.page_size,
                self.page_table_len,
                start_pos,
                stream,
            )?;
            crate::kernels::launch_write_paged_kv_cache_batched(
                &value.buffer,
                self.value_pages.as_buffer(),
                &self.page_table,
                batch_count,
                row_count,
                kv_heads,
                v_head_dim,
                self.page_size,
                self.page_table_len,
                start_pos,
                stream,
            )?;
            stream.synchronize()
        }

        #[allow(clippy::too_many_arguments)]
        fn write_batched_positions(
            &mut self,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            positions: &DeviceBuffer,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            if key.rows != value.rows {
                bail!(
                    "CUDA paged batch positioned KV cache write row mismatch: key {}, value {}",
                    key.rows,
                    value.rows
                );
            }
            if batch_count != self.batch_count {
                bail!(
                    "CUDA paged batch positioned KV cache write got batch {batch_count}; expected {}",
                    self.batch_count
                );
            }
            if key.rows != batch_count * row_count || key.cols != kv_heads * head_dim {
                bail!(
                    "CUDA paged batch positioned KV cache key tensor shape {}x{} does not match batch {batch_count} x rows {row_count} x kv shape {}x{}",
                    key.rows,
                    key.cols,
                    kv_heads,
                    head_dim
                );
            }
            if value.rows != batch_count * row_count || value.cols != kv_heads * v_head_dim {
                bail!(
                    "CUDA paged batch positioned KV cache value tensor shape {}x{} does not match batch {batch_count} x rows {row_count} x kv shape {}x{}",
                    value.rows,
                    value.cols,
                    kv_heads,
                    v_head_dim
                );
            }
            crate::kernels::launch_write_paged_kv_cache_batched_positions(
                &key.buffer,
                self.key_pages.as_buffer(),
                &self.page_table,
                positions,
                batch_count,
                row_count,
                kv_heads,
                head_dim,
                self.page_size,
                self.page_table_len,
                stream,
            )?;
            crate::kernels::launch_write_paged_kv_cache_batched_positions(
                &value.buffer,
                self.value_pages.as_buffer(),
                &self.page_table,
                positions,
                batch_count,
                row_count,
                kv_heads,
                v_head_dim,
                self.page_size,
                self.page_table_len,
                stream,
            )?;
            stream.synchronize()
        }

        fn copy_prefix_from_first_batch(
            &self,
            token_count: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            if token_count == 0 {
                return Ok(());
            }
            if self.batch_count < 2 {
                bail!("CUDA paged batch KV prefix copy requires at least two batch rows");
            }
            if token_count > self.max_seq {
                bail!(
                    "CUDA paged batch KV prefix copy token count {token_count} exceeds max sequence {}",
                    self.max_seq
                );
            }
            crate::kernels::launch_copy_paged_kv_cache_prefix_batched(
                self.key_pages.as_buffer(),
                &self.page_table,
                self.batch_count,
                token_count,
                kv_heads,
                head_dim,
                self.page_size,
                self.page_table_len,
                stream,
            )?;
            crate::kernels::launch_copy_paged_kv_cache_prefix_batched(
                self.value_pages.as_buffer(),
                &self.page_table,
                self.batch_count,
                token_count,
                kv_heads,
                v_head_dim,
                self.page_size,
                self.page_table_len,
                stream,
            )?;
            stream.synchronize()
        }
    }

    struct CudaLayerKvCache {
        key: DeviceBuffer,
        value: DeviceBuffer,
        max_seq: usize,
        batch_count: usize,
    }

    impl CudaLayerKvCache {
        fn write(
            &mut self,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            if key.rows != value.rows {
                bail!(
                    "CUDA KV cache write row mismatch: key {}, value {}",
                    key.rows,
                    value.rows
                );
            }
            if key.cols != kv_heads * head_dim {
                bail!(
                    "CUDA KV cache key tensor shape {}x{} does not match kv shape {}x{}",
                    key.rows,
                    key.cols,
                    kv_heads,
                    head_dim
                );
            }
            if value.cols != kv_heads * v_head_dim {
                bail!(
                    "CUDA KV cache value tensor shape {}x{} does not match kv shape {}x{}",
                    value.rows,
                    value.cols,
                    kv_heads,
                    v_head_dim
                );
            }
            if start_pos
                .checked_add(key.rows)
                .is_none_or(|end| end > self.max_seq)
            {
                bail!(
                    "CUDA KV cache write range {}..{} exceeds max sequence {}",
                    start_pos,
                    start_pos.saturating_add(key.rows),
                    self.max_seq
                );
            }
            crate::kernels::launch_write_kv_cache(
                &key.buffer,
                &self.key,
                key.rows,
                kv_heads,
                head_dim,
                self.max_seq,
                start_pos,
                stream,
            )?;
            crate::kernels::launch_write_kv_cache(
                &value.buffer,
                &self.value,
                value.rows,
                kv_heads,
                v_head_dim,
                self.max_seq,
                start_pos,
                stream,
            )?;
            stream.synchronize()
        }

        fn write_batched(
            &mut self,
            key: &GpuF32Tensor,
            value: &GpuF32Tensor,
            batch_count: usize,
            row_count: usize,
            start_pos: usize,
            kv_heads: usize,
            head_dim: usize,
            v_head_dim: usize,
            stream: &Stream,
        ) -> Result<()> {
            if key.rows != value.rows {
                bail!(
                    "CUDA batched KV cache write row mismatch: key {}, value {}",
                    key.rows,
                    value.rows
                );
            }
            if batch_count != self.batch_count {
                bail!(
                    "CUDA batched KV cache write batch {batch_count} does not match cache batch {}",
                    self.batch_count
                );
            }
            if key.rows != batch_count * row_count || key.cols != kv_heads * head_dim {
                bail!(
                    "CUDA batched KV cache key tensor shape {}x{} does not match batch {batch_count} row_count {row_count} kv shape {}x{}",
                    key.rows,
                    key.cols,
                    kv_heads,
                    head_dim
                );
            }
            if value.rows != batch_count * row_count || value.cols != kv_heads * v_head_dim {
                bail!(
                    "CUDA batched KV cache value tensor shape {}x{} does not match batch {batch_count} row_count {row_count} kv shape {}x{}",
                    value.rows,
                    value.cols,
                    kv_heads,
                    v_head_dim
                );
            }
            if start_pos
                .checked_add(row_count)
                .is_none_or(|end| end > self.max_seq)
            {
                bail!(
                    "CUDA batched KV cache write range {}..{} exceeds max sequence {}",
                    start_pos,
                    start_pos.saturating_add(row_count),
                    self.max_seq
                );
            }
            crate::kernels::launch_write_kv_cache_batched(
                &key.buffer,
                &self.key,
                batch_count,
                row_count,
                kv_heads,
                head_dim,
                self.max_seq,
                start_pos,
                stream,
            )?;
            crate::kernels::launch_write_kv_cache_batched(
                &value.buffer,
                &self.value,
                batch_count,
                row_count,
                kv_heads,
                v_head_dim,
                self.max_seq,
                start_pos,
                stream,
            )?;
            stream.synchronize()
        }
    }

    pub struct GpuTensor {
        pub shape: Vec<u64>,
        pub dtype: GgufTensorType,
        pub bytes: usize,
    }

    impl fmt::Debug for GpuTensor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("GpuTensor")
                .field("shape", &self.shape)
                .field("dtype", &self.dtype)
                .field("bytes", &self.bytes)
                .finish_non_exhaustive()
        }
    }

    pub struct GpuMatrix {
        pub rows: usize,
        pub cols: usize,
        pub dtype: GgufTensorType,
        pub bytes: usize,
        buffer: DeviceBuffer,
    }

    impl GpuMatrix {
        fn load(gguf: &GgufFile, spec: &MatrixSpec) -> Result<Self> {
            let tensor = gguf
                .tensor(&spec.tensor_name)
                .ok_or_else(|| anyhow!("GGUF tensor {} is missing", spec.tensor_name))?;
            let dtype = tensor.info.dtype;
            if let Some(row_slice) = spec.row_slice {
                let matrix_bytes =
                    matrix_source_bytes(tensor.bytes, &tensor.info.dimensions, dtype, spec)?;
                let matrix_dims = matrix_source_dims(&tensor.info.dimensions, spec)?;
                return Self::load_row_slice(matrix_bytes, &matrix_dims, dtype, spec, row_slice);
            }
            let matrix_bytes =
                matrix_source_bytes(tensor.bytes, &tensor.info.dimensions, dtype, spec)?;
            let matrix_dims = matrix_source_dims(&tensor.info.dimensions, spec)?;
            if !dtype.is_quantized()
                && !matches!(
                    dtype,
                    GgufTensorType::F16 | GgufTensorType::BF16 | GgufTensorType::F32
                )
            {
                bail!(
                    "matrix tensor {} has dtype {}; CUDA normalized matrices require FP16/BF16/F32 storage",
                    spec.name,
                    dtype.label()
                );
            }
            if spec.name == "patch" && !dtype.is_quantized() {
                if let Some(bytes) = normalize_vision_patch_matrix_bytes(
                    gguf,
                    matrix_bytes,
                    &matrix_dims,
                    dtype,
                    spec,
                )? {
                    let buffer = DeviceBuffer::alloc(bytes.len()).with_context(|| {
                        format!("allocating CUDA normalized matrix {}", spec.name)
                    })?;
                    buffer.copy_from_host(&bytes).with_context(|| {
                        format!("copying normalized matrix {} to CUDA device", spec.name)
                    })?;
                    return Ok(Self {
                        rows: spec.rows,
                        cols: spec.cols,
                        dtype,
                        bytes: bytes.len(),
                        buffer,
                    });
                }
            }
            if dtype.is_quantized() {
                let bytes =
                    normalize_quantized_matrix_bytes(matrix_bytes, &matrix_dims, dtype, spec)?;
                let buffer = DeviceBuffer::alloc(bytes.len())
                    .with_context(|| format!("allocating CUDA quantized matrix {}", spec.name))?;
                buffer.copy_from_host(bytes).with_context(|| {
                    format!("copying quantized matrix {} to CUDA device", spec.name)
                })?;
                return Ok(Self {
                    rows: spec.rows,
                    cols: spec.cols,
                    dtype,
                    bytes: bytes.len(),
                    buffer,
                });
            }
            let bytes = normalize_matrix_bytes(matrix_bytes, &matrix_dims, dtype, spec)?;
            let buffer = DeviceBuffer::alloc(bytes.len())
                .with_context(|| format!("allocating CUDA normalized matrix {}", spec.name))?;
            buffer.copy_from_host(&bytes).with_context(|| {
                format!("copying normalized matrix {} to CUDA device", spec.name)
            })?;
            Ok(Self {
                rows: spec.rows,
                cols: spec.cols,
                dtype,
                bytes: bytes.len(),
                buffer,
            })
        }

        fn load_row_slice(
            bytes: &[u8],
            dims: &[u64],
            dtype: GgufTensorType,
            spec: &MatrixSpec,
            row_slice: RowSlice,
        ) -> Result<Self> {
            if !dtype.is_quantized()
                && !matches!(
                    dtype,
                    GgufTensorType::F16 | GgufTensorType::BF16 | GgufTensorType::F32
                )
            {
                bail!(
                    "matrix tensor {} has dtype {}; CUDA normalized matrices require FP16/BF16/F32 storage",
                    spec.name,
                    dtype.label()
                );
            }
            let bytes = if dtype.is_quantized() {
                normalize_quantized_row_slice_matrix_bytes(bytes, dims, dtype, spec, row_slice)?
                    .to_vec()
            } else {
                normalize_row_slice_matrix_bytes(bytes, dims, dtype, spec, row_slice)?
            };
            let buffer = DeviceBuffer::alloc(bytes.len())
                .with_context(|| format!("allocating CUDA normalized matrix {}", spec.name))?;
            buffer.copy_from_host(&bytes).with_context(|| {
                format!("copying normalized matrix {} to CUDA device", spec.name)
            })?;
            Ok(Self {
                rows: spec.rows,
                cols: spec.cols,
                dtype,
                bytes: bytes.len(),
                buffer,
            })
        }

        pub fn copy_to_host_u16(&self) -> Result<Vec<u16>> {
            if !matches!(self.dtype, GgufTensorType::F16 | GgufTensorType::BF16) {
                bail!(
                    "matrix dtype {} cannot be copied as u16",
                    self.dtype.label()
                );
            }
            self.buffer.copy_to_host(self.rows * self.cols)
        }

        pub fn copy_to_host_f32(&self) -> Result<Vec<f32>> {
            if !matches!(self.dtype, GgufTensorType::F32) {
                bail!(
                    "matrix dtype {} cannot be copied as f32",
                    self.dtype.label()
                );
            }
            self.buffer.copy_to_host(self.rows * self.cols)
        }

        fn is_quantized(&self) -> bool {
            self.dtype.is_quantized()
        }

        fn quant_type_id(&self) -> Result<i32> {
            match self.dtype {
                GgufTensorType::MXFP4 => Ok(39),
                GgufTensorType::NVFP4 => Ok(40),
                GgufTensorType::Q4_0 => Ok(2),
                GgufTensorType::Q4_0_4_4 => Ok(31),
                GgufTensorType::Q4_0_4_8 => Ok(32),
                GgufTensorType::Q4_0_8_8 => Ok(33),
                GgufTensorType::Q4_1 => Ok(3),
                GgufTensorType::Q1_0 => Ok(41),
                GgufTensorType::Q5_0 => Ok(6),
                GgufTensorType::Q5_1 => Ok(7),
                GgufTensorType::Q8_0 => Ok(8),
                GgufTensorType::Q8_1 => Ok(9),
                GgufTensorType::IQ2_XXS => Ok(16),
                GgufTensorType::IQ2_XS => Ok(17),
                GgufTensorType::IQ3_XXS => Ok(18),
                GgufTensorType::IQ1_S => Ok(19),
                GgufTensorType::IQ2_S => Ok(22),
                GgufTensorType::IQ3_S => Ok(21),
                GgufTensorType::IQ4_NL => Ok(20),
                GgufTensorType::IQ4_NL_4_4 => Ok(36),
                GgufTensorType::IQ4_NL_4_8 => Ok(37),
                GgufTensorType::IQ4_NL_8_8 => Ok(38),
                GgufTensorType::IQ4_XS => Ok(23),
                GgufTensorType::IQ1_M => Ok(29),
                GgufTensorType::Q2_K => Ok(10),
                GgufTensorType::Q3_K => Ok(11),
                GgufTensorType::Q4_K => Ok(12),
                GgufTensorType::Q5_K => Ok(13),
                GgufTensorType::Q6_K => Ok(14),
                GgufTensorType::Q8_K => Ok(15),
                GgufTensorType::TQ1_0 => Ok(34),
                GgufTensorType::TQ2_0 => Ok(35),
                other => bail!("matrix dtype {} is not quantized", other.label()),
            }
        }

        /// Convert a quantized weight matrix into a resident FP16 copy, so the
        /// decode path skips the per-token dequant-to-f32 and runs the existing
        /// FP16 GEMM directly. A no-op for already-float matrices. Trades ~2
        /// bytes/param of VRAM for a large decode speedup; opt-in at load time.
        fn into_f16(self, stream: &Stream) -> Result<Self> {
            if !self.is_quantized() {
                return Ok(self);
            }
            let elements = self
                .rows
                .checked_mul(self.cols)
                .context("CUDA f16 weight element count overflows usize")?;
            let quant_type = self.quant_type_id()?;
            // Dequantize into an f32 scratch, then narrow to f16 in place. Both
            // kernels preserve the [rows, cols] row-major layout the FP16 GEMM
            // expects, so the converted matrix is a drop-in for a native-f16 one.
            let f32_scratch = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .context("allocating f32 scratch for f16 weight conversion")?;
            crate::kernels::launch_dequantize_matrix(
                &self.buffer,
                &f32_scratch,
                elements,
                quant_type,
                stream,
            )?;
            let f16_bytes = elements
                .checked_mul(std::mem::size_of::<u16>())
                .context("CUDA f16 weight byte count overflows usize")?;
            let f16_buffer =
                DeviceBuffer::alloc(f16_bytes).context("allocating f16 weight buffer")?;
            crate::kernels::launch_cast_f32_to_f16(&f32_scratch, &f16_buffer, elements, stream)?;
            // Finish the conversion before f32_scratch is freed on drop.
            stream.synchronize()?;
            Ok(Self {
                rows: self.rows,
                cols: self.cols,
                dtype: GgufTensorType::F16,
                bytes: f16_bytes,
                buffer: f16_buffer,
            })
        }

        fn gemm_dtype(&self) -> Result<GemmDType> {
            match self.dtype {
                GgufTensorType::F16 => Ok(GemmDType::F16),
                GgufTensorType::BF16 => Ok(GemmDType::BF16),
                GgufTensorType::F32 => Ok(GemmDType::F32),
                GgufTensorType::I8
                | GgufTensorType::I16
                | GgufTensorType::I32
                | GgufTensorType::I64
                | GgufTensorType::F64 => bail!(
                    "{} matrix dtype is not supported for CUDA projection",
                    self.dtype.label()
                ),
                GgufTensorType::MXFP4
                | GgufTensorType::NVFP4
                | GgufTensorType::Q1_0
                | GgufTensorType::Q4_0
                | GgufTensorType::Q4_0_4_4
                | GgufTensorType::Q4_0_4_8
                | GgufTensorType::Q4_0_8_8
                | GgufTensorType::Q4_1
                | GgufTensorType::Q5_0
                | GgufTensorType::Q5_1
                | GgufTensorType::Q8_0
                | GgufTensorType::Q8_1
                | GgufTensorType::IQ2_XXS
                | GgufTensorType::IQ2_XS
                | GgufTensorType::IQ3_XXS
                | GgufTensorType::IQ1_S
                | GgufTensorType::IQ2_S
                | GgufTensorType::IQ3_S
                | GgufTensorType::IQ4_NL
                | GgufTensorType::IQ4_NL_4_4
                | GgufTensorType::IQ4_NL_4_8
                | GgufTensorType::IQ4_NL_8_8
                | GgufTensorType::IQ4_XS
                | GgufTensorType::IQ1_M
                | GgufTensorType::Q2_K
                | GgufTensorType::Q3_K
                | GgufTensorType::Q4_K
                | GgufTensorType::Q5_K
                | GgufTensorType::Q6_K
                | GgufTensorType::Q8_K
                | GgufTensorType::TQ1_0
                | GgufTensorType::TQ2_0 => bail!(
                    "quantized matrix dtype {} does not have a dense GEMM dtype",
                    self.dtype.label()
                ),
            }
        }
    }

    impl fmt::Debug for GpuMatrix {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("GpuMatrix")
                .field("rows", &self.rows)
                .field("cols", &self.cols)
                .field("dtype", &self.dtype)
                .field("bytes", &self.bytes)
                .finish_non_exhaustive()
        }
    }

    pub struct GpuVector {
        pub len: usize,
        pub source_dtype: GgufTensorType,
        pub bytes: usize,
        buffer: DeviceBuffer,
    }

    impl GpuVector {
        fn load(gguf: &GgufFile, info: &hi_gguf::TensorInfo) -> Result<Self> {
            let view = gguf.tensor_view(info)?;
            let len = usize::try_from(
                *info
                    .dimensions
                    .first()
                    .ok_or_else(|| anyhow!("vector tensor {} has no dimensions", info.name))?,
            )
            .context("vector length does not fit usize")?;
            let values = read_tensor_as_f32(view.bytes, info.dtype, len)
                .with_context(|| format!("reading vector {} as f32", info.name))?;
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .context("CUDA vector byte length overflows usize")?;
            let buffer = DeviceBuffer::alloc(bytes)
                .with_context(|| format!("allocating CUDA vector {}", info.name))?;
            buffer
                .copy_from_host(&values)
                .with_context(|| format!("copying vector {} to CUDA device", info.name))?;
            Ok(Self {
                len,
                source_dtype: info.dtype,
                bytes,
                buffer,
            })
        }

        fn load_slice(
            gguf: &GgufFile,
            info: &hi_gguf::TensorInfo,
            offset: usize,
            len: usize,
            source_len: usize,
        ) -> Result<Self> {
            let view = gguf.tensor_view(info)?;
            let actual_len = usize::try_from(
                *info
                    .dimensions
                    .first()
                    .ok_or_else(|| anyhow!("vector tensor {} has no dimensions", info.name))?,
            )
            .context("vector length does not fit usize")?;
            if actual_len != source_len {
                bail!(
                    "packed vector tensor {} has length {actual_len}; expected {source_len}",
                    info.name
                );
            }
            let end = offset
                .checked_add(len)
                .context("packed vector slice end overflows usize")?;
            let values = read_tensor_as_f32(view.bytes, info.dtype, source_len)
                .with_context(|| format!("reading packed vector {} as f32", info.name))?;
            let values = values.get(offset..end).ok_or_else(|| {
                anyhow!(
                    "packed vector {} slice {offset}..{end} is out of range",
                    info.name
                )
            })?;
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .context("CUDA vector byte length overflows usize")?;
            let buffer = DeviceBuffer::alloc(bytes)
                .with_context(|| format!("allocating CUDA vector slice {}", info.name))?;
            buffer
                .copy_from_host(values)
                .with_context(|| format!("copying vector slice {} to CUDA device", info.name))?;
            Ok(Self {
                len,
                source_dtype: info.dtype,
                bytes,
                buffer,
            })
        }

        fn load_from_spec(
            gguf: &GgufFile,
            info: &hi_gguf::TensorInfo,
            spec: &VectorSpec,
        ) -> Result<Self> {
            if let Some(expert) = spec.expert_index {
                return Self::load_expert_slice(
                    gguf,
                    info,
                    expert,
                    spec.expert_count.ok_or_else(|| {
                        anyhow!("expert vector spec {} is missing expert count", spec.name)
                    })?,
                    spec.offset,
                    spec.len,
                    spec.source_len,
                );
            }
            Self::load_slice(gguf, info, spec.offset, spec.len, spec.source_len)
        }

        fn load_expert_slice(
            gguf: &GgufFile,
            info: &hi_gguf::TensorInfo,
            expert: usize,
            experts: usize,
            offset: usize,
            len: usize,
            source_len: usize,
        ) -> Result<Self> {
            if expert >= experts {
                bail!("expert index {expert} is outside expert count {experts}");
            }
            let dims = info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("vector dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if !expert_vector_dims_match(&dims, source_len, experts) {
                bail!(
                    "expert vector tensor {} has shape {:?}; expected [{source_len}, {experts}] or [{experts}, {source_len}]",
                    info.name,
                    dims
                );
            }
            let end = offset
                .checked_add(len)
                .context("expert vector slice end overflows usize")?;
            if end > source_len {
                bail!(
                    "expert vector {} slice {offset}..{end} exceeds source length {source_len}",
                    info.name
                );
            }
            let element_count = source_len
                .checked_mul(experts)
                .context("expert vector element count overflows usize")?;
            let view = gguf.tensor_view(info)?;
            let source = read_tensor_as_f32(view.bytes, info.dtype, element_count)
                .with_context(|| format!("reading expert vector {} as f32", info.name))?;
            let mut values = vec![0.0; len];
            match dims.as_slice() {
                [dim0, dim1] if *dim0 == source_len && *dim1 == experts => {
                    let start = expert
                        .checked_mul(source_len)
                        .context("expert vector offset overflows usize")?;
                    values.copy_from_slice(&source[start + offset..start + end]);
                }
                [dim0, dim1] if *dim0 == experts && *dim1 == source_len => {
                    for idx in 0..len {
                        values[idx] = source[expert + experts * (offset + idx)];
                    }
                }
                _ => unreachable!("expert vector shape was validated above"),
            }
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .context("CUDA expert vector byte length overflows usize")?;
            let buffer = DeviceBuffer::alloc(bytes)
                .with_context(|| format!("allocating CUDA expert vector slice {}", info.name))?;
            buffer.copy_from_host(&values).with_context(|| {
                format!("copying expert vector slice {} to CUDA device", info.name)
            })?;
            Ok(Self {
                len,
                source_dtype: info.dtype,
                bytes,
                buffer,
            })
        }

        pub fn copy_to_host_f32(&self) -> Result<Vec<f32>> {
            self.buffer.copy_to_host(self.len)
        }
    }

    impl fmt::Debug for GpuVector {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("GpuVector")
                .field("len", &self.len)
                .field("source_dtype", &self.source_dtype)
                .field("bytes", &self.bytes)
                .finish_non_exhaustive()
        }
    }

    struct GpuF32Tensor {
        rows: usize,
        cols: usize,
        buffer: DeviceBuffer,
    }

    struct MropePositionsDevice {
        rows: usize,
        t: DeviceBuffer,
        h: DeviceBuffer,
        w: DeviceBuffer,
        host: Vec<[u32; 3]>,
    }

    impl GpuF32Tensor {
        fn element_count(&self) -> Result<usize> {
            self.rows
                .checked_mul(self.cols)
                .ok_or_else(|| anyhow!("CUDA f32 tensor element count overflows usize"))
        }

        fn ensure_same_shape(&self, other: &Self, operation: &str) -> Result<()> {
            if self.rows != other.rows || self.cols != other.cols {
                bail!(
                    "{operation} shape mismatch: left {}x{}, right {}x{}",
                    self.rows,
                    self.cols,
                    other.rows,
                    other.cols
                );
            }
            Ok(())
        }

        fn copy_to_host(&self) -> Result<Vec<f32>> {
            self.buffer.copy_to_host(self.rows * self.cols)
        }
    }

    struct QwenDims {
        context: usize,
        embed: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
    }

    #[derive(Clone, Copy)]
    enum QwenRopeLayout<'a> {
        Single {
            seq_len: usize,
            position_offset: usize,
        },
        Batched {
            batch_count: usize,
            seq_len: usize,
            position_offset: usize,
        },
        BatchedPositions {
            positions: &'a DeviceBuffer,
            host_positions: &'a [usize],
            batch_count: usize,
            seq_len: usize,
        },
        Mrope {
            positions: &'a MropePositionsDevice,
            sections: [usize; 4],
        },
    }

    impl<'a> QwenRopeLayout<'a> {
        fn rows(self) -> usize {
            match self {
                Self::Single { seq_len, .. } => seq_len,
                Self::Batched {
                    batch_count,
                    seq_len,
                    ..
                } => batch_count * seq_len,
                Self::BatchedPositions {
                    batch_count,
                    seq_len,
                    ..
                } => batch_count * seq_len,
                Self::Mrope { positions, .. } => positions.rows,
            }
        }

        fn position_for_row(self, row: usize) -> usize {
            match self {
                Self::Single {
                    position_offset, ..
                } => position_offset + row,
                Self::Batched {
                    seq_len,
                    position_offset,
                    ..
                } => position_offset + row % seq_len,
                Self::BatchedPositions {
                    host_positions,
                    seq_len,
                    ..
                } => host_positions[row / seq_len] + row % seq_len,
                Self::Mrope { positions, .. } => positions.host[row][0] as usize,
            }
        }
    }

    #[derive(Clone, Copy)]
    struct QwenMlaDims {
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        qk_head_dim: usize,
        v_head_dim: usize,
    }

    #[derive(Clone, Copy)]
    struct QwenSsmDims {
        conv_kernel: usize,
        state_size: usize,
        group_count: usize,
        time_step_rank: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: usize,
        qkvz_dim: usize,
        ba_dim: usize,
        head_v_dim: usize,
    }

    fn qwen_mla_dims(config: &QwenGgufConfig) -> Result<Option<QwenMlaDims>> {
        if !config.attention_mla_tensor_layout {
            return Ok(None);
        }
        let q_lora_rank = config
            .attention_q_lora_rank
            .map(usize::try_from)
            .transpose()
            .context("qwen attention.q_lora_rank does not fit usize")?
            .ok_or_else(|| anyhow!("MLA tensor layout requires attention.q_lora_rank"))?;
        let kv_lora_rank = config
            .attention_kv_lora_rank
            .map(usize::try_from)
            .transpose()
            .context("qwen attention.kv_lora_rank does not fit usize")?
            .ok_or_else(|| anyhow!("MLA tensor layout requires attention.kv_lora_rank"))?;
        let qk_nope_head_dim = config
            .attention_qk_nope_head_dim
            .map(usize::try_from)
            .transpose()
            .context("qwen attention.qk_nope_head_dim does not fit usize")?
            .ok_or_else(|| anyhow!("MLA tensor layout requires attention.qk_nope_head_dim"))?;
        let qk_rope_head_dim = config
            .attention_qk_rope_head_dim
            .map(usize::try_from)
            .transpose()
            .context("qwen attention.qk_rope_head_dim does not fit usize")?
            .ok_or_else(|| anyhow!("MLA tensor layout requires attention.qk_rope_head_dim"))?;
        let qk_head_dim = qk_nope_head_dim
            .checked_add(qk_rope_head_dim)
            .context("MLA qk head dimension overflows usize")?;
        let v_head_dim = config
            .attention_value_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention value head dimension does not fit usize")?
            .ok_or_else(|| anyhow!("MLA tensor layout requires a value head dimension"))?;
        if q_lora_rank == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0 || v_head_dim == 0 {
            bail!(
                "MLA tensor layout requires non-zero q_lora_rank, kv_lora_rank, qk_rope_head_dim, and value head dimension"
            );
        }
        Ok(Some(QwenMlaDims {
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
        }))
    }

    fn qwen_ssm_dims(config: &QwenGgufConfig) -> Result<Option<QwenSsmDims>> {
        if !config.recurrent_ssm_tensor_layout {
            return Ok(None);
        }
        let conv_kernel = config
            .ssm_conv_kernel
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.conv_kernel does not fit usize")?
            .ok_or_else(|| anyhow!("SSM tensor layout requires ssm.conv_kernel"))?;
        let inner_size = config
            .ssm_inner_size
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.inner_size does not fit usize")?
            .ok_or_else(|| anyhow!("SSM tensor layout requires ssm.inner_size"))?;
        let state_size = config
            .ssm_state_size
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.state_size does not fit usize")?
            .ok_or_else(|| anyhow!("SSM tensor layout requires ssm.state_size"))?;
        let time_step_rank = config
            .ssm_time_step_rank
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.time_step_rank does not fit usize")?
            .ok_or_else(|| anyhow!("SSM tensor layout requires ssm.time_step_rank"))?;
        let group_count = config
            .ssm_group_count
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.group_count does not fit usize")?
            .ok_or_else(|| anyhow!("SSM tensor layout requires ssm.group_count"))?;
        if conv_kernel == 0
            || inner_size == 0
            || state_size == 0
            || time_step_rank == 0
            || group_count == 0
        {
            bail!("SSM tensor layout requires non-zero SSM metadata values");
        }
        if time_step_rank % group_count != 0 {
            bail!(
                "SSM tensor layout requires time_step_rank {time_step_rank} to be divisible by group_count {group_count}"
            );
        }
        if inner_size % time_step_rank != 0 {
            bail!(
                "SSM tensor layout requires inner_size {inner_size} to be divisible by time_step_rank {time_step_rank}"
            );
        }
        let head_v_dim = inner_size / time_step_rank;
        let key_dim = state_size
            .checked_mul(group_count)
            .context("SSM key dimension overflows usize")?;
        let value_dim = head_v_dim
            .checked_mul(time_step_rank)
            .context("SSM value dimension overflows usize")?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_dim))
            .context("SSM convolution dimension overflows usize")?;
        let qkvz_dim = key_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_dim.checked_mul(2)?))
            .context("SSM qkvz dimension overflows usize")?;
        let ba_dim = time_step_rank
            .checked_mul(2)
            .context("SSM beta/alpha dimension overflows usize")?;
        Ok(Some(QwenSsmDims {
            conv_kernel,
            state_size,
            group_count,
            time_step_rank,
            key_dim,
            value_dim,
            conv_dim,
            qkvz_dim,
            ba_dim,
            head_v_dim,
        }))
    }

    fn split_qwen_ssm_qkvz_host(
        qkvz: &[f32],
        rows: usize,
        ssm: &QwenSsmDims,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let expected = rows
            .checked_mul(ssm.qkvz_dim)
            .context("CUDA recurrent SSM qkvz element count overflows usize")?;
        if qkvz.len() != expected {
            bail!(
                "CUDA recurrent SSM qkvz host buffer has {} values; expected {rows} x {} = {expected}",
                qkvz.len(),
                ssm.qkvz_dim
            );
        }
        let repeat = ssm.time_step_rank / ssm.group_count;
        let value_group_dim = repeat
            .checked_mul(ssm.head_v_dim)
            .context("CUDA recurrent SSM grouped value dimension overflows usize")?;
        let source_group_dim = ssm
            .state_size
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_group_dim.checked_mul(2)?))
            .context("CUDA recurrent SSM grouped qkvz dimension overflows usize")?;
        let mut mixed_qkv = vec![0.0; rows * ssm.conv_dim];
        let mut z = vec![0.0; rows * ssm.value_dim];
        for row in 0..rows {
            let source_row = row
                .checked_mul(ssm.qkvz_dim)
                .context("CUDA recurrent SSM source row offset overflows usize")?;
            let q_dest_row = row
                .checked_mul(ssm.conv_dim)
                .context("CUDA recurrent SSM q destination row overflows usize")?;
            let k_dest_row = q_dest_row + ssm.key_dim;
            let value_dest_row = q_dest_row + 2 * ssm.key_dim;
            let z_dest_row = row
                .checked_mul(ssm.value_dim)
                .context("CUDA recurrent SSM z destination row overflows usize")?;
            for group in 0..ssm.group_count {
                let mut source = source_row + group * source_group_dim;
                let q_dest = q_dest_row + group * ssm.state_size;
                mixed_qkv[q_dest..q_dest + ssm.state_size]
                    .copy_from_slice(&qkvz[source..source + ssm.state_size]);
                source += ssm.state_size;
                let k_dest = k_dest_row + group * ssm.state_size;
                mixed_qkv[k_dest..k_dest + ssm.state_size]
                    .copy_from_slice(&qkvz[source..source + ssm.state_size]);
                source += ssm.state_size;
                let value_dest = value_dest_row + group * value_group_dim;
                mixed_qkv[value_dest..value_dest + value_group_dim]
                    .copy_from_slice(&qkvz[source..source + value_group_dim]);
                source += value_group_dim;
                let z_dest = z_dest_row + group * value_group_dim;
                z[z_dest..z_dest + value_group_dim]
                    .copy_from_slice(&qkvz[source..source + value_group_dim]);
            }
        }
        Ok((mixed_qkv, z))
    }

    fn qwen_ssm_depthwise_conv_host(
        mixed_qkv: &[f32],
        conv_weight: &[f32],
        rows: usize,
        ssm: &QwenSsmDims,
        weight_name: &str,
    ) -> Result<Vec<f32>> {
        let expected_input = rows
            .checked_mul(ssm.conv_dim)
            .context("CUDA recurrent SSM conv input element count overflows usize")?;
        if mixed_qkv.len() != expected_input {
            bail!(
                "CUDA recurrent SSM conv input has {} values; expected {rows} x {} = {expected_input}",
                mixed_qkv.len(),
                ssm.conv_dim
            );
        }
        let expected_weight = ssm
            .conv_dim
            .checked_mul(ssm.conv_kernel)
            .context("CUDA recurrent SSM conv weight element count overflows usize")?;
        if conv_weight.len() != expected_weight {
            bail!(
                "CUDA recurrent SSM conv weight {weight_name} has {} values; expected {} x {} = {expected_weight}",
                conv_weight.len(),
                ssm.conv_dim,
                ssm.conv_kernel
            );
        }
        let mut output = vec![0.0; expected_input];
        for row in 0..rows {
            for channel in 0..ssm.conv_dim {
                let mut sum = 0.0f32;
                for kernel in 0..ssm.conv_kernel {
                    let input_row = row as isize + kernel as isize + 1 - ssm.conv_kernel as isize;
                    if input_row < 0 {
                        continue;
                    }
                    let input_row = usize::try_from(input_row)
                        .context("CUDA recurrent SSM conv input row does not fit usize")?;
                    sum += conv_weight[channel * ssm.conv_kernel + kernel]
                        * mixed_qkv[input_row * ssm.conv_dim + channel];
                }
                output[row * ssm.conv_dim + channel] = silu_scalar(sum);
            }
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen_ssm_gated_delta_host(
        conv: &[f32],
        mixed_ba: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        rows: usize,
        ssm: &QwenSsmDims,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let expected_conv = rows
            .checked_mul(ssm.conv_dim)
            .context("CUDA recurrent SSM delta input element count overflows usize")?;
        if conv.len() != expected_conv {
            bail!(
                "CUDA recurrent SSM delta input for {prefix} has {} values; expected {rows} x {} = {expected_conv}",
                conv.len(),
                ssm.conv_dim
            );
        }
        let expected_ba = rows
            .checked_mul(ssm.ba_dim)
            .context("CUDA recurrent SSM ba element count overflows usize")?;
        if mixed_ba.len() != expected_ba {
            bail!(
                "CUDA recurrent SSM ba input for {prefix} has {} values; expected {rows} x {} = {expected_ba}",
                mixed_ba.len(),
                ssm.ba_dim
            );
        }
        if dt_bias.len() != ssm.time_step_rank {
            bail!(
                "CUDA recurrent SSM {prefix}.ssm_dt.bias has length {}; expected {}",
                dt_bias.len(),
                ssm.time_step_rank
            );
        }
        if a_log.len() != ssm.time_step_rank {
            bail!(
                "CUDA recurrent SSM {prefix}.ssm_a has length {}; expected {}",
                a_log.len(),
                ssm.time_step_rank
            );
        }

        let mut query = vec![0.0; rows * ssm.key_dim];
        let mut key = vec![0.0; rows * ssm.key_dim];
        let mut value = vec![0.0; rows * ssm.value_dim];
        for row in 0..rows {
            let source = row * ssm.conv_dim;
            query[row * ssm.key_dim..(row + 1) * ssm.key_dim]
                .copy_from_slice(&conv[source..source + ssm.key_dim]);
            key[row * ssm.key_dim..(row + 1) * ssm.key_dim]
                .copy_from_slice(&conv[source + ssm.key_dim..source + 2 * ssm.key_dim]);
            value[row * ssm.value_dim..(row + 1) * ssm.value_dim]
                .copy_from_slice(&conv[source + 2 * ssm.key_dim..source + ssm.conv_dim]);
        }
        for row in 0..rows {
            for group in 0..ssm.group_count {
                let start = row * ssm.key_dim + group * ssm.state_size;
                l2_normalize_host(&mut query[start..start + ssm.state_size]);
                l2_normalize_host(&mut key[start..start + ssm.state_size]);
            }
        }

        let repeat = ssm.time_step_rank / ssm.group_count;
        let group_ba_dim = repeat
            .checked_mul(2)
            .context("CUDA recurrent SSM grouped ba dimension overflows usize")?;
        let q_scale = 1.0f32 / (ssm.state_size as f32).sqrt();
        let mut state = vec![0.0; ssm.time_step_rank * ssm.state_size * ssm.head_v_dim];
        let mut output = vec![0.0; rows * ssm.value_dim];
        let mut kv_mem = vec![0.0; ssm.head_v_dim];
        let mut delta = vec![0.0; ssm.head_v_dim];
        for row in 0..rows {
            for head in 0..ssm.time_step_rank {
                let group = head / repeat;
                let local_head = head % repeat;
                let q_start = row * ssm.key_dim + group * ssm.state_size;
                let k_start = row * ssm.key_dim + group * ssm.state_size;
                let v_start = row * ssm.value_dim + head * ssm.head_v_dim;
                let ba_group = row * ssm.ba_dim + group * group_ba_dim;
                let beta = sigmoid(mixed_ba[ba_group + local_head]);
                let alpha = mixed_ba[ba_group + repeat + local_head];
                let decay = (-a_log[head].exp() * softplus(alpha + dt_bias[head])).exp();
                let state_start = head * ssm.state_size * ssm.head_v_dim;

                for state_dim in 0..ssm.state_size {
                    for value_dim in 0..ssm.head_v_dim {
                        state[state_start + state_dim * ssm.head_v_dim + value_dim] *= decay;
                    }
                }
                kv_mem.fill(0.0);
                for state_dim in 0..ssm.state_size {
                    let key_value = key[k_start + state_dim];
                    for value_dim in 0..ssm.head_v_dim {
                        kv_mem[value_dim] +=
                            state[state_start + state_dim * ssm.head_v_dim + value_dim] * key_value;
                    }
                }
                for value_dim in 0..ssm.head_v_dim {
                    delta[value_dim] = (value[v_start + value_dim] - kv_mem[value_dim]) * beta;
                }
                for state_dim in 0..ssm.state_size {
                    let key_value = key[k_start + state_dim];
                    for value_dim in 0..ssm.head_v_dim {
                        state[state_start + state_dim * ssm.head_v_dim + value_dim] +=
                            key_value * delta[value_dim];
                    }
                }
                for value_dim in 0..ssm.head_v_dim {
                    let mut sum = 0.0f32;
                    for state_dim in 0..ssm.state_size {
                        sum += state[state_start + state_dim * ssm.head_v_dim + value_dim]
                            * query[q_start + state_dim]
                            * q_scale;
                    }
                    output[v_start + value_dim] = sum;
                }
            }
        }
        Ok(output)
    }

    fn qwen_ssm_gated_rms_norm_host(
        core: &[f32],
        z: &[f32],
        norm_weight: &[f32],
        rows: usize,
        ssm: &QwenSsmDims,
        eps: f32,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let expected = rows
            .checked_mul(ssm.value_dim)
            .context("CUDA recurrent SSM norm element count overflows usize")?;
        if core.len() != expected || z.len() != expected {
            bail!(
                "CUDA recurrent SSM norm for {prefix} got core/z lengths {}/{}; expected {expected}",
                core.len(),
                z.len()
            );
        }
        if norm_weight.len() != ssm.head_v_dim {
            bail!(
                "CUDA recurrent SSM {prefix}.ssm_norm.weight has length {}; expected {}",
                norm_weight.len(),
                ssm.head_v_dim
            );
        }
        let mut output = vec![0.0; expected];
        for row in 0..rows {
            for head in 0..ssm.time_step_rank {
                let start = row * ssm.value_dim + head * ssm.head_v_dim;
                let mut variance = 0.0f32;
                for value_dim in 0..ssm.head_v_dim {
                    let value = core[start + value_dim];
                    variance += value * value;
                }
                let scale = (variance / ssm.head_v_dim as f32 + eps).sqrt().recip();
                for value_dim in 0..ssm.head_v_dim {
                    output[start + value_dim] = core[start + value_dim]
                        * scale
                        * norm_weight[value_dim]
                        * silu_scalar(z[start + value_dim]);
                }
            }
        }
        Ok(output)
    }

    fn l2_normalize_host(values: &mut [f32]) {
        let norm_sq = values.iter().map(|value| value * value).sum::<f32>();
        let inv_norm = (norm_sq + 1.0e-6).sqrt().recip();
        for value in values {
            *value *= inv_norm;
        }
    }

    fn apply_rope_host(
        values: &mut [f32],
        position: usize,
        base: f32,
        scale: f32,
        split_half: bool,
    ) -> Result<()> {
        if values.len() % 2 != 0 {
            bail!(
                "CUDA host RoPE head dimension {} must be even",
                values.len()
            );
        }
        let half = values.len() / 2;
        let position = position as f32 * scale;
        if split_half {
            for idx in 0..half {
                let freq = base.powf(-(idx as f32 * 2.0) / values.len() as f32);
                let angle = position * freq;
                let (sin, cos) = angle.sin_cos();
                let left = values[idx];
                let right = values[idx + half];
                values[idx] = left * cos - right * sin;
                values[idx + half] = right * cos + left * sin;
            }
        } else {
            for idx in (0..values.len()).step_by(2) {
                let pair = idx / 2;
                let freq = base.powf(-(pair as f32 * 2.0) / values.len() as f32);
                let angle = position * freq;
                let (sin, cos) = angle.sin_cos();
                let left = values[idx];
                let right = values[idx + 1];
                values[idx] = left * cos - right * sin;
                values[idx + 1] = right * cos + left * sin;
            }
        }
        Ok(())
    }

    fn apply_rope_layout_host(
        values: &mut [f32],
        row: usize,
        layout: QwenRopeLayout<'_>,
        base: f32,
        scale: f32,
        split_half: bool,
    ) -> Result<()> {
        match layout {
            QwenRopeLayout::Mrope {
                positions,
                sections,
            } => apply_mrope_host(
                values,
                positions
                    .host
                    .get(row)
                    .copied()
                    .ok_or_else(|| anyhow!("CUDA host MLA MRoPE row {row} is out of range"))?,
                base,
                scale,
                sections,
                split_half,
            ),
            _ => apply_rope_host(
                values,
                layout.position_for_row(row),
                base,
                scale,
                split_half,
            ),
        }
    }

    fn apply_mrope_host(
        values: &mut [f32],
        position: [u32; 3],
        base: f32,
        scale: f32,
        sections: [usize; 4],
        split_half: bool,
    ) -> Result<()> {
        if values.len() % 2 != 0 {
            bail!(
                "CUDA host MRoPE head dimension {} must be even",
                values.len()
            );
        }
        let half = values.len() / 2;
        let section_sum = sections
            .iter()
            .try_fold(0usize, |acc, value| acc.checked_add(*value))
            .context("CUDA host MRoPE section sum overflows usize")?;
        if section_sum == 0 {
            bail!("CUDA host MRoPE dimension sections must not all be zero");
        }
        if section_sum > half {
            bail!(
                "CUDA host MRoPE dimension sections sum {section_sum} exceeds rotary pair count {half}"
            );
        }

        let h_start = sections[0];
        let w_start = h_start + sections[1];
        let e_start = w_start + sections[2];
        if split_half {
            for idx in 0..half {
                let sector = idx % section_sum;
                let position = if sector >= h_start && sector < w_start {
                    position[1]
                } else if sector >= w_start && sector < e_start {
                    position[2]
                } else {
                    position[0]
                };
                let freq = base.powf(-(idx as f32 * 2.0) / values.len() as f32);
                let angle = position as f32 * scale * freq;
                let (sin, cos) = angle.sin_cos();
                let left = values[idx];
                let right = values[idx + half];
                values[idx] = left * cos - right * sin;
                values[idx + half] = right * cos + left * sin;
            }
        } else {
            for idx in (0..values.len()).step_by(2) {
                let pair = idx / 2;
                let sector = pair % section_sum;
                let position = if sector >= h_start && sector < w_start {
                    position[1]
                } else if sector >= w_start && sector < e_start {
                    position[2]
                } else {
                    position[0]
                };
                let freq = base.powf(-(pair as f32 * 2.0) / values.len() as f32);
                let angle = position as f32 * scale * freq;
                let (sin, cos) = angle.sin_cos();
                let left = values[idx];
                let right = values[idx + 1];
                values[idx] = left * cos - right * sin;
                values[idx + 1] = right * cos + left * sin;
            }
        }
        Ok(())
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn silu_scalar(value: f32) -> f32 {
        value * sigmoid(value)
    }

    fn softplus(value: f32) -> f32 {
        if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            (1.0 + value.exp()).ln()
        }
    }

    struct MatrixSpec {
        name: String,
        tensor_name: String,
        rows: usize,
        cols: usize,
        expert_index: Option<usize>,
        row_slice: Option<RowSlice>,
    }

    struct VectorSpec {
        name: String,
        tensor_name: String,
        offset: usize,
        len: usize,
        source_len: usize,
        expert_index: Option<usize>,
        expert_count: Option<usize>,
    }

    #[derive(Clone, Copy)]
    struct RowSlice {
        offset: usize,
        source_rows: usize,
    }

    impl MatrixSpec {
        fn dense(name: impl Into<String>, rows: usize, cols: usize) -> Self {
            let name = name.into();
            Self {
                tensor_name: name.clone(),
                name,
                rows,
                cols,
                expert_index: None,
                row_slice: None,
            }
        }

        fn expert_alias(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            expert: usize,
            rows: usize,
            cols: usize,
        ) -> Self {
            Self {
                name: name.into(),
                tensor_name: tensor_name.into(),
                rows,
                cols,
                expert_index: Some(expert),
                row_slice: None,
            }
        }

        fn alias(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            rows: usize,
            cols: usize,
        ) -> Self {
            Self {
                name: name.into(),
                tensor_name: tensor_name.into(),
                rows,
                cols,
                expert_index: None,
                row_slice: None,
            }
        }

        fn row_slice(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            offset: usize,
            source_rows: usize,
            rows: usize,
            cols: usize,
        ) -> Self {
            Self {
                name: name.into(),
                tensor_name: tensor_name.into(),
                rows,
                cols,
                expert_index: None,
                row_slice: Some(RowSlice {
                    offset,
                    source_rows,
                }),
            }
        }

        fn expert_row_slice(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            expert: usize,
            offset: usize,
            source_rows: usize,
            rows: usize,
            cols: usize,
        ) -> Self {
            Self {
                name: name.into(),
                tensor_name: tensor_name.into(),
                rows,
                cols,
                expert_index: Some(expert),
                row_slice: Some(RowSlice {
                    offset,
                    source_rows,
                }),
            }
        }
    }

    impl VectorSpec {
        fn slice(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            offset: usize,
            len: usize,
            source_len: usize,
        ) -> Self {
            Self {
                name: name.into(),
                tensor_name: tensor_name.into(),
                offset,
                len,
                source_len,
                expert_index: None,
                expert_count: None,
            }
        }

        fn alias(name: impl Into<String>, tensor_name: impl Into<String>, len: usize) -> Self {
            Self::slice(name, tensor_name, 0, len, len)
        }

        fn expert_slice(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            expert_index: usize,
            expert_count: usize,
            offset: usize,
            len: usize,
            source_len: usize,
        ) -> Self {
            Self {
                name: name.into(),
                tensor_name: tensor_name.into(),
                offset,
                len,
                source_len,
                expert_index: Some(expert_index),
                expert_count: Some(expert_count),
            }
        }

        fn expert_alias(
            name: impl Into<String>,
            tensor_name: impl Into<String>,
            expert_index: usize,
            expert_count: usize,
            len: usize,
        ) -> Self {
            Self::expert_slice(name, tensor_name, expert_index, expert_count, 0, len, len)
        }
    }

    fn qwen_matrix_specs(gguf: &GgufFile, config: &QwenGgufConfig) -> Result<Vec<MatrixSpec>> {
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let vocab = config
            .vocab_size
            .map(usize::try_from)
            .transpose()
            .context("qwen vocab size does not fit usize")?
            .unwrap_or_else(|| {
                gguf.tokenizer()
                    .map(|tokenizer| tokenizer.token_count())
                    .unwrap_or(0)
            });
        if vocab == 0 {
            bail!("qwen vocab size could not be determined from metadata or tokenizer");
        }
        let heads = usize::try_from(config.attention_head_count)
            .context("qwen attention_head_count does not fit usize")?;
        let kv_heads = usize::try_from(config.attention_head_count_kv)
            .context("qwen attention_head_count_kv does not fit usize")?;
        if heads == 0 || kv_heads == 0 {
            bail!("invalid qwen head metadata for CUDA matrix specs");
        }
        let head_dim = config
            .attention_key_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention key head dimension does not fit usize")?
            .ok_or_else(|| anyhow!("invalid qwen head metadata for CUDA matrix specs"))?;
        let v_head_dim = config
            .attention_value_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention value head dimension does not fit usize")?
            .ok_or_else(|| anyhow!("invalid qwen value head metadata for CUDA matrix specs"))?;
        let q_dim = head_dim
            .checked_mul(heads)
            .context("qwen q dimension overflows usize")?;
        let k_dim = head_dim
            .checked_mul(kv_heads)
            .context("qwen k dimension overflows usize")?;
        let v_dim = v_head_dim
            .checked_mul(kv_heads)
            .context("qwen v dimension overflows usize")?;
        let attention_output_dim = v_head_dim
            .checked_mul(heads)
            .context("qwen attention output dimension overflows usize")?;
        let dense_ff = config
            .feed_forward_length
            .map(usize::try_from)
            .transpose()
            .context("qwen feed_forward_length does not fit usize")?;

        let mut specs = vec![dense_matrix_spec_with_aliases(
            gguf,
            "token_embd.weight".to_string(),
            qwen_dense_token_embd_weight_names(),
            vocab,
            embed,
        )];
        let output_weight_names = qwen_dense_output_weight_names();
        if output_weight_names
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        {
            specs.push(dense_matrix_spec_with_aliases(
                gguf,
                "output.weight".to_string(),
                output_weight_names,
                vocab,
                embed,
            ));
        }
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            if qwen_ssm_layer_tensors_present(gguf, &prefix) {
                let ssm = qwen_ssm_dims(config)?
                    .ok_or_else(|| anyhow!("complete SSM tensor layout missing SSM metadata"))?;
                if qwen_ssm_in_weight_names(&prefix)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some())
                {
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ssm_in.weight"),
                        qwen_ssm_in_weight_names(&prefix),
                        ssm.qkvz_dim,
                        embed,
                    ));
                } else {
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_qkv.weight"),
                        qwen_ssm_qkv_weight_names(&prefix),
                        ssm.conv_dim,
                        embed,
                    ));
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_gate.weight"),
                        qwen_ssm_gate_weight_names(&prefix),
                        ssm.value_dim,
                        embed,
                    ));
                }
                specs.extend([
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ssm_conv1d.weight"),
                        qwen_ssm_conv1d_weight_names(&prefix),
                        ssm.conv_dim,
                        ssm.conv_kernel,
                    ),
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ssm_ba.weight"),
                        qwen_ssm_ba_weight_names(&prefix),
                        ssm.ba_dim,
                        embed,
                    ),
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ssm_out.weight"),
                        qwen_ssm_out_weight_names(&prefix),
                        embed,
                        ssm.value_dim,
                    ),
                ]);
            } else if qwen_mla_attention_tensors_present(gguf, &prefix) {
                let mla = qwen_mla_dims(config)?
                    .ok_or_else(|| anyhow!("complete MLA tensor layout missing MLA metadata"))?;
                let mla_q_dim = mla
                    .qk_head_dim
                    .checked_mul(heads)
                    .context("MLA q dimension overflows usize")?;
                let kv_a_dim = mla
                    .kv_lora_rank
                    .checked_add(mla.qk_rope_head_dim)
                    .context("MLA kv_a dimension overflows usize")?;
                let kv_b_dim = heads
                    .checked_mul(
                        mla.qk_nope_head_dim
                            .checked_add(mla.v_head_dim)
                            .context("MLA kv_b per-head dimension overflows usize")?,
                    )
                    .context("MLA kv_b dimension overflows usize")?;
                specs.extend([
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_q_a.weight"),
                        qwen_mla_q_a_weight_names(&prefix),
                        mla.q_lora_rank,
                        embed,
                    ),
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_q_b.weight"),
                        qwen_mla_q_b_weight_names(&prefix),
                        mla_q_dim,
                        mla.q_lora_rank,
                    ),
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_kv_a_mqa.weight"),
                        qwen_mla_kv_a_weight_names(&prefix),
                        kv_a_dim,
                        embed,
                    ),
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_kv_b.weight"),
                        qwen_mla_kv_b_weight_names(&prefix),
                        kv_b_dim,
                        mla.kv_lora_rank,
                    ),
                ]);
            } else if !qkv_split_tensors_present(gguf, &prefix)
                && let Some(source) = qwen_dense_packed_qkv_weight_names(&prefix)
                    .into_iter()
                    .find(|name| gguf.tensor(name).is_some())
            {
                let qkv_dim = q_dim
                    .checked_add(k_dim)
                    .and_then(|value| value.checked_add(v_dim))
                    .context("qwen packed qkv dimension overflows usize")?;
                specs.extend([
                    MatrixSpec::row_slice(
                        format!("{prefix}.attn_q.weight"),
                        source.clone(),
                        0,
                        qkv_dim,
                        q_dim,
                        embed,
                    ),
                    MatrixSpec::row_slice(
                        format!("{prefix}.attn_k.weight"),
                        source.clone(),
                        q_dim,
                        qkv_dim,
                        k_dim,
                        embed,
                    ),
                    MatrixSpec::row_slice(
                        format!("{prefix}.attn_v.weight"),
                        source,
                        q_dim + k_dim,
                        qkv_dim,
                        v_dim,
                        embed,
                    ),
                ]);
            } else {
                if let Some(source) = qwen_dense_gated_attention_q_weight_name(
                    gguf,
                    &prefix,
                    u64::try_from(q_dim).context("qwen q dimension does not fit u64")?,
                    u64::try_from(embed).context("qwen embedding dimension does not fit u64")?,
                ) {
                    specs.push(MatrixSpec::alias(
                        format!("{prefix}.attn_q_gated.weight"),
                        source,
                        q_dim
                            .checked_mul(2)
                            .context("qwen gated q dimension overflows usize")?,
                        embed,
                    ));
                } else {
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_q.weight"),
                        qwen_dense_attention_weight_names(&prefix, "q"),
                        q_dim,
                        embed,
                    ));
                }
                specs.extend([
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_k.weight"),
                        qwen_dense_attention_weight_names(&prefix, "k"),
                        k_dim,
                        embed,
                    ),
                    dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.attn_v.weight"),
                        qwen_dense_attention_weight_names(&prefix, "v"),
                        v_dim,
                        embed,
                    ),
                ]);
            }
            if !qwen_ssm_layer_tensors_present(gguf, &prefix) {
                specs.push(dense_matrix_spec_with_aliases(
                    gguf,
                    format!("{prefix}.attn_output.weight"),
                    qwen_dense_attention_weight_names(&prefix, "output"),
                    embed,
                    attention_output_dim,
                ));
            }
            if config.expert_count.is_some() && moe_router_tensor_present(gguf, &prefix) {
                let experts = config
                    .expert_count
                    .map(usize::try_from)
                    .transpose()
                    .context("qwen expert_count does not fit usize")?
                    .ok_or_else(|| anyhow!("qwen MoE metadata missing expert_count"))?;
                let expert_ff = expert_ff_dim(gguf, config, &prefix, experts, embed)?;
                let use_per_expert_tensors = !moe_packed_expert_tensors_complete(gguf, &prefix)
                    && moe_per_expert_tensors_complete(gguf, &prefix, experts);
                specs.push(dense_matrix_spec_with_aliases(
                    gguf,
                    format!("{prefix}.ffn_gate_inp.weight"),
                    qwen_moe_router_weight_names(&prefix),
                    experts,
                    embed,
                ));
                for expert in 0..experts {
                    if let Some(source) =
                        moe_packed_expert_gate_up_source(gguf, &prefix, expert_ff, embed, experts)?
                    {
                        specs.push(MatrixSpec::expert_row_slice(
                            format!("{prefix}.ffn_gate_exps.{expert}.weight"),
                            source.name.clone(),
                            expert,
                            source.gate_offset,
                            source.source_rows,
                            expert_ff,
                            embed,
                        ));
                        specs.push(MatrixSpec::expert_row_slice(
                            format!("{prefix}.ffn_up_exps.{expert}.weight"),
                            source.name,
                            expert,
                            source.up_offset,
                            source.source_rows,
                            expert_ff,
                            embed,
                        ));
                    } else if let Some(source) = moe_per_expert_packed_gate_up_source(
                        gguf, &prefix, expert_ff, embed, expert,
                    )? {
                        specs.push(MatrixSpec::row_slice(
                            format!("{prefix}.ffn_gate_exps.{expert}.weight"),
                            source.name.clone(),
                            source.gate_offset,
                            source.source_rows,
                            expert_ff,
                            embed,
                        ));
                        specs.push(MatrixSpec::row_slice(
                            format!("{prefix}.ffn_up_exps.{expert}.weight"),
                            source.name,
                            source.up_offset,
                            source.source_rows,
                            expert_ff,
                            embed,
                        ));
                    } else {
                        specs.push(moe_expert_matrix_spec(
                            gguf,
                            &prefix,
                            "gate",
                            qwen_moe_packed_expert_weight_names(&prefix, "gate"),
                            qwen_moe_per_expert_weight_names(&prefix, "gate", expert as u64),
                            expert,
                            expert_ff,
                            embed,
                            use_per_expert_tensors,
                        ));
                        specs.push(moe_expert_matrix_spec(
                            gguf,
                            &prefix,
                            "up",
                            qwen_moe_packed_expert_weight_names(&prefix, "up"),
                            qwen_moe_per_expert_weight_names(&prefix, "up", expert as u64),
                            expert,
                            expert_ff,
                            embed,
                            use_per_expert_tensors,
                        ));
                    }
                    specs.push(moe_expert_matrix_spec(
                        gguf,
                        &prefix,
                        "down",
                        qwen_moe_packed_expert_weight_names(&prefix, "down"),
                        qwen_moe_per_expert_weight_names(&prefix, "down", expert as u64),
                        expert,
                        embed,
                        expert_ff,
                        use_per_expert_tensors,
                    ));
                }
                if let Some(source) =
                    moe_shared_expert_packed_gate_up_source(gguf, &prefix, expert_ff, embed)?
                {
                    specs.push(MatrixSpec::row_slice(
                        format!("{prefix}.ffn_gate_shexp.weight"),
                        source.name.clone(),
                        source.gate_offset,
                        source.source_rows,
                        expert_ff,
                        embed,
                    ));
                    specs.push(MatrixSpec::row_slice(
                        format!("{prefix}.ffn_up_shexp.weight"),
                        source.name,
                        source.up_offset,
                        source.source_rows,
                        expert_ff,
                        embed,
                    ));
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ffn_down_shexp.weight"),
                        qwen_moe_shared_expert_weight_names(&prefix, "down"),
                        embed,
                        expert_ff,
                    ));
                } else if qwen_moe_shared_expert_weight_names(&prefix, "gate")
                    .iter()
                    .any(|name| gguf.tensor(name).is_some())
                {
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ffn_gate_shexp.weight"),
                        qwen_moe_shared_expert_weight_names(&prefix, "gate"),
                        expert_ff,
                        embed,
                    ));
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ffn_up_shexp.weight"),
                        qwen_moe_shared_expert_weight_names(&prefix, "up"),
                        expert_ff,
                        embed,
                    ));
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ffn_down_shexp.weight"),
                        qwen_moe_shared_expert_weight_names(&prefix, "down"),
                        embed,
                        expert_ff,
                    ));
                }
                if qwen_moe_shared_expert_gate_weight_names(&prefix)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some())
                {
                    specs.push(dense_matrix_spec_with_aliases(
                        gguf,
                        format!("{prefix}.ffn_gate_inp_shexp.weight"),
                        qwen_moe_shared_expert_gate_weight_names(&prefix),
                        1,
                        embed,
                    ));
                }
            } else {
                let ff =
                    dense_ff.ok_or_else(|| anyhow!("qwen metadata missing feed_forward_length"))?;
                if let Some(source) = dense_packed_ffn_source(gguf, &prefix, ff, embed)? {
                    specs.push(MatrixSpec::row_slice(
                        format!("{prefix}.ffn_gate.weight"),
                        source.name.clone(),
                        source.gate_offset,
                        ff.checked_mul(2)
                            .context("qwen packed ffn row count overflows usize")?,
                        ff,
                        embed,
                    ));
                    specs.push(MatrixSpec::row_slice(
                        format!("{prefix}.ffn_up.weight"),
                        source.name,
                        source.up_offset,
                        ff.checked_mul(2)
                            .context("qwen packed ffn row count overflows usize")?,
                        ff,
                        embed,
                    ));
                } else {
                    specs.extend([
                        dense_matrix_spec_with_aliases(
                            gguf,
                            format!("{prefix}.ffn_gate.weight"),
                            qwen_dense_ffn_weight_names(&prefix, "gate"),
                            ff,
                            embed,
                        ),
                        dense_matrix_spec_with_aliases(
                            gguf,
                            format!("{prefix}.ffn_up.weight"),
                            qwen_dense_ffn_weight_names(&prefix, "up"),
                            ff,
                            embed,
                        ),
                    ]);
                }
                specs.push(dense_matrix_spec_with_aliases(
                    gguf,
                    format!("{prefix}.ffn_down.weight"),
                    qwen_dense_ffn_weight_names(&prefix, "down"),
                    embed,
                    ff,
                ));
            }
        }
        Ok(specs)
    }

    fn dense_matrix_spec_with_aliases(
        gguf: &GgufFile,
        logical_name: String,
        aliases: Vec<String>,
        rows: usize,
        cols: usize,
    ) -> MatrixSpec {
        for source_name in aliases {
            if gguf.tensor(&source_name).is_some() {
                if source_name == logical_name {
                    return MatrixSpec::dense(logical_name, rows, cols);
                }
                return MatrixSpec::alias(logical_name, source_name, rows, cols);
            }
        }
        MatrixSpec::dense(logical_name, rows, cols)
    }

    fn qwen_vector_alias_specs(
        gguf: &GgufFile,
        config: &QwenGgufConfig,
    ) -> Result<Vec<VectorSpec>> {
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let vocab = config
            .vocab_size
            .map(usize::try_from)
            .transpose()
            .context("qwen vocab size does not fit usize")?
            .unwrap_or_else(|| {
                gguf.tokenizer()
                    .map(|tokenizer| tokenizer.token_count())
                    .unwrap_or(0)
            });
        if vocab == 0 {
            bail!("qwen vocab size could not be determined from metadata or tokenizer");
        }
        let heads = usize::try_from(config.attention_head_count)
            .context("qwen attention_head_count does not fit usize")?;
        let kv_heads = usize::try_from(config.attention_head_count_kv)
            .context("qwen attention_head_count_kv does not fit usize")?;
        if heads == 0 || kv_heads == 0 {
            bail!("invalid qwen head metadata for CUDA vector specs");
        }
        let head_dim = config
            .attention_key_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention key head dimension does not fit usize")?
            .ok_or_else(|| anyhow!("invalid qwen head metadata for CUDA vector specs"))?;
        let v_head_dim = config
            .attention_value_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention value head dimension does not fit usize")?
            .ok_or_else(|| anyhow!("invalid qwen value head metadata for CUDA vector specs"))?;
        let q_dim = head_dim
            .checked_mul(heads)
            .context("qwen q dimension overflows usize")?;
        let k_dim = head_dim
            .checked_mul(kv_heads)
            .context("qwen k dimension overflows usize")?;
        let v_dim = v_head_dim
            .checked_mul(kv_heads)
            .context("qwen v dimension overflows usize")?;
        let dense_ff = config
            .feed_forward_length
            .map(usize::try_from)
            .transpose()
            .context("qwen feed_forward_length does not fit usize")?;
        let mut specs = Vec::new();
        let mla_dims = qwen_mla_dims(config)?;
        let ssm_dims = qwen_ssm_dims(config)?;
        push_vector_alias_spec(
            gguf,
            &mut specs,
            "output_norm.weight",
            qwen_dense_output_norm_weight_names(),
            embed,
        );
        push_vector_alias_spec(
            gguf,
            &mut specs,
            "output.bias",
            qwen_dense_output_bias_names(),
            vocab,
        );
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            let uses_mla_attention = qwen_mla_attention_tensors_present(gguf, &prefix);
            let uses_recurrent_ssm = qwen_ssm_layer_tensors_present(gguf, &prefix);
            let uses_moe = config
                .expert_count
                .is_some_and(|_| moe_router_tensor_present(gguf, &prefix));
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.attn_norm.weight"),
                qwen_dense_attention_norm_weight_names(&prefix),
                embed,
            );
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.ffn_norm.weight"),
                qwen_dense_ffn_norm_weight_names(&prefix),
                embed,
            );
            if uses_moe {
                let experts = config
                    .expert_count
                    .map(usize::try_from)
                    .transpose()
                    .context("qwen expert_count does not fit usize")?
                    .ok_or_else(|| anyhow!("qwen MoE metadata missing expert_count"))?;
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.ffn_gate_inp.bias"),
                    qwen_moe_router_bias_names(&prefix),
                    experts,
                );
                let expert_ff = expert_ff_dim(gguf, config, &prefix, experts, embed)?;
                let use_per_expert_tensors = !moe_packed_expert_tensors_complete(gguf, &prefix)
                    && moe_per_expert_tensors_complete(gguf, &prefix, experts);
                push_moe_expert_bias_vector_specs(
                    gguf,
                    &mut specs,
                    &prefix,
                    experts,
                    expert_ff,
                    embed,
                    use_per_expert_tensors,
                )?;
                push_moe_shared_expert_bias_vector_specs(
                    gguf, &mut specs, &prefix, expert_ff, embed,
                )?;
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.ffn_gate_inp_shexp.bias"),
                    qwen_moe_shared_expert_gate_bias_names(&prefix),
                    1,
                );
            } else {
                if let Some(ff) = dense_ff {
                    push_dense_ffn_bias_vector_specs(gguf, &mut specs, &prefix, ff, embed)?;
                }
            }
            if uses_recurrent_ssm {
                let ssm = ssm_dims
                    .ok_or_else(|| anyhow!("complete SSM tensor layout missing SSM metadata"))?;
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.ssm_dt.bias"),
                    qwen_ssm_dt_bias_names(&prefix),
                    ssm.time_step_rank,
                );
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.ssm_a"),
                    qwen_ssm_a_names(&prefix),
                    ssm.time_step_rank,
                );
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.ssm_norm.weight"),
                    qwen_ssm_norm_weight_names(&prefix),
                    ssm.head_v_dim,
                );
                continue;
            }
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.attn_output.bias"),
                qwen_dense_attention_bias_names(&prefix, "output"),
                embed,
            );
            if uses_mla_attention {
                let mla = mla_dims
                    .ok_or_else(|| anyhow!("complete MLA tensor layout missing MLA metadata"))?;
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.attn_q_a_norm.weight"),
                    qwen_mla_q_a_norm_weight_names(&prefix),
                    mla.q_lora_rank,
                );
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.attn_kv_a_norm.weight"),
                    qwen_mla_kv_a_norm_weight_names(&prefix),
                    mla.kv_lora_rank,
                );
                continue;
            }
            let gated_q = qwen_dense_gated_attention_q_weight_name(
                gguf,
                &prefix,
                u64::try_from(q_dim).context("qwen q dimension does not fit u64")?,
                u64::try_from(embed).context("qwen embedding dimension does not fit u64")?,
            )
            .is_some();
            if gated_q {
                if qwen_dense_gated_attention_q_bias_name(
                    gguf,
                    &prefix,
                    u64::try_from(q_dim).context("qwen q dimension does not fit u64")?,
                )
                .is_some()
                {
                    push_vector_alias_spec(
                        gguf,
                        &mut specs,
                        &format!("{prefix}.attn_q_gated.bias"),
                        qwen_dense_attention_bias_names(&prefix, "q"),
                        q_dim
                            .checked_mul(2)
                            .context("qwen gated q bias length overflows usize")?,
                    );
                }
            } else {
                push_vector_alias_spec(
                    gguf,
                    &mut specs,
                    &format!("{prefix}.attn_q.bias"),
                    qwen_dense_attention_bias_names(&prefix, "q"),
                    q_dim,
                );
            }
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.attn_k.bias"),
                qwen_dense_attention_bias_names(&prefix, "k"),
                k_dim,
            );
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.attn_v.bias"),
                qwen_dense_attention_bias_names(&prefix, "v"),
                v_dim,
            );
            let split_q_bias_present = qwen_dense_attention_bias_names(&prefix, "q")
                .iter()
                .any(|name| gguf.tensor(name).is_some());
            let split_k_bias_present = qwen_dense_attention_bias_names(&prefix, "k")
                .iter()
                .any(|name| gguf.tensor(name).is_some());
            let split_v_bias_present = qwen_dense_attention_bias_names(&prefix, "v")
                .iter()
                .any(|name| gguf.tensor(name).is_some());
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.attn_q_norm.weight"),
                qwen_dense_attention_head_norm_weight_names(&prefix, "q"),
                head_dim,
            );
            push_vector_alias_spec(
                gguf,
                &mut specs,
                &format!("{prefix}.attn_k_norm.weight"),
                qwen_dense_attention_head_norm_weight_names(&prefix, "k"),
                head_dim,
            );
            let Some(source) = qwen_dense_packed_qkv_bias_names(&prefix)
                .into_iter()
                .find(|name| gguf.tensor(name).is_some())
            else {
                continue;
            };
            let qkv_dim = q_dim
                .checked_add(k_dim)
                .and_then(|value| value.checked_add(v_dim))
                .context("packed qkv bias dimension overflows usize")?;
            if !split_q_bias_present {
                specs.push(VectorSpec::slice(
                    format!("{prefix}.attn_q.bias"),
                    source.clone(),
                    0,
                    q_dim,
                    qkv_dim,
                ));
            }
            if !split_k_bias_present {
                specs.push(VectorSpec::slice(
                    format!("{prefix}.attn_k.bias"),
                    source.clone(),
                    q_dim,
                    k_dim,
                    qkv_dim,
                ));
            }
            if !split_v_bias_present {
                specs.push(VectorSpec::slice(
                    format!("{prefix}.attn_v.bias"),
                    source,
                    q_dim + k_dim,
                    v_dim,
                    qkv_dim,
                ));
            }
        }
        Ok(specs)
    }

    fn push_vector_alias_spec(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        logical_name: &str,
        aliases: Vec<String>,
        len: usize,
    ) {
        for source_name in aliases {
            if source_name == logical_name {
                if gguf.tensor(logical_name).is_some() {
                    return;
                }
                continue;
            }
            if gguf.tensor(&source_name).is_some() {
                specs.push(VectorSpec::alias(
                    logical_name.to_string(),
                    source_name,
                    len,
                ));
                return;
            }
        }
    }

    fn push_dense_ffn_bias_vector_specs(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        prefix: &str,
        ff: usize,
        embed: usize,
    ) -> Result<()> {
        let gate_bias_names = qwen_dense_ffn_bias_names(prefix, "gate");
        let up_bias_names = qwen_dense_ffn_bias_names(prefix, "up");
        let split_gate_bias_present = gate_bias_names
            .iter()
            .any(|name| gguf.tensor(name).is_some());
        let split_up_bias_present = up_bias_names.iter().any(|name| gguf.tensor(name).is_some());
        push_vector_alias_spec(
            gguf,
            specs,
            &format!("{prefix}.ffn_gate.bias"),
            gate_bias_names,
            ff,
        );
        push_vector_alias_spec(
            gguf,
            specs,
            &format!("{prefix}.ffn_up.bias"),
            up_bias_names,
            ff,
        );
        push_vector_alias_spec(
            gguf,
            specs,
            &format!("{prefix}.ffn_down.bias"),
            qwen_dense_ffn_bias_names(prefix, "down"),
            embed,
        );
        let packed_source = qwen_dense_packed_ffn_gate_up_bias_names(prefix)
            .into_iter()
            .map(|name| (name, true))
            .chain(
                qwen_dense_packed_ffn_up_gate_bias_names(prefix)
                    .into_iter()
                    .map(|name| (name, false)),
            )
            .find(|(name, _)| gguf.tensor(name).is_some());
        let Some((source, gate_first)) = packed_source else {
            return Ok(());
        };
        let packed_len = ff
            .checked_mul(2)
            .context("qwen packed ffn bias length overflows usize")?;
        let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
        if !split_gate_bias_present {
            specs.push(VectorSpec::slice(
                format!("{prefix}.ffn_gate.bias"),
                source.clone(),
                gate_offset,
                ff,
                packed_len,
            ));
        }
        if !split_up_bias_present {
            specs.push(VectorSpec::slice(
                format!("{prefix}.ffn_up.bias"),
                source,
                up_offset,
                ff,
                packed_len,
            ));
        }
        Ok(())
    }

    fn push_moe_expert_bias_vector_specs(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        prefix: &str,
        experts: usize,
        ff: usize,
        embed: usize,
        use_per_expert_tensor: bool,
    ) -> Result<()> {
        for expert in 0..experts {
            push_moe_expert_gate_up_bias_vector_spec(
                gguf,
                specs,
                prefix,
                expert,
                experts,
                ff,
                use_per_expert_tensor,
                true,
            )?;
            push_moe_expert_gate_up_bias_vector_spec(
                gguf,
                specs,
                prefix,
                expert,
                experts,
                ff,
                use_per_expert_tensor,
                false,
            )?;
            push_moe_expert_bias_vector_spec(
                gguf,
                specs,
                prefix,
                expert,
                experts,
                "down",
                embed,
                use_per_expert_tensor,
            );
        }
        Ok(())
    }

    fn push_moe_expert_gate_up_bias_vector_spec(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        prefix: &str,
        expert: usize,
        experts: usize,
        ff: usize,
        use_per_expert_tensor: bool,
        gate: bool,
    ) -> Result<()> {
        let (kind, logical_kind) = if gate {
            ("gate", "ffn_gate_exps")
        } else {
            ("up", "ffn_up_exps")
        };
        let logical_name = format!("{prefix}.{logical_kind}.{expert}.bias");
        if use_per_expert_tensor {
            if let Some(source) =
                moe_per_expert_packed_gate_up_bias_source(gguf, prefix, ff, expert)?
            {
                let offset = if gate {
                    source.gate_offset
                } else {
                    source.up_offset
                };
                specs.push(VectorSpec::slice(
                    logical_name,
                    source.name,
                    offset,
                    ff,
                    source.source_rows,
                ));
                return Ok(());
            }
            push_vector_alias_spec(
                gguf,
                specs,
                &logical_name,
                qwen_moe_per_expert_bias_names(prefix, kind, expert as u64),
                ff,
            );
            return Ok(());
        }
        if let Some(source) = moe_packed_expert_gate_up_bias_source(gguf, prefix, ff, experts)? {
            let offset = if gate {
                source.gate_offset
            } else {
                source.up_offset
            };
            specs.push(VectorSpec::expert_slice(
                logical_name,
                source.name,
                expert,
                experts,
                offset,
                ff,
                source.source_rows,
            ));
            return Ok(());
        }
        push_expert_vector_alias_spec(
            gguf,
            specs,
            &logical_name,
            qwen_moe_packed_expert_bias_names(prefix, kind),
            expert,
            experts,
            ff,
        );
        Ok(())
    }

    fn push_moe_expert_bias_vector_spec(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        prefix: &str,
        expert: usize,
        experts: usize,
        kind: &str,
        len: usize,
        use_per_expert_tensor: bool,
    ) {
        let logical_name = format!("{prefix}.ffn_{kind}_exps.{expert}.bias");
        if use_per_expert_tensor {
            push_vector_alias_spec(
                gguf,
                specs,
                &logical_name,
                qwen_moe_per_expert_bias_names(prefix, kind, expert as u64),
                len,
            );
        } else {
            push_expert_vector_alias_spec(
                gguf,
                specs,
                &logical_name,
                qwen_moe_packed_expert_bias_names(prefix, kind),
                expert,
                experts,
                len,
            );
        }
    }

    fn push_moe_shared_expert_bias_vector_specs(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        prefix: &str,
        ff: usize,
        embed: usize,
    ) -> Result<()> {
        if let Some(source) = moe_shared_expert_packed_gate_up_bias_source(gguf, prefix, ff)? {
            specs.push(VectorSpec::slice(
                format!("{prefix}.ffn_gate_shexp.bias"),
                source.name.clone(),
                source.gate_offset,
                ff,
                source.source_rows,
            ));
            specs.push(VectorSpec::slice(
                format!("{prefix}.ffn_up_shexp.bias"),
                source.name,
                source.up_offset,
                ff,
                source.source_rows,
            ));
        } else {
            push_vector_alias_spec(
                gguf,
                specs,
                &format!("{prefix}.ffn_gate_shexp.bias"),
                qwen_moe_shared_expert_bias_names(prefix, "gate"),
                ff,
            );
            push_vector_alias_spec(
                gguf,
                specs,
                &format!("{prefix}.ffn_up_shexp.bias"),
                qwen_moe_shared_expert_bias_names(prefix, "up"),
                ff,
            );
        }
        push_vector_alias_spec(
            gguf,
            specs,
            &format!("{prefix}.ffn_down_shexp.bias"),
            qwen_moe_shared_expert_bias_names(prefix, "down"),
            embed,
        );
        Ok(())
    }

    fn push_expert_vector_alias_spec(
        gguf: &GgufFile,
        specs: &mut Vec<VectorSpec>,
        logical_name: &str,
        aliases: Vec<String>,
        expert: usize,
        experts: usize,
        len: usize,
    ) {
        for source_name in aliases {
            if gguf.tensor(&source_name).is_some() {
                specs.push(VectorSpec::expert_alias(
                    logical_name.to_string(),
                    source_name,
                    expert,
                    experts,
                    len,
                ));
                return;
            }
        }
    }

    fn qkv_split_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
        ["q", "k", "v"].iter().all(|suffix| {
            qwen_dense_attention_weight_names(prefix, suffix)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
        })
    }

    fn moe_router_tensor_present(gguf: &GgufFile, prefix: &str) -> bool {
        qwen_moe_router_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
    }

    struct PackedFfnSource {
        name: String,
        gate_offset: usize,
        up_offset: usize,
    }

    struct PackedMoeGateUpSource {
        name: String,
        source_rows: usize,
        gate_offset: usize,
        up_offset: usize,
    }

    fn dense_packed_ffn_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
        embed: usize,
    ) -> Result<Option<PackedFfnSource>> {
        let aliases = qwen_dense_packed_ffn_gate_up_weight_names(prefix)
            .into_iter()
            .map(|name| (name, true))
            .chain(
                qwen_dense_packed_ffn_up_gate_weight_names(prefix)
                    .into_iter()
                    .map(|name| (name, false)),
            )
            .chain(std::iter::once((format!("{prefix}.ffn_gate.weight"), true)))
            // Fused gate+up stored under `ffn_up` at 2x width (llama.cpp Phi-3 layout);
            // gate is the first half. The [2*ff, embed] shape check below rejects a plain
            // `ffn_up` (ff rows), so separate-tensor models fall through to the else branch.
            .chain(std::iter::once((format!("{prefix}.ffn_up.weight"), true)));
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            let packed_rows = ff
                .checked_mul(2)
                .context("qwen packed ffn row count overflows usize")?;
            if matrix_dims_match(&dims, packed_rows, embed) {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedFfnSource {
                    name,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn moe_packed_expert_gate_up_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
        embed: usize,
        experts: usize,
    ) -> Result<Option<PackedMoeGateUpSource>> {
        let source_rows = ff
            .checked_mul(2)
            .context("qwen packed MoE gate/up rows overflow usize")?;
        let aliases = qwen_moe_packed_expert_gate_up_weight_names(prefix)
            .into_iter()
            .map(|name| (name, true))
            .chain(
                qwen_moe_packed_expert_up_gate_weight_names(prefix)
                    .into_iter()
                    .map(|name| (name, false)),
            );
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if dims.len() == 3
                && dims[2] == experts
                && matrix_dims_match(&dims[..2], source_rows, embed)
            {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedMoeGateUpSource {
                    name,
                    source_rows,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn moe_packed_expert_gate_up_bias_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
        experts: usize,
    ) -> Result<Option<PackedMoeGateUpSource>> {
        let source_rows = ff
            .checked_mul(2)
            .context("qwen packed MoE gate/up bias length overflows usize")?;
        let aliases = qwen_moe_packed_expert_gate_up_bias_names(prefix)
            .into_iter()
            .map(|name| (name, true))
            .chain(
                qwen_moe_packed_expert_up_gate_bias_names(prefix)
                    .into_iter()
                    .map(|name| (name, false)),
            );
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if expert_vector_dims_match(&dims, source_rows, experts) {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedMoeGateUpSource {
                    name,
                    source_rows,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn moe_per_expert_packed_gate_up_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
        embed: usize,
        expert: usize,
    ) -> Result<Option<PackedMoeGateUpSource>> {
        let source_rows = ff
            .checked_mul(2)
            .context("qwen packed per-expert MoE gate/up rows overflow usize")?;
        let aliases = qwen_moe_per_expert_gate_up_weight_names(
            prefix,
            u64::try_from(expert).context("expert index does not fit u64")?,
        )
        .into_iter()
        .map(|name| (name, true))
        .chain(
            qwen_moe_per_expert_up_gate_weight_names(
                prefix,
                u64::try_from(expert).context("expert index does not fit u64")?,
            )
            .into_iter()
            .map(|name| (name, false)),
        );
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if dims.len() == 2 && matrix_dims_match(&dims, source_rows, embed) {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedMoeGateUpSource {
                    name,
                    source_rows,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn moe_per_expert_packed_gate_up_bias_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
        expert: usize,
    ) -> Result<Option<PackedMoeGateUpSource>> {
        let source_rows = ff
            .checked_mul(2)
            .context("qwen packed per-expert MoE gate/up bias length overflows usize")?;
        let aliases = qwen_moe_per_expert_gate_up_bias_names(
            prefix,
            u64::try_from(expert).context("expert index does not fit u64")?,
        )
        .into_iter()
        .map(|name| (name, true))
        .chain(
            qwen_moe_per_expert_up_gate_bias_names(
                prefix,
                u64::try_from(expert).context("expert index does not fit u64")?,
            )
            .into_iter()
            .map(|name| (name, false)),
        );
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if dims == [source_rows] {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedMoeGateUpSource {
                    name,
                    source_rows,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn moe_shared_expert_packed_gate_up_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
        embed: usize,
    ) -> Result<Option<PackedMoeGateUpSource>> {
        let source_rows = ff
            .checked_mul(2)
            .context("qwen packed shared expert gate/up rows overflow usize")?;
        let aliases = qwen_moe_shared_expert_gate_up_weight_names(prefix)
            .into_iter()
            .map(|name| (name, true))
            .chain(
                qwen_moe_shared_expert_up_gate_weight_names(prefix)
                    .into_iter()
                    .map(|name| (name, false)),
            );
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if dims.len() == 2 && matrix_dims_match(&dims, source_rows, embed) {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedMoeGateUpSource {
                    name,
                    source_rows,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn moe_shared_expert_packed_gate_up_bias_source(
        gguf: &GgufFile,
        prefix: &str,
        ff: usize,
    ) -> Result<Option<PackedMoeGateUpSource>> {
        let source_rows = ff
            .checked_mul(2)
            .context("qwen packed shared expert gate/up bias length overflows usize")?;
        let aliases = qwen_moe_shared_expert_gate_up_bias_names(prefix)
            .into_iter()
            .map(|name| (name, true))
            .chain(
                qwen_moe_shared_expert_up_gate_bias_names(prefix)
                    .into_iter()
                    .map(|name| (name, false)),
            );
        for (name, gate_first) in aliases {
            let Some(tensor) = gguf.tensor(&name) else {
                continue;
            };
            let dims = tensor
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            if dims == [source_rows] {
                let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
                return Ok(Some(PackedMoeGateUpSource {
                    name,
                    source_rows,
                    gate_offset,
                    up_offset,
                }));
            }
        }
        Ok(None)
    }

    fn matrix_dims_match(dims: &[usize], rows: usize, cols: usize) -> bool {
        matches!(
            dims,
            [dim0, dim1] if (*dim0 == cols && *dim1 == rows) || (*dim0 == rows && *dim1 == cols)
        )
    }

    fn expert_vector_dims_match(dims: &[usize], len: usize, experts: usize) -> bool {
        matches!(
            dims,
            [dim0, dim1] if (*dim0 == len && *dim1 == experts) || (*dim0 == experts && *dim1 == len)
        )
    }

    fn moe_expert_matrix_spec(
        gguf: &GgufFile,
        prefix: &str,
        kind: &str,
        packed_names: Vec<String>,
        per_expert_names: Vec<String>,
        expert: usize,
        rows: usize,
        cols: usize,
        use_per_expert_tensor: bool,
    ) -> MatrixSpec {
        let logical_name = format!("{prefix}.ffn_{kind}_exps.{expert}.weight");
        if use_per_expert_tensor {
            let per_expert_name = per_expert_names
                .into_iter()
                .find(|name| gguf.tensor(name).is_some());
            MatrixSpec::alias(
                logical_name,
                per_expert_name.unwrap_or_else(|| format!("{prefix}.ffn_{kind}.{expert}.weight")),
                rows,
                cols,
            )
        } else {
            let packed_name = packed_names
                .into_iter()
                .find(|name| gguf.tensor(name).is_some())
                .unwrap_or_else(|| format!("{prefix}.ffn_{kind}_exps.weight"));
            MatrixSpec::expert_alias(logical_name, packed_name, expert, rows, cols)
        }
    }

    fn expert_ff_dim(
        gguf: &GgufFile,
        config: &QwenGgufConfig,
        prefix: &str,
        experts: usize,
        embed: usize,
    ) -> Result<usize> {
        if let Some(ff) = config.expert_feed_forward_length {
            return usize::try_from(ff)
                .context("qwen expert_feed_forward_length does not fit usize");
        }
        let packed_gate_up_names = qwen_moe_packed_expert_gate_up_weight_names(prefix)
            .into_iter()
            .chain(qwen_moe_packed_expert_up_gate_weight_names(prefix));
        for name in packed_gate_up_names {
            let Some(view) = gguf.tensor(&name) else {
                continue;
            };
            let dims = view
                .info
                .dimensions
                .iter()
                .map(|dim| {
                    usize::try_from(*dim).context("expert tensor dimension does not fit usize")
                })
                .collect::<Result<Vec<_>>>()?;
            if dims.len() == 3 && dims[2] == experts {
                let packed_rows = dims
                    .iter()
                    .copied()
                    .find(|dim| *dim != embed && *dim != experts)
                    .ok_or_else(|| anyhow!("could not infer qwen packed MoE gate/up rows"))?;
                if packed_rows % 2 == 0 {
                    return Ok(packed_rows / 2);
                }
            }
        }
        let per_expert_packed_gate_up_names = qwen_moe_per_expert_gate_up_weight_names(prefix, 0)
            .into_iter()
            .chain(qwen_moe_per_expert_up_gate_weight_names(prefix, 0));
        for name in per_expert_packed_gate_up_names {
            let Some(view) = gguf.tensor(&name) else {
                continue;
            };
            let dims = view
                .info
                .dimensions
                .iter()
                .map(|dim| {
                    usize::try_from(*dim).context("expert tensor dimension does not fit usize")
                })
                .collect::<Result<Vec<_>>>()?;
            let packed_rows = match dims.as_slice() {
                [dim0, dim1] if *dim0 == embed => *dim1,
                [dim0, dim1] if *dim1 == embed => *dim0,
                _ => continue,
            };
            if packed_rows % 2 == 0 {
                return Ok(packed_rows / 2);
            }
        }
        let use_per_expert_tensor = !moe_packed_expert_tensors_complete(gguf, prefix)
            && moe_per_expert_tensors_complete(gguf, prefix, experts);
        let names = if use_per_expert_tensor {
            qwen_moe_per_expert_weight_names(prefix, "gate", 0)
        } else {
            qwen_moe_packed_expert_weight_names(prefix, "gate")
        };
        let name = names
            .iter()
            .find(|name| gguf.tensor(name).is_some())
            .or_else(|| names.first())
            .ok_or_else(|| anyhow!("expert tensor alias list is empty"))?;
        let view = gguf
            .tensor(name)
            .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
        let dims = view
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).context("expert tensor dimension does not fit usize"))
            .collect::<Result<Vec<_>>>()?;
        let dims = if use_per_expert_tensor {
            if dims.len() != 2 {
                bail!("expert tensor {name} has shape {dims:?}; expected rank 2");
            }
            dims
        } else {
            if dims.len() != 3 || dims[2] != experts {
                bail!(
                    "expert tensor {name} has shape {:?}; expected rank 3 with expert dimension {experts}",
                    dims
                );
            }
            dims[..2].to_vec()
        };
        dims.iter()
            .copied()
            .find(|dim| *dim != embed)
            .ok_or_else(|| anyhow!("could not infer qwen MoE expert feed-forward length"))
    }

    fn moe_packed_expert_tensors_complete(gguf: &GgufFile, prefix: &str) -> bool {
        ["gate", "up", "down"].iter().all(|kind| {
            qwen_moe_packed_expert_weight_names(prefix, kind)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
        })
    }

    fn moe_per_expert_tensors_complete(gguf: &GgufFile, prefix: &str, experts: usize) -> bool {
        (0..experts).all(|expert| {
            let expert = expert as u64;
            let split_complete = ["gate", "up", "down"].iter().all(|kind| {
                qwen_moe_per_expert_weight_names(prefix, kind, expert)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some())
            });
            let packed_gate_up_present = qwen_moe_per_expert_gate_up_weight_names(prefix, expert)
                .into_iter()
                .chain(qwen_moe_per_expert_up_gate_weight_names(prefix, expert))
                .any(|name| gguf.tensor(&name).is_some());
            let down_present = qwen_moe_per_expert_weight_names(prefix, "down", expert)
                .iter()
                .any(|name| gguf.tensor(name).is_some());
            split_complete || (packed_gate_up_present && down_present)
        })
    }

    fn matrix_source_dims(dims: &[u64], spec: &MatrixSpec) -> Result<Vec<u64>> {
        match spec.expert_index {
            Some(_) => {
                if dims.len() != 3 {
                    bail!(
                        "expert tensor {} must be rank 3, got {:?}",
                        spec.tensor_name,
                        dims
                    );
                }
                Ok(dims[..2].to_vec())
            }
            None => Ok(dims.to_vec()),
        }
    }

    fn matrix_source_bytes<'a>(
        bytes: &'a [u8],
        dims: &[u64],
        dtype: GgufTensorType,
        spec: &MatrixSpec,
    ) -> Result<&'a [u8]> {
        let Some(expert) = spec.expert_index else {
            return Ok(bytes);
        };
        if dims.len() != 3 {
            bail!(
                "expert tensor {} must be rank 3, got {:?}",
                spec.tensor_name,
                dims
            );
        }
        let experts =
            usize::try_from(dims[2]).context("expert tensor expert count does not fit usize")?;
        if expert >= experts {
            bail!("expert index {expert} is outside expert tensor count {experts}");
        }
        let elements_per_expert = dims[0]
            .checked_mul(dims[1])
            .context("expert tensor element count overflows u64")?;
        let bytes_per_expert = usize::try_from(dtype.byte_len(elements_per_expert)?)
            .context("expert tensor byte count does not fit usize")?;
        let start = expert
            .checked_mul(bytes_per_expert)
            .context("expert tensor byte offset overflows usize")?;
        let end = start
            .checked_add(bytes_per_expert)
            .context("expert tensor byte end overflows usize")?;
        bytes.get(start..end).ok_or_else(|| {
            anyhow!(
                "expert tensor {} byte slice is out of range",
                spec.tensor_name
            )
        })
    }

    fn normalize_matrix_bytes(
        bytes: &[u8],
        dims: &[u64],
        dtype: GgufTensorType,
        spec: &MatrixSpec,
    ) -> Result<Vec<u8>> {
        if dims.len() != 2 {
            bail!("tensor {} must be rank 2, got {:?}", spec.name, dims);
        }
        let element_size = usize::try_from(dtype.element_size())
            .context("GGUF tensor element size does not fit usize")?;
        let expected_bytes = spec
            .rows
            .checked_mul(spec.cols)
            .and_then(|elements| elements.checked_mul(element_size))
            .context("normalized matrix byte length overflows usize")?;
        if bytes.len() != expected_bytes {
            bail!(
                "tensor {} byte length {} does not match expected normalized matrix byte length {expected_bytes}",
                spec.name,
                bytes.len()
            );
        }

        let rows = u64::try_from(spec.rows).context("matrix rows do not fit u64")?;
        let cols = u64::try_from(spec.cols).context("matrix cols do not fit u64")?;
        let mut out = vec![0u8; expected_bytes];
        match dims {
            [dim0, dim1] if *dim0 == cols && *dim1 == rows => {
                for row in 0..spec.rows {
                    for col in 0..spec.cols {
                        copy_matrix_element(bytes, &mut out, element_size, row, col, spec.cols)?;
                    }
                }
            }
            [dim0, dim1] if *dim0 == rows && *dim1 == cols => {
                for row in 0..spec.rows {
                    for col in 0..spec.cols {
                        let source = col
                            .checked_mul(spec.rows)
                            .and_then(|offset| offset.checked_add(row))
                            .context("transposed matrix source index overflows usize")?;
                        let dest = row
                            .checked_mul(spec.cols)
                            .and_then(|offset| offset.checked_add(col))
                            .context("matrix destination index overflows usize")?;
                        copy_element(bytes, &mut out, element_size, source, dest)?;
                    }
                }
            }
            _ => bail!(
                "tensor {} has shape {:?}; expected [{}, {}] or [{}, {}]",
                spec.name,
                dims,
                spec.cols,
                spec.rows,
                spec.rows,
                spec.cols
            ),
        }
        Ok(out)
    }

    fn normalize_row_slice_matrix_bytes(
        bytes: &[u8],
        dims: &[u64],
        dtype: GgufTensorType,
        spec: &MatrixSpec,
        row_slice: RowSlice,
    ) -> Result<Vec<u8>> {
        if dims.len() != 2 {
            bail!("packed tensor {} must be rank 2, got {:?}", spec.name, dims);
        }
        let element_size = usize::try_from(dtype.element_size())
            .context("GGUF tensor element size does not fit usize")?;
        let expected_source_bytes = row_slice
            .source_rows
            .checked_mul(spec.cols)
            .and_then(|elements| elements.checked_mul(element_size))
            .context("packed matrix source byte length overflows usize")?;
        if bytes.len() != expected_source_bytes {
            bail!(
                "packed tensor {} byte length {} does not match expected source byte length {expected_source_bytes}",
                spec.tensor_name,
                bytes.len()
            );
        }
        let end = row_slice
            .offset
            .checked_add(spec.rows)
            .context("packed matrix row slice end overflows usize")?;
        if end > row_slice.source_rows {
            bail!(
                "packed tensor {} row slice {}..{end} exceeds source rows {}",
                spec.tensor_name,
                row_slice.offset,
                row_slice.source_rows
            );
        }

        let rows = u64::try_from(spec.rows).context("matrix rows do not fit u64")?;
        let cols = u64::try_from(spec.cols).context("matrix cols do not fit u64")?;
        let source_rows =
            u64::try_from(row_slice.source_rows).context("source rows do not fit u64")?;
        let expected_bytes = spec
            .rows
            .checked_mul(spec.cols)
            .and_then(|elements| elements.checked_mul(element_size))
            .context("normalized packed matrix byte length overflows usize")?;
        let mut out = vec![0u8; expected_bytes];
        match dims {
            [dim0, dim1] if *dim0 == cols && *dim1 == source_rows => {
                for row in 0..spec.rows {
                    let source_row = row_slice.offset + row;
                    for col in 0..spec.cols {
                        let source = source_row
                            .checked_mul(spec.cols)
                            .and_then(|offset| offset.checked_add(col))
                            .context("packed matrix source index overflows usize")?;
                        let dest = row
                            .checked_mul(spec.cols)
                            .and_then(|offset| offset.checked_add(col))
                            .context("packed matrix destination index overflows usize")?;
                        copy_element(bytes, &mut out, element_size, source, dest)?;
                    }
                }
            }
            [dim0, dim1] if *dim0 == source_rows && *dim1 == cols => {
                for row in 0..spec.rows {
                    let source_row = row_slice.offset + row;
                    for col in 0..spec.cols {
                        let source = col
                            .checked_mul(row_slice.source_rows)
                            .and_then(|offset| offset.checked_add(source_row))
                            .context("packed transposed matrix source index overflows usize")?;
                        let dest = row
                            .checked_mul(spec.cols)
                            .and_then(|offset| offset.checked_add(col))
                            .context("packed matrix destination index overflows usize")?;
                        copy_element(bytes, &mut out, element_size, source, dest)?;
                    }
                }
            }
            _ => bail!(
                "packed tensor {} has shape {:?}; expected [{}, {}] or [{}, {}] for logical {}x{} row slice",
                spec.tensor_name,
                dims,
                spec.cols,
                row_slice.source_rows,
                row_slice.source_rows,
                spec.cols,
                rows,
                cols
            ),
        }
        Ok(out)
    }

    fn normalize_vision_patch_matrix_bytes(
        gguf: &GgufFile,
        bytes: &[u8],
        dims: &[u64],
        dtype: GgufTensorType,
        spec: &MatrixSpec,
    ) -> Result<Option<Vec<u8>>> {
        if dims.len() != 4 {
            return Ok(None);
        }
        let patch_x = usize::try_from(dims[0]).context("patch tensor x dim does not fit usize")?;
        let patch_y = usize::try_from(dims[1]).context("patch tensor y dim does not fit usize")?;
        let channels =
            usize::try_from(dims[2]).context("patch tensor channel dim does not fit usize")?;
        let rows =
            usize::try_from(dims[3]).context("patch tensor output dim does not fit usize")?;
        if channels != 3 || rows != spec.rows {
            bail!(
                "tensor {} has patch shape {:?}; expected [patch, patch, 3, {}]",
                spec.name,
                dims,
                spec.rows
            );
        }
        let slice_cols = channels
            .checked_mul(patch_x)
            .and_then(|value| value.checked_mul(patch_y))
            .context("patch tensor slice columns overflow usize")?;
        if slice_cols == 0 || spec.cols % slice_cols != 0 {
            bail!(
                "tensor {} patch slice cols {slice_cols} do not divide expected cols {}",
                spec.name,
                spec.cols
            );
        }
        let temporal = spec.cols / slice_cols;
        let element_size = usize::try_from(dtype.element_size())
            .context("GGUF patch tensor element size does not fit usize")?;
        let slice_bytes = rows
            .checked_mul(slice_cols)
            .and_then(|elements| elements.checked_mul(element_size))
            .context("patch tensor slice byte length overflows usize")?;
        if bytes.len() != slice_bytes {
            bail!(
                "tensor {} byte length {} does not match expected patch slice byte length {slice_bytes}",
                spec.name,
                bytes.len()
            );
        }

        let mut slices = Vec::with_capacity(temporal);
        slices.push(bytes);
        for temporal_idx in 1..temporal {
            let name = format!("{}.{}", spec.tensor_name, temporal_idx);
            let tensor = gguf
                .tensor(&name)
                .ok_or_else(|| anyhow!("missing CUDA vision patch temporal tensor {name}"))?;
            if tensor.info.dtype != dtype || tensor.info.dimensions != dims {
                bail!(
                    "tensor {name} has dtype {} and shape {:?}; expected {} {:?}",
                    tensor.info.dtype.label(),
                    tensor.info.dimensions,
                    dtype.label(),
                    dims
                );
            }
            if tensor.bytes.len() != slice_bytes {
                bail!(
                    "tensor {name} byte length {} does not match expected patch slice byte length {slice_bytes}",
                    tensor.bytes.len()
                );
            }
            slices.push(tensor.bytes);
        }

        let expected_bytes = spec
            .rows
            .checked_mul(spec.cols)
            .and_then(|elements| elements.checked_mul(element_size))
            .context("normalized patch matrix byte length overflows usize")?;
        let mut out = vec![0u8; expected_bytes];
        for row in 0..rows {
            for channel in 0..channels {
                for (temporal_idx, slice) in slices.iter().enumerate() {
                    for y in 0..patch_y {
                        for x in 0..patch_x {
                            let source_index = row
                                .checked_mul(channels)
                                .and_then(|value| value.checked_add(channel))
                                .and_then(|value| value.checked_mul(patch_y))
                                .and_then(|value| value.checked_add(y))
                                .and_then(|value| value.checked_mul(patch_x))
                                .and_then(|value| value.checked_add(x))
                                .context("patch tensor source index overflows usize")?;
                            let col = channel
                                .checked_mul(temporal)
                                .and_then(|value| value.checked_add(temporal_idx))
                                .and_then(|value| value.checked_mul(patch_y))
                                .and_then(|value| value.checked_add(y))
                                .and_then(|value| value.checked_mul(patch_x))
                                .and_then(|value| value.checked_add(x))
                                .context("patch tensor destination column overflows usize")?;
                            let dest_index = row
                                .checked_mul(spec.cols)
                                .and_then(|value| value.checked_add(col))
                                .context("patch tensor destination index overflows usize")?;
                            copy_element(slice, &mut out, element_size, source_index, dest_index)?;
                        }
                    }
                }
            }
        }
        Ok(Some(out))
    }

    fn normalize_quantized_row_slice_matrix_bytes<'a>(
        bytes: &'a [u8],
        dims: &[u64],
        dtype: GgufTensorType,
        spec: &MatrixSpec,
        row_slice: RowSlice,
    ) -> Result<&'a [u8]> {
        if dims.len() != 2 {
            bail!("packed tensor {} must be rank 2, got {:?}", spec.name, dims);
        }
        let cols = u64::try_from(spec.cols).context("matrix cols do not fit u64")?;
        let source_rows =
            u64::try_from(row_slice.source_rows).context("source rows do not fit u64")?;
        match dims {
            [dim0, dim1] if *dim0 == cols && *dim1 == source_rows => {}
            [dim0, dim1] if *dim0 == source_rows && *dim1 == cols => {
                bail!(
                    "quantized packed matrix tensor {} has transposed shape {:?}; CUDA quantized row slicing requires GGUF shape [{cols}, {source_rows}]",
                    spec.tensor_name,
                    dims
                );
            }
            _ => bail!(
                "packed tensor {} has shape {:?}; expected quantized GGUF shape [{}, {}]",
                spec.tensor_name,
                dims,
                cols,
                source_rows
            ),
        }
        let row_elements = cols;
        let row_bytes = usize::try_from(dtype.byte_len(row_elements)?)
            .context("quantized row byte length does not fit usize")?;
        let source_bytes = row_slice
            .source_rows
            .checked_mul(row_bytes)
            .context("quantized packed source byte length overflows usize")?;
        if bytes.len() != source_bytes {
            bail!(
                "packed tensor {} byte length {} does not match expected quantized source byte length {source_bytes}",
                spec.tensor_name,
                bytes.len()
            );
        }
        let end_row = row_slice
            .offset
            .checked_add(spec.rows)
            .context("quantized packed row slice end overflows usize")?;
        if end_row > row_slice.source_rows {
            bail!(
                "packed tensor {} row slice {}..{end_row} exceeds source rows {}",
                spec.tensor_name,
                row_slice.offset,
                row_slice.source_rows
            );
        }
        let start = row_slice
            .offset
            .checked_mul(row_bytes)
            .context("quantized packed row byte start overflows usize")?;
        let len = spec
            .rows
            .checked_mul(row_bytes)
            .context("quantized packed row byte length overflows usize")?;
        let end = start
            .checked_add(len)
            .context("quantized packed row byte end overflows usize")?;
        bytes.get(start..end).ok_or_else(|| {
            anyhow!(
                "packed tensor {} byte slice {}..{} is out of range",
                spec.tensor_name,
                start,
                end
            )
        })
    }

    fn normalize_quantized_matrix_bytes<'a>(
        bytes: &'a [u8],
        dims: &[u64],
        dtype: GgufTensorType,
        spec: &MatrixSpec,
    ) -> Result<&'a [u8]> {
        if dims.len() != 2 {
            bail!("tensor {} must be rank 2, got {:?}", spec.name, dims);
        }
        let rows = u64::try_from(spec.rows).context("matrix rows do not fit u64")?;
        let cols = u64::try_from(spec.cols).context("matrix cols do not fit u64")?;
        match dims {
            [dim0, dim1] if *dim0 == cols && *dim1 == rows => {}
            [dim0, dim1] if *dim0 == rows && *dim1 == cols => {
                bail!(
                    "quantized matrix tensor {} has transposed shape {:?}; CUDA quantized loading currently requires GGUF shape [{cols}, {rows}]",
                    spec.name,
                    dims
                );
            }
            _ => bail!(
                "tensor {} has shape {:?}; expected quantized GGUF shape [{}, {}]",
                spec.name,
                dims,
                cols,
                rows
            ),
        }
        let elements = rows
            .checked_mul(cols)
            .context("quantized matrix element count overflows u64")?;
        let expected = usize::try_from(dtype.byte_len(elements)?)
            .context("quantized matrix byte length does not fit usize")?;
        if bytes.len() != expected {
            bail!(
                "tensor {} byte length {} does not match expected quantized matrix byte length {expected}",
                spec.name,
                bytes.len()
            );
        }
        Ok(bytes)
    }

    fn copy_matrix_element(
        source: &[u8],
        dest: &mut [u8],
        element_size: usize,
        row: usize,
        col: usize,
        cols: usize,
    ) -> Result<()> {
        let index = row
            .checked_mul(cols)
            .and_then(|offset| offset.checked_add(col))
            .context("matrix element index overflows usize")?;
        copy_element(source, dest, element_size, index, index)
    }

    fn copy_element(
        source: &[u8],
        dest: &mut [u8],
        element_size: usize,
        source_index: usize,
        dest_index: usize,
    ) -> Result<()> {
        let source_start = source_index
            .checked_mul(element_size)
            .context("source byte index overflows usize")?;
        let dest_start = dest_index
            .checked_mul(element_size)
            .context("destination byte index overflows usize")?;
        let source_end = source_start + element_size;
        let dest_end = dest_start + element_size;
        dest[dest_start..dest_end].copy_from_slice(&source[source_start..source_end]);
        Ok(())
    }

    fn read_tensor_as_f32(
        bytes: &[u8],
        dtype: GgufTensorType,
        element_count: usize,
    ) -> Result<Vec<f32>> {
        dequantize_tensor_as_f32(bytes, dtype, element_count)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn qwen25_window_plan_reorders_merged_groups_then_reverses_merger_rows() {
            let plan = vision_window_plan([1, 4, 6], 1, 2, 4).unwrap();

            assert_eq!(
                plan.patch_row_order,
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21,
                    22, 23
                ]
            );
            assert_eq!(plan.merged_reverse_order, vec![0, 1, 4, 2, 3, 5]);
            assert_eq!(
                plan.window_start,
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16
                ]
            );
            assert_eq!(
                plan.window_end,
                vec![
                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24,
                    24, 24, 24, 24
                ]
            );
        }
    }
}

#[cfg(feature = "native-cuda")]
pub use native::{
    CudaMmprojProjector, CudaQwenGpuModel, CudaVisionEncoder, GpuMatrix, GpuTensor, GpuVector,
};

#[cfg(not(feature = "native-cuda"))]
mod non_native {
    use anyhow::{Result, bail};
    use hi_gguf::GgufFile;

    use super::{CudaMmprojProjectorInfo, CudaQwenGpuModelInfo, CudaVisionEncoderInfo};

    #[derive(Debug)]
    pub struct CudaMmprojProjector {
        #[allow(dead_code)]
        info: CudaMmprojProjectorInfo,
    }

    impl CudaMmprojProjector {
        pub fn from_gguf(_gguf: &GgufFile) -> Result<Self> {
            bail!(
                "hi-cuda was built without native-cuda support; CUDA mmproj loading is unavailable"
            )
        }

        pub fn info(&self) -> &CudaMmprojProjectorInfo {
            &self.info
        }

        pub fn project_features_host(&self, _features: &[f32], _rows: usize) -> Result<Vec<f32>> {
            bail!(
                "hi-cuda was built without native-cuda support; CUDA mmproj projection is unavailable"
            )
        }
    }

    #[derive(Debug)]
    pub struct CudaVisionEncoder {
        #[allow(dead_code)]
        info: CudaVisionEncoderInfo,
    }

    impl CudaVisionEncoder {
        pub fn from_gguf(_gguf: &GgufFile) -> Result<Self> {
            bail!(
                "hi-cuda was built without native-cuda support; CUDA Qwen-VL vision loading is unavailable"
            )
        }

        pub fn info(&self) -> &CudaVisionEncoderInfo {
            &self.info
        }

        pub fn encode_patches_host(
            &self,
            _patches: &[f32],
            _grids: &[[usize; 3]],
        ) -> Result<Vec<f32>> {
            bail!(
                "hi-cuda was built without native-cuda support; CUDA Qwen-VL vision encoding is unavailable"
            )
        }
    }

    #[derive(Debug)]
    pub struct CudaQwenGpuModel {
        #[allow(dead_code)]
        info: CudaQwenGpuModelInfo,
    }

    impl CudaQwenGpuModel {
        pub fn from_gguf(_gguf: &GgufFile) -> Result<Self> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU tensor loading is unavailable"
            )
        }

        pub fn info(&self) -> &CudaQwenGpuModelInfo {
            &self.info
        }

        pub(crate) fn reset_generation_timing(&self) {}

        pub(crate) fn take_generation_timing(&self) -> (u64, u64) {
            (0, 0)
        }

        pub fn forget_recurrent_page_contexts(
            &self,
            _page_tables: &[Vec<usize>],
            _label: &str,
        ) -> Result<usize> {
            Ok(0)
        }

        pub fn recurrent_page_context_count(&self) -> usize {
            0
        }

        pub fn full_context_logits_host(&self, _token_ids: &[u32]) -> Result<Vec<f32>> {
            bail!("hi-cuda was built without native-cuda support; GPU Qwen forward is unavailable")
        }

        pub fn embed_tokens_host(&self, _token_ids: &[u32]) -> Result<Vec<f32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen embedding lookup is unavailable"
            )
        }

        pub fn last_logits_host(&self, _token_ids: &[u32]) -> Result<Vec<f32>> {
            bail!("hi-cuda was built without native-cuda support; GPU Qwen forward is unavailable")
        }

        pub fn greedy_next_token(&self, _token_ids: &[u32]) -> Result<u32> {
            bail!("hi-cuda was built without native-cuda support; GPU Qwen forward is unavailable")
        }

        pub fn generate_greedy_tokens(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_stop_ids(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_stop_sequences(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_stop_sequences_and_cancellation<F>(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _stop_token_sequences: &[Vec<u32>],
            _is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_paged(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_paged_with_stop_ids(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_paged_with_stop_sequences(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_paged_with_stop_sequences_and_cancellation<F>(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
            _is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_with_limits(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_with_limits_and_cancellation<F>(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_and_limits_and_cancellation<F>(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows: usize,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows: usize,
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal MRoPE generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows: usize,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows: usize,
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal MRoPE generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_limits(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_limits_and_cancellation<F>(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_page_tables(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_batch_paged_with_page_tables_and_cancellation<F>(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn greedy_next_tokens_batch_paged_with_page_tables(
            &self,
            _inputs: &[Vec<u32>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn sampled_next_tokens_batch_paged_with_page_tables(
            &self,
            _inputs: &[Vec<u32>],
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _samples: &[f32],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn prefill_greedy_next_tokens_batch_paged_with_page_tables(
            &self,
            _inputs: &[Vec<u32>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn prefill_greedy_next_tokens_paged_reusing_prefix(
            &self,
            _input_ids: &[u32],
            _reuse_tokens: usize,
            _page_size: usize,
            _page_table: &[usize],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn prefill_sampled_next_tokens_batch_paged_with_page_tables(
            &self,
            _inputs: &[Vec<u32>],
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _samples: &[f32],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn decode_greedy_next_tokens_batch_paged_with_page_tables(
            &self,
            _token_ids: &[u32],
            _position: usize,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn decode_sampled_next_tokens_batch_paged_with_page_tables(
            &self,
            _token_ids: &[u32],
            _position: usize,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _samples: &[f32],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_limits(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_limits_and_cancellation<F>(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_and_limits_and_cancellation<F>(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows: usize,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows: usize,
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal MRoPE generation is unavailable"
            )
        }

        #[cfg(feature = "native-cuda")]
        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prefix_embeddings_per_request: &[Vec<f32>],
            prefix_rows: usize,
            inputs: &[Vec<u32>],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if inputs.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = inputs.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched multimodal greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            if prefix_embeddings_per_request.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation got {} visual prefixes for {batch_count} requests",
                    prefix_embeddings_per_request.len()
                );
            }
            if max_tokens_per_request.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation got {} token limits for {batch_count} requests",
                    max_tokens_per_request.len()
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            let token_prompt_len = inputs[0].len();
            if token_prompt_len == 0 {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires non-empty prompts"
                );
            }
            if inputs.iter().any(|input| input.len() != token_prompt_len) {
                bail!(
                    "CUDA lease-backed paged batched multimodal greedy generation requires equal prompt lengths"
                );
            }
            for prefix_embeddings in prefix_embeddings_per_request {
                self.validate_prefix_embeddings(prefix_embeddings, prefix_rows, dims.embed)?;
            }
            let prompt_len = prefix_rows.checked_add(token_prompt_len).context(
                "CUDA lease-backed paged batched multimodal prompt length overflows usize",
            )?;
            if prompt_len > dims.context {
                bail!(
                    "multimodal input length {prompt_len} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched multimodal greedy generation",
                prompt_len,
                max_decode_steps,
                dims.context,
            )?;

            let hidden_host = self.batched_prefix_prompt_embeddings_host(
                prefix_embeddings_per_request,
                prefix_rows,
                inputs,
                &dims,
            )?;
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_len).context(
                    "CUDA lease-backed paged batched multimodal hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA lease-backed paged batched multimodal prompt embeddings",
            )?;
            let token_capacity = prompt_len
                .checked_add(max_decode_steps)
                .context(
                    "CUDA lease-backed paged batched multimodal greedy token capacity overflows usize",
                )?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let logits = self.full_context_logits_from_hidden_device_batched_paged_cache(
                hidden,
                batch_count,
                prompt_len,
                &mut cache,
            )?;

            self.generate_batched_greedy_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_len,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_position| {
                    self.decode_batch_logits_paged_device(next_tokens, next_position, &mut cache)
                },
            )
        }

        #[cfg(feature = "native-cuda")]
        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            prompt_embeddings_per_request: &[Vec<f32>],
            prompt_rows: usize,
            position_ids_per_request: &[Vec<[u32; 3]>],
            next_rope_positions: &[usize],
            max_tokens_per_request: &[usize],
            eos_token_id: Option<u32>,
            page_size: usize,
            page_tables: &[Vec<usize>],
            physical_page_count: usize,
            stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            if prompt_embeddings_per_request.is_empty() {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation requires at least one request"
                );
            }
            if !self.supports_batched_multimodal_generation() {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation currently supports batched text-compatible decoder model layouts only"
                );
            }
            let dims = self.qwen_dims()?;
            let batch_count = prompt_embeddings_per_request.len();
            validate_batched_stop_token_sequences(
                "CUDA lease-backed paged batched multimodal MRoPE greedy generation",
                stop_token_sequences_per_request,
                batch_count,
            )?;
            if page_tables.len() != batch_count {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation got {} page table(s) for {batch_count} request(s)",
                    page_tables.len()
                );
            }
            if position_ids_per_request.len() != batch_count
                || next_rope_positions.len() != batch_count
                || max_tokens_per_request.len() != batch_count
            {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation got mismatched batch metadata for {batch_count} request(s)"
                );
            }
            if max_tokens_per_request.iter().any(|limit| *limit == 0) {
                bail!(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy generation requires positive token limits"
                );
            }
            let max_decode_steps = max_tokens_per_request.iter().copied().max().unwrap_or(1);
            if prompt_rows == 0 || prompt_rows > dims.context {
                bail!(
                    "multimodal prompt length {prompt_rows} exceeds qwen context length {}",
                    dims.context
                );
            }
            validate_batched_generation_context_budget(
                "CUDA lease-backed paged batched multimodal MRoPE greedy generation",
                prompt_rows,
                max_decode_steps,
                dims.context,
            )?;
            for next_rope_position in next_rope_positions {
                if *next_rope_position > dims.context {
                    bail!(
                        "multimodal next RoPE position {next_rope_position} exceeds qwen context length {}",
                        dims.context
                    );
                }
            }

            let total_values = batch_count
                .checked_mul(prompt_rows)
                .and_then(|value| value.checked_mul(dims.embed))
                .context(
                    "CUDA lease-backed paged batched multimodal MRoPE hidden value count overflows usize",
                )?;
            let mut hidden_host = Vec::with_capacity(total_values);
            let mut flat_positions = Vec::with_capacity(batch_count * prompt_rows);
            for (embeddings, position_ids) in prompt_embeddings_per_request
                .iter()
                .zip(position_ids_per_request)
            {
                self.validate_prompt_embeddings(embeddings, prompt_rows, dims.embed)?;
                if position_ids.len() != prompt_rows {
                    bail!(
                        "CUDA lease-backed paged batched multimodal MRoPE greedy generation got {} position row(s); expected {prompt_rows}",
                        position_ids.len()
                    );
                }
                hidden_host.extend_from_slice(embeddings);
                flat_positions.extend_from_slice(position_ids);
            }
            let hidden = self.f32_tensor_from_host(
                &hidden_host,
                batch_count.checked_mul(prompt_rows).context(
                    "CUDA lease-backed paged batched multimodal MRoPE hidden row count overflows usize",
                )?,
                dims.embed,
                "CUDA lease-backed paged batched multimodal MRoPE prompt embeddings",
            )?;
            let token_capacity = prompt_rows
                .checked_add(max_decode_steps)
                .context(
                    "CUDA lease-backed paged batched multimodal MRoPE greedy token capacity overflows usize",
                )?
                .min(dims.context);
            let mut pool_slot = self.paged_batch_pool.borrow_mut();
            let recreate_pool = pool_slot
                .as_ref()
                .is_none_or(|pool| !pool.can_cover(&dims, page_size, physical_page_count));
            if recreate_pool {
                *pool_slot = Some(CudaPagedBatchDevicePool::new(
                    self.config.block_count,
                    &dims,
                    page_size,
                    physical_page_count,
                    &self.stream,
                )?);
            }
            let pool = pool_slot
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA paged batch device pool was not initialized"))?;
            pool.zero(&self.stream)?;
            let mut cache = CudaPagedBatchKvCache::new_with_page_tables_from_pool(
                &dims,
                batch_count,
                page_size,
                token_capacity,
                page_tables,
                pool,
                &self.stream,
            )?;
            let logits = self
                .full_context_logits_from_hidden_device_batched_paged_cache_with_position_ids(
                    hidden,
                    batch_count,
                    prompt_rows,
                    &flat_positions,
                    &mut cache,
                )?;
            let mut next_rope_positions = next_rope_positions.to_vec();

            self.generate_batched_greedy_from_logits_with_cancellation(
                logits,
                batch_count,
                prompt_rows,
                max_decode_steps,
                max_tokens_per_request,
                eos_token_id,
                stop_token_sequences_per_request,
                is_cancelled,
                |next_tokens, next_cache_position| {
                    let decoded = self.decode_batch_logits_paged_device_with_rope_positions(
                        next_tokens,
                        next_cache_position,
                        &next_rope_positions,
                        &mut cache,
                    )?;
                    for position in &mut next_rope_positions {
                        *position = position
                            .checked_add(1)
                            .context("next MRoPE decode position overflows usize")?;
                    }
                    Ok(decoded)
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows: usize,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prefix_embeddings_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows_per_request: &[usize],
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged ragged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows_per_request: &[usize],
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged ragged multimodal MRoPE generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_batch_with_prompt_embeddings_positions_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows: usize,
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal MRoPE generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows: usize,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows_per_request: &[usize],
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged ragged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_ragged_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows_per_request: &[usize],
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged ragged multimodal MRoPE generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_paged_page_tables_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows: usize,
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal MRoPE generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prefix_embeddings_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prefix_embeddings_per_request: &[Vec<f32>],
            _prefix_rows: usize,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_with_prompt_embeddings_positions_paged_and_limits_and_cancellation<
            F,
        >(
            &self,
            _prompt_embeddings_per_request: &[Vec<f32>],
            _prompt_rows: usize,
            _position_ids_per_request: &[Vec<[u32; 3]>],
            _next_rope_positions: &[usize],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal MRoPE generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_limits(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_limits_and_cancellation<F>(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_page_tables(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
        ) -> Result<Vec<Vec<u32>>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_batch_paged_with_page_tables_and_cancellation<F>(
            &self,
            _inputs: &[Vec<u32>],
            _max_tokens_per_request: &[usize],
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seeds: &[Option<u64>],
            _page_size: usize,
            _page_tables: &[Vec<usize>],
            _physical_page_count: usize,
            _stop_token_sequences_per_request: &[Vec<Vec<u32>>],
            _is_cancelled: F,
        ) -> Result<Vec<Vec<u32>>>
        where
            F: FnMut(usize) -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prefix_embeddings(
            &self,
            _prefix_embeddings: &[f32],
            _prefix_rows: usize,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prefix_embeddings_and_stop_sequences(
            &self,
            _prefix_embeddings: &[f32],
            _prefix_rows: usize,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings_and_positions(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _position_ids: &[[u32; 3]],
            _next_rope_position: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_with_prompt_embeddings_and_positions_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _position_ids: &[[u32; 3]],
            _next_rope_position: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prefix_embeddings_paged_and_stop_sequences(
            &self,
            _prefix_embeddings: &[f32],
            _prefix_rows: usize,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        pub fn generate_greedy_tokens_with_prompt_embeddings_paged_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_greedy_tokens_with_prompt_embeddings_and_positions_paged_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _position_ids: &[[u32; 3]],
            _next_rope_position: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        pub fn generate_sampled_tokens(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_stop_ids(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_stop_sequences(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_stop_sequences_and_cancellation<F>(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _stop_token_sequences: &[Vec<u32>],
            _is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged_with_stop_ids(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
            _stop_token_ids: &[u32],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged_with_stop_sequences(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_paged_with_stop_sequences_and_cancellation<F>(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
            _is_cancelled: F,
        ) -> Result<Vec<u32>>
        where
            F: FnMut() -> bool,
        {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged generation is unavailable"
            )
        }

        pub fn generate_sampled_tokens_with_prefix_embeddings(
            &self,
            _prefix_embeddings: &[f32],
            _prefix_rows: usize,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prefix_embeddings_and_stop_sequences(
            &self,
            _prefix_embeddings: &[f32],
            _prefix_rows: usize,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        pub fn generate_sampled_tokens_with_prompt_embeddings(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_positions(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _position_ids: &[[u32; 3]],
            _next_rope_position: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_positions_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _position_ids: &[[u32; 3]],
            _next_rope_position: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prefix_embeddings_paged_and_stop_sequences(
            &self,
            _prefix_embeddings: &[f32],
            _prefix_rows: usize,
            _input_ids: &[u32],
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_paged_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn generate_sampled_tokens_with_prompt_embeddings_and_positions_paged_and_stop_sequences(
            &self,
            _prompt_embeddings: &[f32],
            _prompt_rows: usize,
            _position_ids: &[[u32; 3]],
            _next_rope_position: usize,
            _max_tokens: usize,
            _eos_token_id: Option<u32>,
            _temperature: f32,
            _top_p: f32,
            _top_k: Option<u32>,
            _seed: Option<u64>,
            _page_size: usize,
            _stop_token_sequences: &[Vec<u32>],
        ) -> Result<Vec<u32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged multimodal generation is unavailable"
            )
        }

        pub fn kv_decode_logits_host(&self, _prefix: &[u32], _token_id: u32) -> Result<Vec<f32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen KV decode is unavailable"
            )
        }

        pub fn paged_kv_decode_logits_host(
            &self,
            _prefix: &[u32],
            _token_id: u32,
            _page_size: usize,
        ) -> Result<Vec<f32>> {
            bail!(
                "hi-cuda was built without native-cuda support; GPU Qwen paged KV decode is unavailable"
            )
        }
    }
}

#[cfg(not(feature = "native-cuda"))]
pub use non_native::{CudaMmprojProjector, CudaQwenGpuModel, CudaVisionEncoder};
