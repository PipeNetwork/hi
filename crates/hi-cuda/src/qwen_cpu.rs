use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{
    GgufFile, GgufTokenizer, QwenGgufConfig, dequantize_tensor_as_f32,
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
    qwen_ssm_a_names, qwen_ssm_alpha_weight_names, qwen_ssm_ba_weight_names,
    qwen_ssm_beta_weight_names, qwen_ssm_conv1d_weight_names, qwen_ssm_dt_bias_names,
    qwen_ssm_gate_weight_names, qwen_ssm_in_weight_names, qwen_ssm_layer_tensors_present,
    qwen_ssm_norm_weight_names, qwen_ssm_out_weight_names, qwen_ssm_qkv_weight_names,
};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct QwenCpuRunOptions {
    pub max_tokens: usize,
    pub top_k: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub include_logits: bool,
}

impl Default for QwenCpuRunOptions {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            top_k: 5,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
            include_logits: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct QwenCpuRunOutput {
    pub backend: &'static str,
    pub input_tokens: Vec<u32>,
    pub next_token: u32,
    pub next_text: String,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub top_logits: Vec<TopLogit>,
    pub logit_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TopLogit {
    pub token_id: u32,
    pub token: Option<String>,
    pub logit: f32,
}

#[derive(Debug)]
pub struct QwenCpuReference {
    config: QwenGgufConfig,
    tokenizer: GgufTokenizer,
    embeddings: EmbeddingTable,
    layers: Vec<QwenLayer>,
    output_norm: Vec<f32>,
    output: Option<Matrix>,
    output_bias: Option<Vec<f32>>,
    rms_eps: f32,
    // False when built tokenizer-only (`from_gguf_tokenizer_only`) — the weight tensors
    // were skipped, so `forward` / `last_logits` are unavailable.
    weights_loaded: bool,
}

impl QwenCpuReference {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Self::from_gguf_inner(gguf, true)
    }

    /// Build a reference that carries only the tokenizer + config, skipping the (dequantized
    /// f32) weight tensors. The GPU execution path only needs the tokenizer, so this avoids
    /// materializing a full f32 copy of the model on the CPU — which for a 30B model is ~120
    /// GB and minutes of dequant work, making large models effectively unloadable otherwise.
    /// `forward` / `last_logits` return an error on a tokenizer-only reference.
    pub fn from_gguf_tokenizer_only(gguf: &GgufFile) -> Result<Self> {
        Self::from_gguf_inner(gguf, false)
    }

    fn from_gguf_inner(gguf: &GgufFile, load_weights: bool) -> Result<Self> {
        let config = gguf.qwen_config()?;
        // The tokenizer-only reference (GPU execution) skips tensor validation: the GPU model
        // built alongside it validates, so a redundant ~5 s pass on a large MoE is avoided.
        // The weight-loading reference validates, since it dequantizes those tensors here.
        if load_weights {
            gguf.validate_qwen_tensors()?;
        }
        let tokenizer = gguf.tokenizer()?;
        let embedding_length = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let vocab_size = config
            .vocab_size
            .map(usize::try_from)
            .transpose()
            .context("qwen vocab size does not fit usize")?
            .unwrap_or_else(|| tokenizer.token_count());
        if vocab_size != tokenizer.token_count() {
            bail!(
                "qwen vocab size {vocab_size} does not match tokenizer size {}",
                tokenizer.token_count()
            );
        }

        let rms_eps = config.rms_norm_eps.unwrap_or(1.0e-6);
        if !load_weights {
            return Ok(Self {
                config,
                tokenizer,
                embeddings: EmbeddingTable {
                    vocab: 0,
                    embed: 0,
                    data: Vec::new(),
                },
                layers: Vec::new(),
                output_norm: Vec::new(),
                output: None,
                output_bias: None,
                rms_eps,
                weights_loaded: false,
            });
        }
        let embeddings = EmbeddingTable::load_aliases(
            gguf,
            &qwen_dense_token_embd_weight_names(),
            vocab_size,
            embedding_length,
        )?;
        let output_norm = load_vector_aliases(
            gguf,
            &qwen_dense_output_norm_weight_names(),
            embedding_length,
        )?;
        let output_weight_names = qwen_dense_output_weight_names();
        let output = if output_weight_names
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        {
            Some(load_matrix_aliases(
                gguf,
                &output_weight_names,
                vocab_size,
                embedding_length,
            )?)
        } else {
            None
        };
        let output_bias =
            optional_vector_aliases(gguf, &qwen_dense_output_bias_names(), vocab_size)?;

        let mut layers = Vec::new();
        for idx in 0..config.block_count {
            layers.push(QwenLayer::load(gguf, &config, idx)?);
        }

        Ok(Self {
            config,
            tokenizer,
            embeddings,
            layers,
            output_norm,
            output,
            output_bias,
            rms_eps,
            weights_loaded: true,
        })
    }

    pub fn config(&self) -> &QwenGgufConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &GgufTokenizer {
        &self.tokenizer
    }

    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if !self.weights_loaded {
            bail!(
                "Qwen CPU reference was built tokenizer-only (GPU execution); its weights are \
                 not loaded, so CPU forward/last_logits is unavailable"
            );
        }
        if input_ids.is_empty() {
            bail!("Qwen CPU reference requires at least one input token");
        }
        let max_len = usize::try_from(self.config.context_length)
            .context("qwen context_length does not fit usize")?;
        if input_ids.len() > max_len {
            bail!(
                "input length {} exceeds qwen context length {max_len}",
                input_ids.len()
            );
        }

        let mut hidden = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(hidden, self.rms_eps)?;
        }
        for token in &mut hidden {
            rms_norm_in_place(token, &self.output_norm, self.rms_eps)?;
        }

        let mut logits = Vec::with_capacity(hidden.len());
        for token in &hidden {
            let mut token_logits = match &self.output {
                Some(output) => output.mul_vec(token)?,
                None => self.embeddings.as_logits(token)?,
            };
            if let Some(bias) = self.output_bias.as_deref() {
                add_vector_bias(&mut token_logits, bias)?;
            }
            logits.push(token_logits);
        }
        Ok(logits)
    }

    pub fn last_logits(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.forward(input_ids)?
            .pop()
            .ok_or_else(|| anyhow!("Qwen CPU reference produced no logits"))
    }

    pub fn greedy_next_token(&self, input_ids: &[u32]) -> Result<u32> {
        argmax(&self.last_logits(input_ids)?)
    }

    pub fn sample_next_token(
        &self,
        input_ids: &[u32],
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
        self.sample_next_token_with_options(input_ids, temperature, top_p, None, None)
    }

    pub fn sample_next_token_with_options(
        &self,
        input_ids: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        seed: Option<u64>,
    ) -> Result<u32> {
        let logits = self.last_logits(input_ids)?;
        if let Some(seed) = seed {
            let mut rng = StdRng::seed_from_u64(seed);
            sample_from_logits_with_rng(&logits, temperature, top_p, top_k, &mut rng)
        } else {
            let mut rng = rand::thread_rng();
            sample_from_logits_with_rng(&logits, temperature, top_p, top_k, &mut rng)
        }
    }

    pub fn generate_greedy(&self, input_ids: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        self.generate(input_ids, max_tokens, 0.0, 1.0)
    }

    pub fn generate(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        self.generate_with_options(input_ids, max_tokens, temperature, top_p, None, None)
    }

    pub fn generate_with_options(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        seed: Option<u64>,
    ) -> Result<Vec<u32>> {
        if let Some(seed) = seed {
            let mut rng = StdRng::seed_from_u64(seed);
            self.generate_with_rng(input_ids, max_tokens, temperature, top_p, top_k, &mut rng)
        } else {
            let mut rng = rand::thread_rng();
            self.generate_with_rng(input_ids, max_tokens, temperature, top_p, top_k, &mut rng)
        }
    }

    pub fn generate_with_rng<R: Rng + ?Sized>(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng: &mut R,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.last_logits(&tokens)?;
            let next = sample_from_logits_with_rng(&logits, temperature, top_p, top_k, rng)?;
            tokens.push(next);
            generated.push(next);
            if Some(next) == self.config.eos_token_id {
                break;
            }
        }
        Ok(generated)
    }

    pub fn run_tokens(
        &self,
        input_ids: &[u32],
        options: QwenCpuRunOptions,
    ) -> Result<QwenCpuRunOutput> {
        let logits = self.last_logits(input_ids)?;
        let next_token = argmax(&logits)?;
        let generated_tokens = self.generate_with_options(
            input_ids,
            options.max_tokens,
            options.temperature,
            options.top_p,
            None,
            options.seed,
        )?;
        let next_text = self.tokenizer.decode(&[next_token])?;
        let generated_text = self.tokenizer.decode(&generated_tokens)?;
        let top_logits = top_logits(&logits, self.tokenizer(), options.top_k)?;
        let logit_count = logits.len();
        Ok(QwenCpuRunOutput {
            backend: "cpu-reference",
            input_tokens: input_ids.to_vec(),
            next_token,
            next_text,
            generated_tokens,
            generated_text,
            top_logits,
            logit_count,
            logits: options.include_logits.then_some(logits),
        })
    }

    pub fn run_prompt(&self, prompt: &str, options: QwenCpuRunOptions) -> Result<QwenCpuRunOutput> {
        let tokens = self.tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            bail!("prompt encoded to zero tokens");
        }
        self.run_tokens(&tokens, options)
    }
}

#[derive(Debug)]
struct QwenLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attention: QwenLayerAttention,
    ffn: QwenFfn,
}

impl QwenLayer {
    fn load(gguf: &GgufFile, config: &QwenGgufConfig, idx: u32) -> Result<Self> {
        let prefix = format!("blk.{idx}");
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let attention = if qwen_ssm_layer_tensors_present(gguf, &prefix) {
            QwenLayerAttention::RecurrentSsm(QwenRecurrentSsm::load(gguf, config, &prefix)?)
        } else {
            QwenLayerAttention::Dense(QwenAttention::load(gguf, config, &prefix)?)
        };
        Ok(Self {
            attn_norm: load_vector_aliases(
                gguf,
                &qwen_dense_attention_norm_weight_names(&prefix),
                embed,
            )?,
            ffn_norm: load_vector_aliases(gguf, &qwen_dense_ffn_norm_weight_names(&prefix), embed)?,
            attention,
            ffn: QwenFfn::load(gguf, config, &prefix)?,
        })
    }

    fn forward(&self, mut hidden: Vec<Vec<f32>>, rms_eps: f32) -> Result<Vec<Vec<f32>>> {
        let mut attn_input = hidden.clone();
        for token in &mut attn_input {
            rms_norm_in_place(token, &self.attn_norm, rms_eps)?;
        }
        let attn_output = self.attention.forward(&attn_input, rms_eps)?;
        add_residual(&mut hidden, &attn_output)?;

        let mut mlp_input = hidden.clone();
        for token in &mut mlp_input {
            rms_norm_in_place(token, &self.ffn_norm, rms_eps)?;
        }
        let mlp_output = self.ffn.forward(&mlp_input)?;
        add_residual(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }
}

#[derive(Debug)]
enum QwenLayerAttention {
    Dense(QwenAttention),
    RecurrentSsm(QwenRecurrentSsm),
}

impl QwenLayerAttention {
    fn forward(&self, input: &[Vec<f32>], rms_eps: f32) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Dense(attention) => attention.forward(input, rms_eps),
            Self::RecurrentSsm(ssm) => ssm.forward(input, rms_eps),
        }
    }
}

#[derive(Debug)]
struct QwenRecurrentSsm {
    input: QwenSsmInputProjection,
    conv1d: Matrix,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    beta_alpha: QwenSsmBetaAlphaProjection,
    norm: Vec<f32>,
    out: Matrix,
    state_size: usize,
    time_step_rank: usize,
    group_count: usize,
    conv_kernel: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    qkvz_dim: usize,
    ba_dim: usize,
    /// Qwen3.5 pairs value head `h` with q/k group `h % group_count`
    /// (round-robin); Qwen3-Next uses block repeat `h / repeat`.
    kv_group_round_robin: bool,
}

#[derive(Debug)]
enum QwenSsmInputProjection {
    Legacy { qkvz: Matrix },
    Optimized { qkv: Matrix, gate: Matrix },
}

/// Beta/alpha (delta-rule strength / decay) projection. Qwen3-Next fuses both
/// into one `ssm_ba` matrix laid out `[beta(repeat) | alpha(repeat)]` per q/k
/// group; Qwen3.5 ships separate `ssm_beta` / `ssm_alpha` matrices with one
/// column per value head. Projected rows are `[beta(rank) | alpha(rank)]` for
/// the split form.
#[derive(Debug)]
enum QwenSsmBetaAlphaProjection {
    Fused { ba: Matrix },
    Split { beta: Matrix, alpha: Matrix },
}

impl QwenRecurrentSsm {
    fn load(gguf: &GgufFile, config: &QwenGgufConfig, prefix: &str) -> Result<Self> {
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let state_size = config
            .ssm_state_size
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.state_size does not fit usize")?
            .ok_or_else(|| anyhow!("Qwen recurrent SSM requires ssm.state_size"))?;
        let time_step_rank = config
            .ssm_time_step_rank
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.time_step_rank does not fit usize")?
            .ok_or_else(|| anyhow!("Qwen recurrent SSM requires ssm.time_step_rank"))?;
        let group_count = config
            .ssm_group_count
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.group_count does not fit usize")?
            .ok_or_else(|| anyhow!("Qwen recurrent SSM requires ssm.group_count"))?;
        let conv_kernel = config
            .ssm_conv_kernel
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.conv_kernel does not fit usize")?
            .ok_or_else(|| anyhow!("Qwen recurrent SSM requires ssm.conv_kernel"))?;
        let inner_size = config
            .ssm_inner_size
            .map(usize::try_from)
            .transpose()
            .context("qwen ssm.inner_size does not fit usize")?
            .ok_or_else(|| anyhow!("Qwen recurrent SSM requires ssm.inner_size"))?;
        if state_size == 0 || time_step_rank == 0 || group_count == 0 || conv_kernel == 0 {
            bail!("Qwen recurrent SSM metadata values must be non-zero");
        }
        if time_step_rank % group_count != 0 {
            bail!(
                "Qwen recurrent SSM time_step_rank {time_step_rank} must be divisible by group_count {group_count}"
            );
        }
        if inner_size % time_step_rank != 0 {
            bail!(
                "Qwen recurrent SSM inner_size {inner_size} must be divisible by time_step_rank {time_step_rank}"
            );
        }
        let head_v_dim = inner_size / time_step_rank;
        let key_dim = state_size
            .checked_mul(group_count)
            .context("Qwen recurrent SSM key dimension overflows usize")?;
        let value_dim = head_v_dim
            .checked_mul(time_step_rank)
            .context("Qwen recurrent SSM value dimension overflows usize")?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_dim))
            .context("Qwen recurrent SSM convolution dimension overflows usize")?;
        let qkvz_dim = key_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_dim.checked_mul(2)?))
            .context("Qwen recurrent SSM qkvz dimension overflows usize")?;
        let ba_dim = time_step_rank
            .checked_mul(2)
            .context("Qwen recurrent SSM beta/alpha dimension overflows usize")?;

        let input = if qwen_ssm_in_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        {
            QwenSsmInputProjection::Legacy {
                qkvz: load_matrix_aliases(
                    gguf,
                    &qwen_ssm_in_weight_names(prefix),
                    qkvz_dim,
                    embed,
                )?,
            }
        } else {
            QwenSsmInputProjection::Optimized {
                qkv: load_matrix_aliases(
                    gguf,
                    &qwen_ssm_qkv_weight_names(prefix),
                    conv_dim,
                    embed,
                )?,
                gate: load_matrix_aliases(
                    gguf,
                    &qwen_ssm_gate_weight_names(prefix),
                    value_dim,
                    embed,
                )?,
            }
        };

        let beta_alpha = if qwen_ssm_ba_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        {
            QwenSsmBetaAlphaProjection::Fused {
                ba: load_matrix_aliases(gguf, &qwen_ssm_ba_weight_names(prefix), ba_dim, embed)?,
            }
        } else {
            QwenSsmBetaAlphaProjection::Split {
                beta: load_matrix_aliases(
                    gguf,
                    &qwen_ssm_beta_weight_names(prefix),
                    time_step_rank,
                    embed,
                )?,
                alpha: load_matrix_aliases(
                    gguf,
                    &qwen_ssm_alpha_weight_names(prefix),
                    time_step_rank,
                    embed,
                )?,
            }
        };
        Ok(Self {
            input,
            conv1d: load_matrix_aliases(
                gguf,
                &qwen_ssm_conv1d_weight_names(prefix),
                conv_dim,
                conv_kernel,
            )?,
            dt_bias: load_vector_aliases(gguf, &qwen_ssm_dt_bias_names(prefix), time_step_rank)?,
            a: load_vector_aliases(gguf, &qwen_ssm_a_names(prefix), time_step_rank)?,
            beta_alpha,
            norm: load_vector_aliases(gguf, &qwen_ssm_norm_weight_names(prefix), head_v_dim)?,
            out: load_matrix_aliases(gguf, &qwen_ssm_out_weight_names(prefix), embed, value_dim)?,
            state_size,
            time_step_rank,
            group_count,
            conv_kernel,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            qkvz_dim,
            ba_dim,
            kv_group_round_robin: config.ssm_kv_group_round_robin(),
        })
    }

    fn forward(&self, input: &[Vec<f32>], rms_eps: f32) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            bail!("Qwen recurrent SSM requires at least one input token");
        }
        let mut mixed_qkv = Vec::with_capacity(input.len() * self.conv_dim);
        let mut z = Vec::with_capacity(input.len() * self.value_dim);
        let mut ba = Vec::with_capacity(input.len() * self.ba_dim);
        for token in input {
            let (qkv_token, z_token) = match &self.input {
                QwenSsmInputProjection::Legacy { qkvz } => {
                    let projected = qkvz.mul_vec(token)?;
                    self.split_qkvz(&projected)?
                }
                QwenSsmInputProjection::Optimized { qkv, gate } => {
                    let qkv_token = qkv.mul_vec(token)?;
                    if qkv_token.len() != self.conv_dim {
                        bail!(
                            "Qwen recurrent SSM qkv projection length {} does not match conv_dim {}",
                            qkv_token.len(),
                            self.conv_dim
                        );
                    }
                    let z_token = gate.mul_vec(token)?;
                    if z_token.len() != self.value_dim {
                        bail!(
                            "Qwen recurrent SSM gate projection length {} does not match value_dim {}",
                            z_token.len(),
                            self.value_dim
                        );
                    }
                    (qkv_token, z_token)
                }
            };
            mixed_qkv.extend(qkv_token);
            z.extend(z_token);
            let ba_token = match &self.beta_alpha {
                QwenSsmBetaAlphaProjection::Fused { ba } => ba.mul_vec(token)?,
                QwenSsmBetaAlphaProjection::Split { beta, alpha } => {
                    let mut projected = beta.mul_vec(token)?;
                    projected.extend(alpha.mul_vec(token)?);
                    projected
                }
            };
            if ba_token.len() != self.ba_dim {
                bail!(
                    "Qwen recurrent SSM ba projection length {} does not match ba_dim {}",
                    ba_token.len(),
                    self.ba_dim
                );
            }
            ba.extend(ba_token);
        }

        let conv = self.depthwise_conv(&mixed_qkv, input.len())?;
        let core = self.gated_delta(&conv, &ba, input.len())?;
        let normed = self.gated_rms_norm(&core, &z, input.len(), rms_eps)?;
        normed
            .chunks(self.value_dim)
            .map(|token| self.out.mul_vec(token))
            .collect()
    }

    fn split_qkvz(&self, qkvz: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if qkvz.len() != self.qkvz_dim {
            bail!(
                "Qwen recurrent SSM qkvz projection length {} does not match qkvz_dim {}",
                qkvz.len(),
                self.qkvz_dim
            );
        }
        let repeat = self.time_step_rank / self.group_count;
        let value_group_dim = repeat
            .checked_mul(self.head_v_dim)
            .context("Qwen recurrent SSM grouped value dimension overflows usize")?;
        let source_group_dim = self
            .state_size
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_group_dim.checked_mul(2)?))
            .context("Qwen recurrent SSM grouped qkvz dimension overflows usize")?;
        let mut mixed_qkv = vec![0.0; self.conv_dim];
        let mut z = vec![0.0; self.value_dim];
        for group in 0..self.group_count {
            let mut source = group * source_group_dim;
            let q_dest = group * self.state_size;
            mixed_qkv[q_dest..q_dest + self.state_size]
                .copy_from_slice(&qkvz[source..source + self.state_size]);
            source += self.state_size;
            let k_dest = self.key_dim + group * self.state_size;
            mixed_qkv[k_dest..k_dest + self.state_size]
                .copy_from_slice(&qkvz[source..source + self.state_size]);
            source += self.state_size;
            let value_dest = 2 * self.key_dim + group * value_group_dim;
            mixed_qkv[value_dest..value_dest + value_group_dim]
                .copy_from_slice(&qkvz[source..source + value_group_dim]);
            source += value_group_dim;
            let z_dest = group * value_group_dim;
            z[z_dest..z_dest + value_group_dim]
                .copy_from_slice(&qkvz[source..source + value_group_dim]);
        }
        Ok((mixed_qkv, z))
    }

    fn depthwise_conv(&self, mixed_qkv: &[f32], rows: usize) -> Result<Vec<f32>> {
        let expected = rows
            .checked_mul(self.conv_dim)
            .context("Qwen recurrent SSM conv input length overflows usize")?;
        if mixed_qkv.len() != expected {
            bail!(
                "Qwen recurrent SSM conv input length {} does not match expected {expected}",
                mixed_qkv.len()
            );
        }
        let mut output = vec![0.0; expected];
        for row in 0..rows {
            for channel in 0..self.conv_dim {
                let mut sum = 0.0f32;
                for kernel in 0..self.conv_kernel {
                    let input_row = row as isize + kernel as isize + 1 - self.conv_kernel as isize;
                    if input_row < 0 {
                        continue;
                    }
                    let input_row = usize::try_from(input_row)
                        .context("Qwen recurrent SSM conv row does not fit usize")?;
                    sum += self.conv1d.data[channel * self.conv_kernel + kernel]
                        * mixed_qkv[input_row * self.conv_dim + channel];
                }
                output[row * self.conv_dim + channel] = silu(sum);
            }
        }
        Ok(output)
    }

    fn gated_delta(&self, conv: &[f32], ba: &[f32], rows: usize) -> Result<Vec<f32>> {
        let expected_conv = rows
            .checked_mul(self.conv_dim)
            .context("Qwen recurrent SSM delta input length overflows usize")?;
        if conv.len() != expected_conv {
            bail!(
                "Qwen recurrent SSM delta input length {} does not match expected {expected_conv}",
                conv.len()
            );
        }
        let expected_ba = rows
            .checked_mul(self.ba_dim)
            .context("Qwen recurrent SSM ba input length overflows usize")?;
        if ba.len() != expected_ba {
            bail!(
                "Qwen recurrent SSM ba input length {} does not match expected {expected_ba}",
                ba.len()
            );
        }
        let mut query = vec![0.0; rows * self.key_dim];
        let mut key = vec![0.0; rows * self.key_dim];
        let mut value = vec![0.0; rows * self.value_dim];
        for row in 0..rows {
            let source = row * self.conv_dim;
            query[row * self.key_dim..(row + 1) * self.key_dim]
                .copy_from_slice(&conv[source..source + self.key_dim]);
            key[row * self.key_dim..(row + 1) * self.key_dim]
                .copy_from_slice(&conv[source + self.key_dim..source + 2 * self.key_dim]);
            value[row * self.value_dim..(row + 1) * self.value_dim]
                .copy_from_slice(&conv[source + 2 * self.key_dim..source + self.conv_dim]);
        }
        for row in 0..rows {
            for group in 0..self.group_count {
                let start = row * self.key_dim + group * self.state_size;
                l2_normalize(&mut query[start..start + self.state_size]);
                l2_normalize(&mut key[start..start + self.state_size]);
            }
        }

        let repeat = self.time_step_rank / self.group_count;
        let group_ba_dim = repeat
            .checked_mul(2)
            .context("Qwen recurrent SSM grouped ba dimension overflows usize")?;
        let q_scale = (self.state_size as f32).sqrt().recip();
        let mut state = vec![0.0; self.time_step_rank * self.state_size * self.head_v_dim];
        let mut output = vec![0.0; rows * self.value_dim];
        let mut kv_mem = vec![0.0; self.head_v_dim];
        let mut delta = vec![0.0; self.head_v_dim];
        for row in 0..rows {
            for head in 0..self.time_step_rank {
                let group = if self.kv_group_round_robin {
                    head % self.group_count
                } else {
                    head / repeat
                };
                let q_start = row * self.key_dim + group * self.state_size;
                let k_start = row * self.key_dim + group * self.state_size;
                let v_start = row * self.value_dim + head * self.head_v_dim;
                let (beta_raw, alpha_raw) = match &self.beta_alpha {
                    QwenSsmBetaAlphaProjection::Fused { .. } => {
                        // Fused rows are `[beta(repeat) | alpha(repeat)]` per
                        // block group regardless of the q/k pairing above.
                        let ba_group = row * self.ba_dim + (head / repeat) * group_ba_dim;
                        (
                            ba[ba_group + head % repeat],
                            ba[ba_group + repeat + head % repeat],
                        )
                    }
                    QwenSsmBetaAlphaProjection::Split { .. } => {
                        let base = row * self.ba_dim;
                        (ba[base + head], ba[base + self.time_step_rank + head])
                    }
                };
                let beta = sigmoid(beta_raw);
                let alpha = alpha_raw;
                let decay = (-self.a[head].exp() * softplus(alpha + self.dt_bias[head])).exp();
                let state_start = head * self.state_size * self.head_v_dim;

                for state_dim in 0..self.state_size {
                    for value_dim in 0..self.head_v_dim {
                        state[state_start + state_dim * self.head_v_dim + value_dim] *= decay;
                    }
                }
                kv_mem.fill(0.0);
                for state_dim in 0..self.state_size {
                    let key_value = key[k_start + state_dim];
                    for value_dim in 0..self.head_v_dim {
                        kv_mem[value_dim] += state
                            [state_start + state_dim * self.head_v_dim + value_dim]
                            * key_value;
                    }
                }
                for value_dim in 0..self.head_v_dim {
                    delta[value_dim] = (value[v_start + value_dim] - kv_mem[value_dim]) * beta;
                }
                for state_dim in 0..self.state_size {
                    let key_value = key[k_start + state_dim];
                    for value_dim in 0..self.head_v_dim {
                        state[state_start + state_dim * self.head_v_dim + value_dim] +=
                            key_value * delta[value_dim];
                    }
                }
                for value_dim in 0..self.head_v_dim {
                    let mut sum = 0.0f32;
                    for state_dim in 0..self.state_size {
                        sum += state[state_start + state_dim * self.head_v_dim + value_dim]
                            * query[q_start + state_dim]
                            * q_scale;
                    }
                    output[v_start + value_dim] = sum;
                }
            }
        }
        Ok(output)
    }

    fn gated_rms_norm(
        &self,
        core: &[f32],
        z: &[f32],
        rows: usize,
        rms_eps: f32,
    ) -> Result<Vec<f32>> {
        let expected = rows
            .checked_mul(self.value_dim)
            .context("Qwen recurrent SSM norm input length overflows usize")?;
        if core.len() != expected || z.len() != expected {
            bail!(
                "Qwen recurrent SSM norm got core/z lengths {}/{}; expected {expected}",
                core.len(),
                z.len()
            );
        }
        let mut output = vec![0.0; expected];
        for row in 0..rows {
            for head in 0..self.time_step_rank {
                let start = row * self.value_dim + head * self.head_v_dim;
                let mut variance = 0.0f32;
                for value_dim in 0..self.head_v_dim {
                    let value = core[start + value_dim];
                    variance += value * value;
                }
                let scale = (variance / self.head_v_dim as f32 + rms_eps).sqrt().recip();
                for value_dim in 0..self.head_v_dim {
                    output[start + value_dim] = core[start + value_dim]
                        * scale
                        * self.norm[value_dim]
                        * silu(z[start + value_dim]);
                }
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct QwenAttention {
    projection: QwenAttentionProjection,
    o: Matrix,
    o_bias: Option<Vec<f32>>,
    heads: usize,
    kv_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    rope_base: f32,
    rope_scale: f32,
    split_half_rope: bool,
    /// Rotary dims per head; less than `qk_head_dim` for partial rope
    /// (`rope.dimension_count`, e.g. Qwen3.5 rotates 64 of 256 dims).
    rope_rot_dim: usize,
}

#[derive(Debug)]
enum QwenAttentionProjection {
    Dense {
        q: QwenDenseQueryProjection,
        k: Matrix,
        v: Matrix,
        k_bias: Option<Vec<f32>>,
        v_bias: Option<Vec<f32>>,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
    },
    Mla {
        q_a: Matrix,
        q_a_norm: Vec<f32>,
        q_b: Matrix,
        kv_a: Matrix,
        kv_a_norm: Vec<f32>,
        kv_b: Matrix,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
    },
}

#[derive(Debug)]
enum QwenDenseQueryProjection {
    Plain {
        q: Matrix,
        q_bias: Option<Vec<f32>>,
    },
    Gated {
        q_gate: Matrix,
        q_gate_bias: Option<Vec<f32>>,
    },
}

impl QwenAttention {
    fn load(gguf: &GgufFile, config: &QwenGgufConfig, prefix: &str) -> Result<Self> {
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let heads = usize::try_from(config.attention_head_count)
            .context("qwen attention.head_count does not fit usize")?;
        let metadata_kv_heads = usize::try_from(config.attention_head_count_kv)
            .context("qwen attention.head_count_kv does not fit usize")?;
        let uses_mla = qwen_mla_attention_tensors_present(gguf, prefix);
        if heads == 0 || metadata_kv_heads == 0 {
            bail!("qwen attention heads and kv heads must be non-zero");
        }
        if !uses_mla && heads % metadata_kv_heads != 0 {
            bail!(
                "qwen attention heads {heads} must be a non-zero multiple of kv heads {metadata_kv_heads}"
            );
        }
        let kv_heads = if uses_mla { heads } else { metadata_kv_heads };
        let qk_head_dim = config
            .attention_key_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention key head dimension does not fit usize")?
            .ok_or_else(|| {
                anyhow!(
                    "qwen attention key length is incompatible with embedding length {embed} and attention heads {heads}"
                )
            })?;
        let v_head_dim = config
            .attention_value_head_dim()
            .map(usize::try_from)
            .transpose()
            .context("qwen attention value head dimension does not fit usize")?
            .ok_or_else(|| {
                anyhow!(
                    "qwen attention value length is incompatible with embedding length {embed} and attention heads {heads}"
                )
            })?;
        let q_dim = qk_head_dim
            .checked_mul(heads)
            .context("qwen attention q dimension overflows usize")?;
        let k_dim = qk_head_dim
            .checked_mul(kv_heads)
            .context("qwen attention k dimension overflows usize")?;
        let v_dim = v_head_dim
            .checked_mul(kv_heads)
            .context("qwen attention v dimension overflows usize")?;
        let attention_output_dim = v_head_dim
            .checked_mul(heads)
            .context("qwen attention output dimension overflows usize")?;
        let projection = if uses_mla {
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
            if q_lora_rank == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0 {
                bail!(
                    "MLA tensor layout requires non-zero q_lora_rank, kv_lora_rank, and qk_rope_head_dim"
                );
            }
            let expected_qk_head_dim = qk_nope_head_dim
                .checked_add(qk_rope_head_dim)
                .context("MLA qk head dimension overflows usize")?;
            if expected_qk_head_dim != qk_head_dim {
                bail!(
                    "MLA qk head dimensions qk_nope_head_dim {qk_nope_head_dim} + qk_rope_head_dim {qk_rope_head_dim} do not match effective head dim {qk_head_dim}"
                );
            }
            let kv_a_dim = kv_lora_rank
                .checked_add(qk_rope_head_dim)
                .context("MLA kv_a dimension overflows usize")?;
            let kv_b_dim = heads
                .checked_mul(
                    qk_nope_head_dim
                        .checked_add(v_head_dim)
                        .context("MLA kv_b per-head dimension overflows usize")?,
                )
                .context("MLA kv_b dimension overflows usize")?;
            QwenAttentionProjection::Mla {
                q_a: load_matrix_aliases(
                    gguf,
                    &qwen_mla_q_a_weight_names(prefix),
                    q_lora_rank,
                    embed,
                )?,
                q_a_norm: load_vector_aliases(
                    gguf,
                    &qwen_mla_q_a_norm_weight_names(prefix),
                    q_lora_rank,
                )?,
                q_b: load_matrix_aliases(
                    gguf,
                    &qwen_mla_q_b_weight_names(prefix),
                    q_dim,
                    q_lora_rank,
                )?,
                kv_a: load_matrix_aliases(
                    gguf,
                    &qwen_mla_kv_a_weight_names(prefix),
                    kv_a_dim,
                    embed,
                )?,
                kv_a_norm: load_vector_aliases(
                    gguf,
                    &qwen_mla_kv_a_norm_weight_names(prefix),
                    kv_lora_rank,
                )?,
                kv_b: load_matrix_aliases(
                    gguf,
                    &qwen_mla_kv_b_weight_names(prefix),
                    kv_b_dim,
                    kv_lora_rank,
                )?,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
            }
        } else {
            QwenAttentionProjection::Dense {
                q: load_dense_query_projection(gguf, config, prefix, q_dim, embed)?,
                k: load_attention_matrix(gguf, config, prefix, "k", q_dim, k_dim, embed)?,
                v: load_attention_matrix(gguf, config, prefix, "v", q_dim + k_dim, v_dim, embed)?,
                k_bias: optional_attention_bias(gguf, config, prefix, "k", q_dim, k_dim)?,
                v_bias: optional_attention_bias(gguf, config, prefix, "v", q_dim + k_dim, v_dim)?,
                q_norm: optional_vector_aliases(
                    gguf,
                    &qwen_dense_attention_head_norm_weight_names(prefix, "q"),
                    qk_head_dim,
                )?,
                k_norm: optional_vector_aliases(
                    gguf,
                    &qwen_dense_attention_head_norm_weight_names(prefix, "k"),
                    qk_head_dim,
                )?,
            }
        };

        Ok(Self {
            projection,
            o: load_matrix_aliases(
                gguf,
                &qwen_dense_attention_weight_names(prefix, "output"),
                embed,
                attention_output_dim,
            )?,
            o_bias: optional_vector_aliases(
                gguf,
                &qwen_dense_attention_bias_names(prefix, "output"),
                embed,
            )?,
            heads,
            kv_heads,
            qk_head_dim,
            v_head_dim,
            rope_base: config
                .rope_freq_base
                .unwrap_or_else(|| config.default_rope_freq_base()),
            rope_scale: config.rope_freq_scale.unwrap_or(1.0),
            split_half_rope: true,
            rope_rot_dim: config.rope_rot_dim(qk_head_dim),
        })
    }

    fn forward(&self, input: &[Vec<f32>], rms_eps: f32) -> Result<Vec<Vec<f32>>> {
        let seq_len = input.len();
        let embed = self.o.rows;
        let mut q = Vec::with_capacity(seq_len);
        let mut k = Vec::with_capacity(seq_len);
        let mut v = Vec::with_capacity(seq_len);
        let mut gates: Option<Vec<Vec<f32>>> = None;

        for (position, token) in input.iter().enumerate() {
            let (q_token, k_token, v_token, gate_token) = match &self.projection {
                QwenAttentionProjection::Dense {
                    q,
                    k,
                    v,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                } => {
                    let (mut q_token, gate_token) = match q {
                        QwenDenseQueryProjection::Plain { q, q_bias } => {
                            (q.mul_vec_with_bias(token, q_bias.as_deref())?, None)
                        }
                        QwenDenseQueryProjection::Gated {
                            q_gate,
                            q_gate_bias,
                        } => {
                            let q_gate_token =
                                q_gate.mul_vec_with_bias(token, q_gate_bias.as_deref())?;
                            let (q_token, gate_token) = split_gated_q_projection(
                                &q_gate_token,
                                self.heads,
                                self.qk_head_dim,
                            )?;
                            (q_token, Some(gate_token))
                        }
                    };
                    let mut k_token = k.mul_vec_with_bias(token, k_bias.as_deref())?;
                    let v_token = v.mul_vec_with_bias(token, v_bias.as_deref())?;

                    for head in 0..self.heads {
                        let range = head * self.qk_head_dim..(head + 1) * self.qk_head_dim;
                        if let Some(weight) = q_norm {
                            rms_norm_in_place(&mut q_token[range.clone()], weight, rms_eps)?;
                        }
                        let rot = range.start..range.start + self.rope_rot_dim;
                        apply_rope(
                            &mut q_token[rot],
                            position,
                            self.rope_base,
                            self.rope_scale,
                            self.split_half_rope,
                        )?;
                    }
                    for head in 0..self.kv_heads {
                        let range = head * self.qk_head_dim..(head + 1) * self.qk_head_dim;
                        if let Some(weight) = k_norm {
                            rms_norm_in_place(&mut k_token[range.clone()], weight, rms_eps)?;
                        }
                        let rot = range.start..range.start + self.rope_rot_dim;
                        apply_rope(
                            &mut k_token[rot],
                            position,
                            self.rope_base,
                            self.rope_scale,
                            self.split_half_rope,
                        )?;
                    }
                    (q_token, k_token, v_token, gate_token)
                }
                QwenAttentionProjection::Mla {
                    q_a,
                    q_a_norm,
                    q_b,
                    kv_a,
                    kv_a_norm,
                    kv_b,
                    q_lora_rank,
                    kv_lora_rank,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                } => {
                    let q_lora_rank = *q_lora_rank;
                    let kv_lora_rank = *kv_lora_rank;
                    let qk_nope_head_dim = *qk_nope_head_dim;
                    let qk_rope_head_dim = *qk_rope_head_dim;
                    let mut q_latent = q_a.mul_vec(token)?;
                    if q_latent.len() != q_lora_rank {
                        bail!(
                            "MLA q latent length {} does not match q_lora_rank {}",
                            q_latent.len(),
                            q_lora_rank
                        );
                    }
                    rms_norm_in_place(&mut q_latent, q_a_norm, rms_eps)?;
                    let mut q_token = q_b.mul_vec(&q_latent)?;
                    for head in 0..self.heads {
                        let rope_start = head * self.qk_head_dim + qk_nope_head_dim;
                        let rope_end = rope_start + qk_rope_head_dim;
                        apply_rope(
                            &mut q_token[rope_start..rope_end],
                            position,
                            self.rope_base,
                            self.rope_scale,
                            self.split_half_rope,
                        )?;
                    }

                    let kv_projected = kv_a.mul_vec(token)?;
                    if kv_projected.len() != kv_lora_rank + qk_rope_head_dim {
                        bail!(
                            "MLA kv_a output length {} does not match kv_lora_rank {} + qk_rope_head_dim {}",
                            kv_projected.len(),
                            kv_lora_rank,
                            qk_rope_head_dim
                        );
                    }
                    let mut kv_latent = kv_projected[..kv_lora_rank].to_vec();
                    let k_pe = kv_projected[kv_lora_rank..].to_vec();
                    rms_norm_in_place(&mut kv_latent, kv_a_norm, rms_eps)?;
                    let kv_token = kv_b.mul_vec(&kv_latent)?;
                    let mut k_token = vec![0.0; self.heads * self.qk_head_dim];
                    let mut v_token = vec![0.0; self.heads * self.v_head_dim];
                    let kv_b_head_dim = qk_nope_head_dim
                        .checked_add(self.v_head_dim)
                        .context("MLA kv_b per-head dimension overflows usize")?;
                    for head in 0..self.heads {
                        let kv_start = head * kv_b_head_dim;
                        let k_start = head * self.qk_head_dim;
                        let v_start = head * self.v_head_dim;
                        k_token[k_start..k_start + qk_nope_head_dim]
                            .copy_from_slice(&kv_token[kv_start..kv_start + qk_nope_head_dim]);
                        let mut k_rope = k_pe.clone();
                        apply_rope(
                            &mut k_rope,
                            position,
                            self.rope_base,
                            self.rope_scale,
                            self.split_half_rope,
                        )?;
                        k_token[k_start + qk_nope_head_dim
                            ..k_start + qk_nope_head_dim + qk_rope_head_dim]
                            .copy_from_slice(&k_rope);
                        v_token[v_start..v_start + self.v_head_dim].copy_from_slice(
                            &kv_token[kv_start + qk_nope_head_dim
                                ..kv_start + qk_nope_head_dim + self.v_head_dim],
                        );
                    }
                    (q_token, k_token, v_token, None)
                }
            };

            if let Some(gate_token) = gate_token {
                gates.get_or_insert_with(Vec::new).push(gate_token);
            }
            q.push(q_token);
            k.push(k_token);
            v.push(v_token);
        }

        let mut output = vec![vec![0.0; embed]; seq_len];
        let kv_repeats = self.heads / self.kv_heads;
        let scale = (self.qk_head_dim as f32).powf(-0.5);
        let attention_output_dim = self.heads * self.v_head_dim;
        for target in 0..seq_len {
            let mut joined = vec![0.0; attention_output_dim];
            for head in 0..self.heads {
                let kv_head = head / kv_repeats;
                let q_start = head * self.qk_head_dim;
                let k_start = kv_head * self.qk_head_dim;
                let out_start = head * self.v_head_dim;
                let v_start = kv_head * self.v_head_dim;
                let q_vec = &q[target][q_start..q_start + self.qk_head_dim];
                let mut scores = Vec::with_capacity(target + 1);
                for source in 0..=target {
                    let k_vec = &k[source][k_start..k_start + self.qk_head_dim];
                    scores.push(dot(q_vec, k_vec) * scale);
                }
                softmax_in_place(&mut scores);
                for (source, score) in scores.into_iter().enumerate() {
                    let v_vec = &v[source][v_start..v_start + self.v_head_dim];
                    for dim in 0..self.v_head_dim {
                        joined[out_start + dim] += score * v_vec[dim];
                    }
                }
            }
            if let Some(gates) = gates.as_ref() {
                let gate = gates.get(target).ok_or_else(|| {
                    anyhow!("Qwen gated attention missing gate for token {target}")
                })?;
                if gate.len() != joined.len() {
                    bail!(
                        "Qwen gated attention gate length {} does not match attention output length {}",
                        gate.len(),
                        joined.len()
                    );
                }
                for (value, gate) in joined.iter_mut().zip(gate) {
                    *value *= sigmoid(*gate);
                }
            }
            output[target] = self.o.mul_vec_with_bias(&joined, self.o_bias.as_deref())?;
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct QwenMlp {
    gate: Matrix,
    gate_bias: Option<Vec<f32>>,
    up: Matrix,
    up_bias: Option<Vec<f32>>,
    down: Matrix,
    down_bias: Option<Vec<f32>>,
}

impl QwenMlp {
    fn load(gguf: &GgufFile, config: &QwenGgufConfig, prefix: &str) -> Result<Self> {
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let ff = config
            .feed_forward_length
            .map(usize::try_from)
            .transpose()
            .context("qwen feed_forward_length does not fit usize")?
            .ok_or_else(|| anyhow!("qwen metadata missing feed_forward_length"))?;
        Ok(Self {
            gate: load_ffn_gate_matrix(gguf, prefix, true, ff, embed)?,
            gate_bias: optional_ffn_gate_bias(gguf, prefix, true, ff)?,
            up: load_ffn_gate_matrix(gguf, prefix, false, ff, embed)?,
            up_bias: optional_ffn_gate_bias(gguf, prefix, false, ff)?,
            down: load_matrix_aliases(
                gguf,
                &qwen_dense_ffn_weight_names(prefix, "down"),
                embed,
                ff,
            )?,
            down_bias: optional_vector_aliases(
                gguf,
                &qwen_dense_ffn_bias_names(prefix, "down"),
                embed,
            )?,
        })
    }

    fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(input.len());
        for token in input {
            let gate = self
                .gate
                .mul_vec_with_bias(token, self.gate_bias.as_deref())?;
            let up = self.up.mul_vec_with_bias(token, self.up_bias.as_deref())?;
            let hidden = gate
                .iter()
                .zip(up.iter())
                .map(|(gate, up)| silu(*gate) * up)
                .collect::<Vec<_>>();
            output.push(
                self.down
                    .mul_vec_with_bias(&hidden, self.down_bias.as_deref())?,
            );
        }
        Ok(output)
    }
}

#[derive(Debug)]
enum QwenFfn {
    Dense(QwenMlp),
    Moe(QwenMoe),
}

impl QwenFfn {
    fn load(gguf: &GgufFile, config: &QwenGgufConfig, prefix: &str) -> Result<Self> {
        if config.expert_count.is_some()
            && qwen_moe_router_weight_names(prefix)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
        {
            Ok(Self::Moe(QwenMoe::load(gguf, config, prefix)?))
        } else {
            Ok(Self::Dense(QwenMlp::load(gguf, config, prefix)?))
        }
    }

    fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Dense(mlp) => mlp.forward(input),
            Self::Moe(moe) => moe.forward(input),
        }
    }
}

#[derive(Debug)]
struct QwenMoe {
    router: Matrix,
    router_bias: Option<Vec<f32>>,
    experts: Vec<QwenMlp>,
    shared: Option<QwenMlp>,
    shared_gate: Option<Matrix>,
    shared_gate_bias: Option<Vec<f32>>,
    top_k: usize,
    norm_topk_prob: bool,
}

impl QwenMoe {
    fn load(gguf: &GgufFile, config: &QwenGgufConfig, prefix: &str) -> Result<Self> {
        let embed = usize::try_from(config.embedding_length)
            .context("qwen embedding_length does not fit usize")?;
        let experts = config
            .expert_count
            .map(usize::try_from)
            .transpose()
            .context("qwen expert_count does not fit usize")?
            .ok_or_else(|| anyhow!("qwen MoE metadata missing expert_count"))?;
        let top_k = config
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
        let ff = expert_ff_dim(gguf, config, prefix, experts, embed)?;
        let use_per_expert_tensors = !moe_packed_expert_tensors_complete(gguf, prefix)
            && moe_per_expert_tensors_complete(gguf, prefix, experts);
        let mut expert_mlps = Vec::with_capacity(experts);
        for expert in 0..experts {
            expert_mlps.push(QwenMlp {
                gate: load_moe_expert_gate_up_matrix(
                    gguf,
                    prefix,
                    use_per_expert_tensors,
                    expert,
                    experts,
                    ff,
                    embed,
                    true,
                )?,
                gate_bias: optional_moe_expert_gate_up_bias(
                    gguf,
                    prefix,
                    use_per_expert_tensors,
                    expert,
                    experts,
                    ff,
                    true,
                )?,
                up: load_moe_expert_gate_up_matrix(
                    gguf,
                    prefix,
                    use_per_expert_tensors,
                    expert,
                    experts,
                    ff,
                    embed,
                    false,
                )?,
                up_bias: optional_moe_expert_gate_up_bias(
                    gguf,
                    prefix,
                    use_per_expert_tensors,
                    expert,
                    experts,
                    ff,
                    false,
                )?,
                down: load_moe_expert_matrix(
                    gguf,
                    &qwen_moe_packed_expert_weight_names(prefix, "down"),
                    &qwen_moe_per_expert_weight_names(prefix, "down", expert as u64),
                    use_per_expert_tensors,
                    expert,
                    experts,
                    embed,
                    ff,
                )?,
                down_bias: optional_moe_expert_bias(
                    gguf,
                    prefix,
                    use_per_expert_tensors,
                    expert,
                    experts,
                    "down",
                    embed,
                )?,
            });
        }
        let shared_gate = qwen_moe_shared_expert_weight_names(prefix, "gate");
        let shared_up = qwen_moe_shared_expert_weight_names(prefix, "up");
        let shared_down = qwen_moe_shared_expert_weight_names(prefix, "down");
        let shared = if let Some(source) =
            moe_shared_expert_packed_gate_up_source(gguf, prefix, ff, embed)?
        {
            Some(QwenMlp {
                gate: load_packed_matrix_rows(
                    gguf,
                    &source.name,
                    source.gate_offset,
                    ff,
                    embed,
                    source.source_rows,
                )?,
                gate_bias: optional_moe_shared_expert_gate_up_bias(gguf, prefix, ff, true)?,
                up: load_packed_matrix_rows(
                    gguf,
                    &source.name,
                    source.up_offset,
                    ff,
                    embed,
                    source.source_rows,
                )?,
                up_bias: optional_moe_shared_expert_gate_up_bias(gguf, prefix, ff, false)?,
                down: load_matrix_aliases(gguf, &shared_down, embed, ff)?,
                down_bias: optional_vector_aliases(
                    gguf,
                    &qwen_moe_shared_expert_bias_names(prefix, "down"),
                    embed,
                )?,
            })
        } else if shared_gate.iter().any(|name| gguf.tensor(name).is_some()) {
            Some(QwenMlp {
                gate: load_matrix_aliases(gguf, &shared_gate, ff, embed)?,
                gate_bias: optional_vector_aliases(
                    gguf,
                    &qwen_moe_shared_expert_bias_names(prefix, "gate"),
                    ff,
                )?,
                up: load_matrix_aliases(gguf, &shared_up, ff, embed)?,
                up_bias: optional_vector_aliases(
                    gguf,
                    &qwen_moe_shared_expert_bias_names(prefix, "up"),
                    ff,
                )?,
                down: load_matrix_aliases(gguf, &shared_down, embed, ff)?,
                down_bias: optional_vector_aliases(
                    gguf,
                    &qwen_moe_shared_expert_bias_names(prefix, "down"),
                    embed,
                )?,
            })
        } else {
            None
        };
        let shared_gate_names = qwen_moe_shared_expert_gate_weight_names(prefix);
        let shared_gate = if shared.is_some()
            && shared_gate_names
                .iter()
                .any(|name| gguf.tensor(name).is_some())
        {
            Some(load_matrix_aliases(gguf, &shared_gate_names, 1, embed)?)
        } else {
            None
        };
        let shared_gate_bias = if shared_gate.is_some() {
            optional_vector_aliases(gguf, &qwen_moe_shared_expert_gate_bias_names(prefix), 1)?
        } else {
            None
        };
        Ok(Self {
            router: load_matrix_aliases(
                gguf,
                &qwen_moe_router_weight_names(prefix),
                experts,
                embed,
            )?,
            router_bias: optional_vector_aliases(
                gguf,
                &qwen_moe_router_bias_names(prefix),
                experts,
            )?,
            experts: expert_mlps,
            shared,
            shared_gate,
            shared_gate_bias,
            top_k,
            norm_topk_prob: config.expert_weights_norm,
        })
    }

    fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut outputs = Vec::with_capacity(input.len());
        for token in input {
            let logits = self
                .router
                .mul_vec_with_bias(token, self.router_bias.as_deref())?;
            let mut scores = logits;
            softmax_in_place(&mut scores);
            let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            ranked.truncate(self.top_k.min(ranked.len()));
            if self.norm_topk_prob && ranked.len() > 1 {
                let denom = ranked.iter().map(|(_, score)| *score).sum::<f32>();
                if denom > f32::EPSILON {
                    for (_, score) in &mut ranked {
                        *score /= denom;
                    }
                }
            }
            let mut acc = vec![0.0; token.len()];
            for (expert, score) in ranked {
                let expert_out = self.experts[expert].forward(std::slice::from_ref(token))?;
                for (acc, value) in acc.iter_mut().zip(&expert_out[0]) {
                    *acc += score * value;
                }
            }
            if let Some(shared) = &self.shared {
                let shared_out = shared.forward(std::slice::from_ref(token))?;
                let shared_scale = if let Some(shared_gate) = &self.shared_gate {
                    let gate =
                        shared_gate.mul_vec_with_bias(token, self.shared_gate_bias.as_deref())?;
                    if gate.len() != 1 {
                        bail!(
                            "Qwen MoE shared expert gate length {} does not match expected scalar",
                            gate.len()
                        );
                    }
                    sigmoid(gate[0])
                } else {
                    1.0
                };
                for (acc, value) in acc.iter_mut().zip(&shared_out[0]) {
                    *acc += shared_scale * value;
                }
            }
            outputs.push(acc);
        }
        Ok(outputs)
    }
}

#[derive(Debug)]
struct EmbeddingTable {
    vocab: usize,
    embed: usize,
    data: Vec<f32>,
}

impl EmbeddingTable {
    fn load_aliases(gguf: &GgufFile, names: &[String], vocab: usize, embed: usize) -> Result<Self> {
        let Some(primary) = names.first() else {
            bail!("embedding alias list is empty");
        };
        for name in names {
            if gguf.tensor(name).is_some() {
                return Self::load(gguf, name, vocab, embed);
            }
        }
        Self::load(gguf, primary, vocab, embed)
    }

    fn load(gguf: &GgufFile, name: &str, vocab: usize, embed: usize) -> Result<Self> {
        let tensor = load_tensor(gguf, name)?;
        if tensor.dims.len() != 2 {
            bail!("tensor {name} must be rank 2, got {:?}", tensor.dims);
        }
        let mut data = vec![0.0; vocab * embed];
        match tensor.dims.as_slice() {
            [dim0, dim1] if *dim0 == embed && *dim1 == vocab => {
                for token in 0..vocab {
                    for hidden in 0..embed {
                        data[token * embed + hidden] = tensor.data[hidden + embed * token];
                    }
                }
            }
            [dim0, dim1] if *dim0 == vocab && *dim1 == embed => {
                for token in 0..vocab {
                    for hidden in 0..embed {
                        data[token * embed + hidden] = tensor.data[token + vocab * hidden];
                    }
                }
            }
            _ => bail!(
                "tensor {name} has shape {:?}; expected [{embed}, {vocab}] or [{vocab}, {embed}]",
                tensor.dims
            ),
        }
        Ok(Self { vocab, embed, data })
    }

    fn forward(&self, input_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        input_ids
            .iter()
            .map(|id| {
                let id = usize::try_from(*id).context("token id does not fit usize")?;
                if id >= self.vocab {
                    bail!("token id {id} is outside vocab size {}", self.vocab);
                }
                Ok(self.data[id * self.embed..(id + 1) * self.embed].to_vec())
            })
            .collect()
    }

    fn as_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.embed {
            bail!(
                "hidden size {} does not match embedding size {}",
                hidden.len(),
                self.embed
            );
        }
        let mut logits = vec![0.0; self.vocab];
        for (token, logit) in logits.iter_mut().enumerate() {
            *logit = dot(
                &self.data[token * self.embed..(token + 1) * self.embed],
                hidden,
            );
        }
        Ok(logits)
    }
}

#[derive(Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn mul_vec(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.mul_vec_with_bias(input, None)
    }

    fn mul_vec_with_bias(&self, input: &[f32], bias: Option<&[f32]>) -> Result<Vec<f32>> {
        if input.len() != self.cols {
            bail!(
                "matrix input length {} does not match matrix cols {}",
                input.len(),
                self.cols
            );
        }
        if let Some(bias) = bias
            && bias.len() != self.rows
        {
            bail!(
                "matrix bias length {} does not match matrix rows {}",
                bias.len(),
                self.rows
            );
        }
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut sum = bias.map(|bias| bias[row]).unwrap_or(0.0);
            let start = row * self.cols;
            sum += dot(&self.data[start..start + self.cols], input);
            output.push(sum);
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct TensorData {
    dims: Vec<usize>,
    data: Vec<f32>,
}

fn load_matrix(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims.len() != 2 {
        bail!("tensor {name} must be rank 2, got {:?}", tensor.dims);
    }
    let mut data = vec![0.0; rows * cols];
    match tensor.dims.as_slice() {
        [dim0, dim1] if *dim0 == cols && *dim1 == rows => {
            for row in 0..rows {
                for col in 0..cols {
                    data[row * cols + col] = tensor.data[col + cols * row];
                }
            }
        }
        [dim0, dim1] if *dim0 == rows && *dim1 == cols => {
            for row in 0..rows {
                for col in 0..cols {
                    data[row * cols + col] = tensor.data[row + rows * col];
                }
            }
        }
        _ => bail!(
            "tensor {name} has shape {:?}; expected [{cols}, {rows}] or [{rows}, {cols}]",
            tensor.dims
        ),
    }
    Ok(Matrix { rows, cols, data })
}

fn load_matrix_aliases(
    gguf: &GgufFile,
    names: &[String],
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let Some(primary) = names.first() else {
        bail!("matrix alias list is empty");
    };
    for name in names {
        if gguf.tensor(name).is_some() {
            return load_matrix(gguf, name, rows, cols);
        }
    }
    load_matrix(gguf, primary, rows, cols)
}

fn load_attention_matrix(
    gguf: &GgufFile,
    config: &QwenGgufConfig,
    prefix: &str,
    suffix: &str,
    row_offset: usize,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let names = qwen_dense_attention_weight_names(prefix, suffix);
    if names.iter().any(|name| gguf.tensor(name).is_some()) {
        return load_matrix_aliases(gguf, &names, rows, cols);
    }
    let packed_names = qwen_dense_packed_qkv_weight_names(prefix);
    if let Some(packed_name) = packed_names.iter().find(|name| gguf.tensor(name).is_some()) {
        return load_packed_matrix_rows(
            gguf,
            packed_name,
            row_offset,
            rows,
            cols,
            dense_packed_qkv_dim(config)?,
        );
    }
    load_matrix_aliases(gguf, &names, rows, cols)
}

fn load_dense_query_projection(
    gguf: &GgufFile,
    config: &QwenGgufConfig,
    prefix: &str,
    q_dim: usize,
    embed: usize,
) -> Result<QwenDenseQueryProjection> {
    let q_dim_u64 = u64::try_from(q_dim).context("qwen q dimension does not fit u64")?;
    let embed_u64 = u64::try_from(embed).context("qwen embedding dimension does not fit u64")?;
    if let Some(name) = qwen_dense_gated_attention_q_weight_name(gguf, prefix, q_dim_u64, embed_u64)
    {
        let gated_q_dim = q_dim
            .checked_mul(2)
            .context("qwen gated attention q dimension overflows usize")?;
        let q_gate_bias =
            if qwen_dense_gated_attention_q_bias_name(gguf, prefix, q_dim_u64).is_some() {
                optional_vector_aliases(
                    gguf,
                    &qwen_dense_attention_bias_names(prefix, "q"),
                    gated_q_dim,
                )?
            } else {
                None
            };
        return Ok(QwenDenseQueryProjection::Gated {
            q_gate: load_matrix(gguf, &name, gated_q_dim, embed)?,
            q_gate_bias,
        });
    }
    Ok(QwenDenseQueryProjection::Plain {
        q: load_attention_matrix(gguf, config, prefix, "q", 0, q_dim, embed)?,
        q_bias: optional_attention_bias(gguf, config, prefix, "q", 0, q_dim)?,
    })
}

fn split_gated_q_projection(
    projected: &[f32],
    heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let q_dim = heads
        .checked_mul(head_dim)
        .context("Qwen gated attention q dimension overflows usize")?;
    let expected = q_dim
        .checked_mul(2)
        .context("Qwen gated attention projected dimension overflows usize")?;
    if projected.len() != expected {
        bail!(
            "Qwen gated attention projection length {} does not match expected {expected}",
            projected.len()
        );
    }
    let mut q = vec![0.0; q_dim];
    let mut gate = vec![0.0; q_dim];
    for head in 0..heads {
        let source = head
            .checked_mul(head_dim)
            .and_then(|value| value.checked_mul(2))
            .context("Qwen gated attention source offset overflows usize")?;
        let dest = head
            .checked_mul(head_dim)
            .context("Qwen gated attention destination offset overflows usize")?;
        q[dest..dest + head_dim].copy_from_slice(&projected[source..source + head_dim]);
        gate[dest..dest + head_dim]
            .copy_from_slice(&projected[source + head_dim..source + 2 * head_dim]);
    }
    Ok((q, gate))
}

fn load_ffn_gate_matrix(
    gguf: &GgufFile,
    prefix: &str,
    gate: bool,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let names = if gate {
        qwen_dense_ffn_weight_names(prefix, "gate")
    } else {
        qwen_dense_ffn_weight_names(prefix, "up")
    };
    for name in &names {
        if gguf.tensor(name).is_some() && matrix_has_shape(gguf, name, rows, cols)? {
            return load_matrix(gguf, name, rows, cols);
        }
    }
    if let Some(source) = dense_packed_ffn_source(gguf, prefix, rows, cols)? {
        let row_offset = if gate {
            source.gate_offset
        } else {
            source.up_offset
        };
        return load_packed_matrix_rows(
            gguf,
            &source.name,
            row_offset,
            rows,
            cols,
            rows.checked_mul(2)
                .context("dense packed ffn row count overflows usize")?,
        );
    }
    load_matrix_aliases(gguf, &names, rows, cols)
}

fn optional_ffn_gate_bias(
    gguf: &GgufFile,
    prefix: &str,
    gate: bool,
    ff: usize,
) -> Result<Option<Vec<f32>>> {
    let kind = if gate { "gate" } else { "up" };
    let names = qwen_dense_ffn_bias_names(prefix, kind);
    if names.iter().any(|name| gguf.tensor(name).is_some()) {
        return optional_vector_aliases(gguf, &names, ff);
    }
    let packed_len = ff
        .checked_mul(2)
        .context("dense packed ffn bias length overflows usize")?;
    for name in qwen_dense_packed_ffn_gate_up_bias_names(prefix) {
        if gguf.tensor(&name).is_some() {
            let offset = if gate { 0 } else { ff };
            return Ok(Some(load_packed_vector_range(
                gguf, &name, offset, ff, packed_len,
            )?));
        }
    }
    for name in qwen_dense_packed_ffn_up_gate_bias_names(prefix) {
        if gguf.tensor(&name).is_some() {
            let offset = if gate { ff } else { 0 };
            return Ok(Some(load_packed_vector_range(
                gguf, &name, offset, ff, packed_len,
            )?));
        }
    }
    Ok(None)
}

struct PackedFfnSource {
    name: String,
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
        // gate is the first half. Only matches when the shape is [2*ff, embed], so a
        // plain `ffn_up` (ff rows) is skipped and handled by the separate-tensor path.
        .chain(std::iter::once((format!("{prefix}.ffn_up.weight"), true)));
    for (name, gate_first) in aliases {
        let Some(_) = gguf.tensor(&name) else {
            continue;
        };
        if !matrix_has_shape(
            gguf,
            &name,
            ff.checked_mul(2)
                .context("dense packed ffn row count overflows usize")?,
            embed,
        )? {
            continue;
        }
        let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
        return Ok(Some(PackedFfnSource {
            name,
            gate_offset,
            up_offset,
        }));
    }
    Ok(None)
}

fn matrix_has_shape(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<bool> {
    let Some(view) = gguf.tensor(name) else {
        return Ok(false);
    };
    let dims = view
        .info
        .dimensions
        .iter()
        .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
        .collect::<Result<Vec<_>>>()?;
    Ok(matches!(
        dims.as_slice(),
        [dim0, dim1] if (*dim0 == cols && *dim1 == rows) || (*dim0 == rows && *dim1 == cols)
    ))
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

fn load_packed_matrix_rows(
    gguf: &GgufFile,
    name: &str,
    row_offset: usize,
    rows: usize,
    cols: usize,
    source_rows: usize,
) -> Result<Matrix> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims.len() != 2 {
        bail!("packed tensor {name} must be rank 2, got {:?}", tensor.dims);
    }
    if row_offset
        .checked_add(rows)
        .is_none_or(|end| end > source_rows)
    {
        bail!(
            "packed tensor {name} row slice {row_offset}..{} exceeds source rows {source_rows}",
            row_offset + rows
        );
    }
    let mut data = vec![0.0; rows * cols];
    match tensor.dims.as_slice() {
        [dim0, dim1] if *dim0 == cols && *dim1 == source_rows => {
            for row in 0..rows {
                let source_row = row_offset + row;
                for col in 0..cols {
                    data[row * cols + col] = tensor.data[col + cols * source_row];
                }
            }
        }
        [dim0, dim1] if *dim0 == source_rows && *dim1 == cols => {
            for row in 0..rows {
                let source_row = row_offset + row;
                for col in 0..cols {
                    data[row * cols + col] = tensor.data[source_row + source_rows * col];
                }
            }
        }
        _ => bail!(
            "packed tensor {name} has shape {:?}; expected [{cols}, {source_rows}] or [{source_rows}, {cols}]",
            tensor.dims
        ),
    }
    Ok(Matrix { rows, cols, data })
}

fn optional_attention_bias(
    gguf: &GgufFile,
    config: &QwenGgufConfig,
    prefix: &str,
    suffix: &str,
    row_offset: usize,
    len: usize,
) -> Result<Option<Vec<f32>>> {
    let names = qwen_dense_attention_bias_names(prefix, suffix);
    if names.iter().any(|name| gguf.tensor(name).is_some()) {
        return optional_vector_aliases(gguf, &names, len);
    }
    let packed_names = qwen_dense_packed_qkv_bias_names(prefix);
    if let Some(packed_name) = packed_names.iter().find(|name| gguf.tensor(name).is_some()) {
        return Ok(Some(load_packed_vector_range(
            gguf,
            packed_name,
            row_offset,
            len,
            dense_packed_qkv_dim(config)?,
        )?));
    }
    Ok(None)
}

fn load_vector_aliases(gguf: &GgufFile, names: &[String], len: usize) -> Result<Vec<f32>> {
    let Some(primary) = names.first() else {
        bail!("vector alias list is empty");
    };
    for name in names {
        if gguf.tensor(name).is_some() {
            return load_vector(gguf, name, len);
        }
    }
    load_vector(gguf, primary, len)
}

fn optional_vector_aliases(
    gguf: &GgufFile,
    names: &[String],
    len: usize,
) -> Result<Option<Vec<f32>>> {
    for name in names {
        if gguf.tensor(name).is_some() {
            return optional_vector(gguf, name, len);
        }
    }
    Ok(None)
}

fn dense_packed_qkv_dim(config: &QwenGgufConfig) -> Result<usize> {
    let heads = usize::try_from(config.attention_head_count)
        .context("attention.head_count does not fit usize")?;
    let kv_heads = usize::try_from(config.attention_head_count_kv)
        .context("attention.head_count_kv does not fit usize")?;
    let qk_head_dim = config
        .attention_key_head_dim()
        .map(usize::try_from)
        .transpose()
        .context("attention key head dimension does not fit usize")?
        .ok_or_else(|| anyhow!("invalid attention metadata for packed qkv"))?;
    let v_head_dim = config
        .attention_value_head_dim()
        .map(usize::try_from)
        .transpose()
        .context("attention value head dimension does not fit usize")?
        .ok_or_else(|| anyhow!("invalid attention metadata for packed qkv"))?;
    if heads == 0 {
        bail!("invalid attention metadata for packed qkv");
    }
    let q_dim = qk_head_dim
        .checked_mul(heads)
        .context("packed qkv q dimension overflows usize")?;
    let k_dim = qk_head_dim
        .checked_mul(kv_heads)
        .context("packed qkv k dimension overflows usize")?;
    let v_dim = v_head_dim
        .checked_mul(kv_heads)
        .context("packed qkv v dimension overflows usize")?;
    q_dim
        .checked_add(k_dim)
        .and_then(|value| value.checked_add(v_dim))
        .context("packed qkv dimension overflows usize")
}

fn load_packed_vector_range(
    gguf: &GgufFile,
    name: &str,
    offset: usize,
    len: usize,
    source_len: usize,
) -> Result<Vec<f32>> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims != [source_len] {
        bail!(
            "packed vector {name} has shape {:?}; expected [{source_len}]",
            tensor.dims
        );
    }
    let end = offset
        .checked_add(len)
        .context("packed vector slice end overflows usize")?;
    tensor
        .data
        .get(offset..end)
        .map(|slice| slice.to_vec())
        .ok_or_else(|| anyhow!("packed vector {name} slice {offset}..{end} is out of range"))
}

fn optional_moe_expert_bias(
    gguf: &GgufFile,
    prefix: &str,
    use_per_expert_tensor: bool,
    expert: usize,
    experts: usize,
    kind: &str,
    len: usize,
) -> Result<Option<Vec<f32>>> {
    if use_per_expert_tensor {
        return optional_vector_aliases(
            gguf,
            &qwen_moe_per_expert_bias_names(prefix, kind, expert as u64),
            len,
        );
    }
    optional_expert_vector_aliases(
        gguf,
        &qwen_moe_packed_expert_bias_names(prefix, kind),
        expert,
        experts,
        len,
    )
}

fn optional_moe_expert_gate_up_bias(
    gguf: &GgufFile,
    prefix: &str,
    use_per_expert_tensor: bool,
    expert: usize,
    experts: usize,
    ff: usize,
    gate: bool,
) -> Result<Option<Vec<f32>>> {
    if use_per_expert_tensor {
        if let Some(source) = moe_per_expert_packed_gate_up_bias_source(gguf, prefix, ff, expert)? {
            let offset = if gate {
                source.gate_offset
            } else {
                source.up_offset
            };
            return load_packed_vector_range(gguf, &source.name, offset, ff, source.source_rows)
                .map(Some);
        }
        let kind = if gate { "gate" } else { "up" };
        return optional_vector_aliases(
            gguf,
            &qwen_moe_per_expert_bias_names(prefix, kind, expert as u64),
            ff,
        );
    }
    if let Some(source) = moe_packed_expert_gate_up_bias_source(gguf, prefix, ff, experts)? {
        let offset = if gate {
            source.gate_offset
        } else {
            source.up_offset
        };
        return load_expert_vector_range(
            gguf,
            &source.name,
            expert,
            experts,
            offset,
            ff,
            source.source_rows,
        )
        .map(Some);
    }
    let kind = if gate { "gate" } else { "up" };
    optional_expert_vector_aliases(
        gguf,
        &qwen_moe_packed_expert_bias_names(prefix, kind),
        expert,
        experts,
        ff,
    )
}

fn optional_moe_shared_expert_gate_up_bias(
    gguf: &GgufFile,
    prefix: &str,
    ff: usize,
    gate: bool,
) -> Result<Option<Vec<f32>>> {
    if let Some(source) = moe_shared_expert_packed_gate_up_bias_source(gguf, prefix, ff)? {
        let offset = if gate {
            source.gate_offset
        } else {
            source.up_offset
        };
        return load_packed_vector_range(gguf, &source.name, offset, ff, source.source_rows)
            .map(Some);
    }
    let kind = if gate { "gate" } else { "up" };
    optional_vector_aliases(gguf, &qwen_moe_shared_expert_bias_names(prefix, kind), ff)
}

fn optional_expert_vector_aliases(
    gguf: &GgufFile,
    names: &[String],
    expert: usize,
    experts: usize,
    len: usize,
) -> Result<Option<Vec<f32>>> {
    for name in names {
        if gguf.tensor(name).is_some() {
            return load_expert_vector(gguf, name, expert, experts, len).map(Some);
        }
    }
    Ok(None)
}

fn load_expert_vector(
    gguf: &GgufFile,
    name: &str,
    expert: usize,
    experts: usize,
    len: usize,
) -> Result<Vec<f32>> {
    load_expert_vector_range(gguf, name, expert, experts, 0, len, len)
}

fn load_expert_vector_range(
    gguf: &GgufFile,
    name: &str,
    expert: usize,
    experts: usize,
    offset: usize,
    len: usize,
    source_len: usize,
) -> Result<Vec<f32>> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims.len() != 2 {
        bail!(
            "expert bias tensor {name} must be rank 2, got {:?}",
            tensor.dims
        );
    }
    if expert >= experts {
        bail!("expert index {expert} is outside expert count {experts}");
    }
    if offset.checked_add(len).is_none_or(|end| end > source_len) {
        bail!(
            "expert bias tensor {name} slice {offset}..{} exceeds source length {source_len}",
            offset + len
        );
    }
    let mut values = vec![0.0; len];
    match tensor.dims.as_slice() {
        [dim0, dim1] if *dim0 == source_len && *dim1 == experts => {
            let start = expert
                .checked_mul(source_len)
                .context("expert bias offset overflows usize")?;
            values.copy_from_slice(&tensor.data[start + offset..start + offset + len]);
        }
        [dim0, dim1] if *dim0 == experts && *dim1 == source_len => {
            for idx in 0..len {
                values[idx] = tensor.data[expert + experts * (offset + idx)];
            }
        }
        _ => bail!(
            "expert bias tensor {name} has shape {:?}; expected [{source_len}, {experts}] or [{experts}, {source_len}]",
            tensor.dims
        ),
    }
    Ok(values)
}

fn load_expert_matrix(
    gguf: &GgufFile,
    name: &str,
    expert: usize,
    experts: usize,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims.len() != 3 {
        bail!("expert tensor {name} must be rank 3, got {:?}", tensor.dims);
    }
    if tensor.dims[2] != experts {
        bail!(
            "expert tensor {name} has expert dimension {}; expected {experts}",
            tensor.dims[2]
        );
    }
    if expert >= experts {
        bail!("expert index {expert} is outside expert count {experts}");
    }
    let stride = tensor.dims[0]
        .checked_mul(tensor.dims[1])
        .context("expert tensor stride overflows usize")?;
    let start = expert
        .checked_mul(stride)
        .context("expert tensor offset overflows usize")?;
    let source = &tensor.data[start..start + stride];
    let mut data = vec![0.0; rows * cols];
    match &tensor.dims[..2] {
        [dim0, dim1] if *dim0 == cols && *dim1 == rows => {
            for row in 0..rows {
                for col in 0..cols {
                    data[row * cols + col] = source[col + cols * row];
                }
            }
        }
        [dim0, dim1] if *dim0 == rows && *dim1 == cols => {
            for row in 0..rows {
                for col in 0..cols {
                    data[row * cols + col] = source[row + rows * col];
                }
            }
        }
        _ => bail!(
            "expert tensor {name} has shape {:?}; expected [{cols}, {rows}, {experts}] or [{rows}, {cols}, {experts}]",
            tensor.dims
        ),
    }
    Ok(Matrix { rows, cols, data })
}

fn load_moe_expert_matrix(
    gguf: &GgufFile,
    packed_names: &[String],
    per_expert_names: &[String],
    use_per_expert_tensor: bool,
    expert: usize,
    experts: usize,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    if use_per_expert_tensor {
        return load_matrix_aliases(gguf, per_expert_names, rows, cols);
    }
    load_expert_matrix_aliases(gguf, packed_names, expert, experts, rows, cols)
}

fn load_moe_expert_gate_up_matrix(
    gguf: &GgufFile,
    prefix: &str,
    use_per_expert_tensor: bool,
    expert: usize,
    experts: usize,
    ff: usize,
    embed: usize,
    gate: bool,
) -> Result<Matrix> {
    if use_per_expert_tensor {
        if let Some(source) = moe_per_expert_packed_gate_up_source(gguf, prefix, ff, embed, expert)?
        {
            let offset = if gate {
                source.gate_offset
            } else {
                source.up_offset
            };
            return load_packed_matrix_rows(
                gguf,
                &source.name,
                offset,
                ff,
                embed,
                source.source_rows,
            );
        }
        let kind = if gate { "gate" } else { "up" };
        return load_matrix_aliases(
            gguf,
            &qwen_moe_per_expert_weight_names(prefix, kind, expert as u64),
            ff,
            embed,
        );
    }
    if let Some(source) = moe_packed_expert_gate_up_source(gguf, prefix, ff, embed, experts)? {
        let offset = if gate {
            source.gate_offset
        } else {
            source.up_offset
        };
        return load_expert_matrix_rows(
            gguf,
            &source.name,
            expert,
            experts,
            offset,
            ff,
            embed,
            source.source_rows,
        );
    }
    let kind = if gate { "gate" } else { "up" };
    load_expert_matrix_aliases(
        gguf,
        &qwen_moe_packed_expert_weight_names(prefix, kind),
        expert,
        experts,
        ff,
        embed,
    )
}

struct MoePackedGateUpSource {
    name: String,
    source_rows: usize,
    gate_offset: usize,
    up_offset: usize,
}

fn moe_packed_expert_gate_up_source(
    gguf: &GgufFile,
    prefix: &str,
    ff: usize,
    embed: usize,
    experts: usize,
) -> Result<Option<MoePackedGateUpSource>> {
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
        let Some(view) = gguf.tensor(&name) else {
            continue;
        };
        let dims = view
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
            return Ok(Some(MoePackedGateUpSource {
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
) -> Result<Option<MoePackedGateUpSource>> {
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
        let Some(view) = gguf.tensor(&name) else {
            continue;
        };
        let dims = view
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
            .collect::<Result<Vec<_>>>()?;
        if expert_vector_dims_match(&dims, source_rows, experts) {
            let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
            return Ok(Some(MoePackedGateUpSource {
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
) -> Result<Option<MoePackedGateUpSource>> {
    let source_rows = ff
        .checked_mul(2)
        .context("qwen packed per-expert MoE gate/up rows overflow usize")?;
    let expert = u64::try_from(expert).context("expert index does not fit u64")?;
    let aliases = qwen_moe_per_expert_gate_up_weight_names(prefix, expert)
        .into_iter()
        .map(|name| (name, true))
        .chain(
            qwen_moe_per_expert_up_gate_weight_names(prefix, expert)
                .into_iter()
                .map(|name| (name, false)),
        );
    for (name, gate_first) in aliases {
        let Some(view) = gguf.tensor(&name) else {
            continue;
        };
        let dims = view
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
            .collect::<Result<Vec<_>>>()?;
        if dims.len() == 2 && matrix_dims_match(&dims, source_rows, embed) {
            let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
            return Ok(Some(MoePackedGateUpSource {
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
) -> Result<Option<MoePackedGateUpSource>> {
    let source_rows = ff
        .checked_mul(2)
        .context("qwen packed per-expert MoE gate/up bias length overflows usize")?;
    let expert = u64::try_from(expert).context("expert index does not fit u64")?;
    let aliases = qwen_moe_per_expert_gate_up_bias_names(prefix, expert)
        .into_iter()
        .map(|name| (name, true))
        .chain(
            qwen_moe_per_expert_up_gate_bias_names(prefix, expert)
                .into_iter()
                .map(|name| (name, false)),
        );
    for (name, gate_first) in aliases {
        let Some(view) = gguf.tensor(&name) else {
            continue;
        };
        let dims = view
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
            .collect::<Result<Vec<_>>>()?;
        if dims == [source_rows] {
            let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
            return Ok(Some(MoePackedGateUpSource {
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
) -> Result<Option<MoePackedGateUpSource>> {
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
        let Some(view) = gguf.tensor(&name) else {
            continue;
        };
        let dims = view
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
            .collect::<Result<Vec<_>>>()?;
        if dims.len() == 2 && matrix_dims_match(&dims, source_rows, embed) {
            let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
            return Ok(Some(MoePackedGateUpSource {
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
) -> Result<Option<MoePackedGateUpSource>> {
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
        let Some(view) = gguf.tensor(&name) else {
            continue;
        };
        let dims = view
            .info
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
            .collect::<Result<Vec<_>>>()?;
        if dims == [source_rows] {
            let (gate_offset, up_offset) = if gate_first { (0, ff) } else { (ff, 0) };
            return Ok(Some(MoePackedGateUpSource {
                name,
                source_rows,
                gate_offset,
                up_offset,
            }));
        }
    }
    Ok(None)
}

fn load_expert_matrix_rows(
    gguf: &GgufFile,
    name: &str,
    expert: usize,
    experts: usize,
    row_offset: usize,
    rows: usize,
    cols: usize,
    source_rows: usize,
) -> Result<Matrix> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims.len() != 3 {
        bail!("expert tensor {name} must be rank 3, got {:?}", tensor.dims);
    }
    if tensor.dims[2] != experts {
        bail!(
            "expert tensor {name} has expert dimension {}; expected {experts}",
            tensor.dims[2]
        );
    }
    if expert >= experts {
        bail!("expert index {expert} is outside expert count {experts}");
    }
    if row_offset
        .checked_add(rows)
        .is_none_or(|end| end > source_rows)
    {
        bail!(
            "expert tensor {name} row slice {row_offset}..{} exceeds source rows {source_rows}",
            row_offset + rows
        );
    }
    let stride = tensor.dims[0]
        .checked_mul(tensor.dims[1])
        .context("expert tensor stride overflows usize")?;
    let start = expert
        .checked_mul(stride)
        .context("expert tensor offset overflows usize")?;
    let source = &tensor.data[start..start + stride];
    let mut data = vec![0.0; rows * cols];
    match &tensor.dims[..2] {
        [dim0, dim1] if *dim0 == cols && *dim1 == source_rows => {
            for row in 0..rows {
                let source_row = row_offset + row;
                for col in 0..cols {
                    data[row * cols + col] = source[col + cols * source_row];
                }
            }
        }
        [dim0, dim1] if *dim0 == source_rows && *dim1 == cols => {
            for row in 0..rows {
                let source_row = row_offset + row;
                for col in 0..cols {
                    data[row * cols + col] = source[source_row + source_rows * col];
                }
            }
        }
        _ => bail!(
            "expert tensor {name} has shape {:?}; expected [{cols}, {source_rows}, {experts}] or [{source_rows}, {cols}, {experts}]",
            tensor.dims
        ),
    }
    Ok(Matrix { rows, cols, data })
}

fn load_expert_matrix_aliases(
    gguf: &GgufFile,
    names: &[String],
    expert: usize,
    experts: usize,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let Some(primary) = names.first() else {
        bail!("expert matrix alias list is empty");
    };
    for name in names {
        if gguf.tensor(name).is_some() {
            return load_expert_matrix(gguf, name, expert, experts, rows, cols);
        }
    }
    load_expert_matrix(gguf, primary, expert, experts, rows, cols)
}

fn expert_ff_dim(
    gguf: &GgufFile,
    config: &QwenGgufConfig,
    prefix: &str,
    experts: usize,
    embed: usize,
) -> Result<usize> {
    if let Some(ff) = config.expert_feed_forward_length {
        return usize::try_from(ff).context("qwen expert_feed_forward_length does not fit usize");
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
            .map(|dim| usize::try_from(*dim).context("expert tensor dimension does not fit usize"))
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
            .map(|dim| usize::try_from(*dim).context("expert tensor dimension does not fit usize"))
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

fn load_vector(gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let tensor = load_tensor(gguf, name)?;
    if tensor.dims != [len] {
        bail!(
            "tensor {name} has shape {:?}; expected [{len}]",
            tensor.dims
        );
    }
    Ok(tensor.data)
}

fn optional_vector(gguf: &GgufFile, name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    if gguf.tensor(name).is_some() {
        Ok(Some(load_vector(gguf, name, len)?))
    } else {
        Ok(None)
    }
}

fn load_tensor(gguf: &GgufFile, name: &str) -> Result<TensorData> {
    let view = gguf
        .tensor(name)
        .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
    let dims = view
        .info
        .dimensions
        .iter()
        .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
        .collect::<Result<Vec<_>>>()?;
    let element_count = view.info.element_count().and_then(|count| {
        usize::try_from(count).context("tensor element count does not fit usize")
    })?;
    let data = dequantize_tensor_as_f32(view.bytes, view.info.dtype, element_count)?;
    Ok(TensorData { dims, data })
}

#[cfg(test)]
fn f16_to_f32(raw: u16) -> f32 {
    let sign = (u32::from(raw & 0x8000)) << 16;
    let exp = (raw >> 10) & 0x1f;
    let frac = u32::from(raw & 0x03ff);
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14i32;
                while frac & 0x0400 == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                let exp_bits = u32::try_from(exp + 127).expect("subnormal exponent") << 23;
                sign | exp_bits | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp_bits = (u32::from(exp) + 112) << 23;
            sign | exp_bits | (frac << 13)
        }
    };
    f32::from_bits(bits)
}

fn rms_norm_in_place(values: &mut [f32], weight: &[f32], eps: f32) -> Result<()> {
    if values.len() != weight.len() {
        bail!(
            "RMSNorm input length {} does not match weight length {}",
            values.len(),
            weight.len()
        );
    }
    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let inv = (mean_square + eps).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value *= inv * weight;
    }
    Ok(())
}

fn apply_rope(
    values: &mut [f32],
    position: usize,
    base: f32,
    scale: f32,
    split_half: bool,
) -> Result<()> {
    if !values.len().is_multiple_of(2) {
        bail!("RoPE head dimension {} must be even", values.len());
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

fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum != 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

fn add_residual(hidden: &mut [Vec<f32>], residual: &[Vec<f32>]) -> Result<()> {
    if hidden.len() != residual.len() {
        bail!(
            "residual sequence length {} does not match hidden length {}",
            residual.len(),
            hidden.len()
        );
    }
    for (hidden_token, residual_token) in hidden.iter_mut().zip(residual) {
        if hidden_token.len() != residual_token.len() {
            bail!(
                "residual hidden length {} does not match hidden length {}",
                residual_token.len(),
                hidden_token.len()
            );
        }
        for (value, residual) in hidden_token.iter_mut().zip(residual_token) {
            *value += residual;
        }
    }
    Ok(())
}

fn add_vector_bias(values: &mut [f32], bias: &[f32]) -> Result<()> {
    if values.len() != bias.len() {
        bail!(
            "bias length {} does not match value length {}",
            bias.len(),
            values.len()
        );
    }
    for (value, bias) in values.iter_mut().zip(bias) {
        *value += bias;
    }
    Ok(())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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

fn l2_normalize(values: &mut [f32]) {
    let norm_sq = values.iter().map(|value| value * value).sum::<f32>();
    let inv_norm = (norm_sq + 1.0e-6).sqrt().recip();
    for value in values {
        *value *= inv_norm;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn argmax(values: &[f32]) -> Result<u32> {
    if values.is_empty() {
        bail!("cannot sample from empty logits");
    }
    let mut best_idx = 0usize;
    let mut best_value = values[0];
    for (idx, value) in values.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }
    u32::try_from(best_idx).context("argmax token index does not fit u32")
}

pub fn sample_from_logits_with_rng<R: Rng + ?Sized>(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: Option<u32>,
    rng: &mut R,
) -> Result<u32> {
    if logits.is_empty() {
        bail!("cannot sample from empty logits");
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return argmax(logits);
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
        return argmax(logits);
    }
    for value in &mut scaled {
        *value = (*value - max).exp();
    }

    let mut ranked = scaled.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    if let Some(top_k) = top_k.and_then(|value| usize::try_from(value).ok())
        && top_k > 0
    {
        ranked.truncate(top_k.min(ranked.len()));
    }
    let cutoff = if top_p.is_finite() {
        top_p.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let total = ranked.iter().map(|(_, weight)| *weight).sum::<f32>();
    if total <= 0.0 {
        return argmax(logits);
    }

    let mut cumulative = 0.0;
    let mut candidates = Vec::new();
    for (idx, weight) in ranked {
        if weight <= 0.0 {
            continue;
        }
        candidates.push((idx, weight));
        cumulative += weight / total;
        if cutoff < 1.0 && cumulative >= cutoff {
            break;
        }
    }
    if candidates.is_empty() {
        return argmax(logits);
    }

    let dist = WeightedIndex::new(candidates.iter().map(|(_, weight)| *weight))
        .context("building top-p sampling distribution")?;
    let selected = dist.sample(rng);
    u32::try_from(candidates[selected].0).context("sampled token index does not fit u32")
}

pub fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: Option<u32>,
    seed: Option<u64>,
) -> Result<u32> {
    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        sample_from_logits_with_rng(logits, temperature, top_p, top_k, &mut rng)
    } else {
        let mut rng = rand::thread_rng();
        sample_from_logits_with_rng(logits, temperature, top_p, top_k, &mut rng)
    }
}

fn top_logits(logits: &[f32], tokenizer: &GgufTokenizer, top_k: usize) -> Result<Vec<TopLogit>> {
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    ranked
        .into_iter()
        .take(top_k.min(logits.len()))
        .map(|(idx, logit)| {
            let token_id = u32::try_from(idx).context("top logit token index does not fit u32")?;
            Ok(TopLogit {
                token_id,
                token: tokenizer.token(token_id).map(ToString::to_string),
                logit,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use hi_local_core::model::ModelFamily;

    #[test]
    fn decodes_fp16_and_bf16_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        let bf16_one = f32::from_bits(u32::from(0x3f80u16) << 16);
        assert_eq!(bf16_one, 1.0);
    }

    #[test]
    fn cpu_reference_returns_known_logits_for_tied_head() {
        let path = tempfile_path("cpu-known-logits");
        write_reference_qwen(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
        assert_eq!(model.greedy_next_token(&[0]).unwrap(), 0);
    }

    #[test]
    fn cpu_reference_generates_greedy_tokens() {
        let path = tempfile_path("cpu-generate");
        write_reference_qwen(&path);

        let model = QwenCpuReference::load(&path).unwrap();

        assert_eq!(model.generate_greedy(&[1], 2).unwrap(), vec![1, 1]);
    }

    #[test]
    fn cpu_reference_loads_qwen_moe_fixture() {
        let path = tempfile_path("cpu-moe");
        write_reference_qwen_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().expert_count, Some(2));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_qwen_moe_packed_gate_up_fixture() {
        let path = tempfile_path("cpu-moe-packed-gate-up");
        write_reference_qwen_moe_packed_gate_up(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "qwen3moe");
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_qwen_equal_custom_kv_fixture() {
        let path = tempfile_path("cpu-qwen-equal-custom-kv");
        write_reference_qwen_equal_custom_kv(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.25 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().attention_key_length, Some(2));
        assert_eq!(model.config().attention_value_length, Some(2));
        assert_eq!(model.config().attention_head_dim(), Some(2));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_qwen_unequal_custom_kv_fixture() {
        let path = tempfile_path("cpu-qwen-unequal-custom-kv");
        write_reference_qwen_unequal_custom_kv(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.25 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().attention_key_length, Some(2));
        assert_eq!(model.config().attention_value_length, Some(3));
        assert_eq!(model.config().attention_key_head_dim(), Some(2));
        assert_eq!(model.config().attention_value_head_dim(), Some(3));
        assert_eq!(model.config().attention_head_dim(), None);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_qwen_next_dense_fixture() {
        let path = tempfile_path("cpu-qwen-next-dense");
        write_reference_qwen_next_dense(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "qwen3next");
        assert_eq!(model.config().family, ModelFamily::Qwen3);
        assert_eq!(model.config().default_rope_freq_base(), 1_000_000.0);
        assert_eq!(model.config().expert_count, None);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_qwen_next_gated_attention_fixture() {
        let path = tempfile_path("cpu-qwen-next-gated-attention");
        write_reference_qwen_next_gated_attention(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "qwen3next");
        assert_eq!(model.config().family, ModelFamily::Qwen3);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_generates_with_qwen_next_recurrent_ssm_fixture() {
        let path = tempfile_path("cpu-qwen-next-recurrent-ssm");
        write_reference_qwen_next_recurrent_ssm(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();
        let generated = model.generate_greedy(&[0], 2).unwrap();

        assert_eq!(model.config().architecture, "qwen3next");
        assert_eq!(model.config().family, ModelFamily::Qwen3);
        assert!(model.config().recurrent_ssm_tensor_layout);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
        assert_eq!(generated, vec![0, 0]);
    }

    #[test]
    fn cpu_reference_generates_with_qwen_next_recurrent_ssm_optimized_fixture() {
        let path = tempfile_path("cpu-qwen-next-recurrent-ssm-optimized");
        write_reference_qwen_next_recurrent_ssm_optimized(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();
        let generated = model.generate_greedy(&[0], 2).unwrap();

        assert_eq!(model.config().architecture, "qwen3next");
        assert_eq!(model.config().family, ModelFamily::Qwen3);
        assert!(model.config().recurrent_ssm_tensor_layout);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
        assert_eq!(generated, vec![0, 0]);
    }

    #[test]
    fn cpu_reference_loads_mistral_dense_alias_fixture() {
        let path = tempfile_path("cpu-mistral-dense-aliases");
        write_reference_mistral_dense_aliases(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "mistral");
        assert_eq!(model.config().family, ModelFamily::Mistral);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_mistral_packed_alias_fixture() {
        let path = tempfile_path("cpu-mistral-packed-aliases");
        write_reference_mistral_packed_aliases(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "mistral");
        assert_eq!(model.config().family, ModelFamily::Mistral);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_mistral_alternate_packed_alias_fixture() {
        let path = tempfile_path("cpu-mistral-alternate-packed-aliases");
        write_reference_mistral_alternate_packed_aliases(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "mistral");
        assert_eq!(model.config().family, ModelFamily::Mistral);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_deepseek_dense_fixture() {
        let path = tempfile_path("cpu-deepseek-dense");
        write_reference_deepseek_dense(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "deepseek");
        assert_eq!(model.config().family, ModelFamily::DeepSeek);
        assert_eq!(model.config().default_rope_freq_base(), 10_000.0);
        assert_eq!(model.config().expert_count, None);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_glm_dense_fixture() {
        let path = tempfile_path("cpu-glm-dense");
        write_reference_glm_dense(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "glm4");
        assert_eq!(model.config().family, ModelFamily::GlmFlash);
        assert_eq!(model.config().default_rope_freq_base(), 1_000_000.0);
        assert_eq!(model.config().expert_count, None);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_glm_flash_dense_fixture() {
        let path = tempfile_path("cpu-glm-flash-dense");
        write_reference_glm_flash_dense(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "glm4flash");
        assert_eq!(model.config().family, ModelFamily::GlmFlash);
        assert_eq!(model.config().default_rope_freq_base(), 1_000_000.0);
        assert_eq!(model.config().expert_count, None);
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_glm_flash_moe_fixture() {
        let path = tempfile_path("cpu-glm-flash-moe");
        write_reference_glm_flash_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "glm4flash");
        assert_eq!(model.config().family, ModelFamily::GlmFlash);
        assert_eq!(model.config().expert_feed_forward_length, Some(2));
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_glm_moe_fixture() {
        let path = tempfile_path("cpu-glm-moe");
        write_reference_glm_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "glm4moe");
        assert_eq!(model.config().family, ModelFamily::GlmFlash);
        assert_eq!(model.config().expert_feed_forward_length, Some(2));
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_mixtral_moe_fixture() {
        let path = tempfile_path("cpu-mixtral-moe");
        write_reference_mixtral_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().family, ModelFamily::Mixtral);
        assert_eq!(model.config().expert_feed_forward_length, Some(2));
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_mixtral_per_expert_moe_fixture() {
        let path = tempfile_path("cpu-mixtral-per-expert-moe");
        write_reference_mixtral_per_expert_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().family, ModelFamily::Mixtral);
        assert_eq!(model.config().expert_feed_forward_length, Some(2));
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_mixtral_alias_per_expert_moe_fixture() {
        let path = tempfile_path("cpu-mixtral-alias-per-expert-moe");
        write_reference_mixtral_alias_per_expert_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().family, ModelFamily::Mixtral);
        assert_eq!(model.config().expert_feed_forward_length, Some(2));
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn cpu_reference_loads_deepseek_moe_fixture() {
        let path = tempfile_path("cpu-deepseek-moe");
        write_reference_deepseek_moe(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let logits = model.last_logits(&[0]).unwrap();
        let expected = (0.5 + 1.0e-6f32).sqrt().recip();

        assert_eq!(model.config().architecture, "deepseek");
        assert_eq!(model.config().family, ModelFamily::DeepSeek);
        assert_eq!(model.config().expert_feed_forward_length, Some(2));
        assert_eq!(model.config().expert_count, Some(2));
        assert_eq!(model.config().expert_used_count, Some(1));
        assert_close(logits[0], expected);
        assert_close(logits[1], 0.0);
        assert_close(logits[2], expected);
    }

    #[test]
    fn run_tokens_reports_top_logits_and_generation_text() {
        let path = tempfile_path("cpu-run");
        write_reference_qwen(&path);

        let model = QwenCpuReference::load(&path).unwrap();
        let output = model
            .run_tokens(
                &[0],
                QwenCpuRunOptions {
                    max_tokens: 1,
                    top_k: 2,
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: None,
                    include_logits: true,
                },
            )
            .unwrap();

        assert_eq!(output.backend, "cpu-reference");
        assert_eq!(output.input_tokens, vec![0]);
        assert_eq!(output.next_token, 0);
        assert_eq!(output.next_text, "a");
        assert_eq!(output.generated_tokens, vec![0]);
        assert_eq!(output.generated_text, "a");
        assert_eq!(output.top_logits.len(), 2);
        assert_eq!(output.top_logits[0].token_id, 0);
        assert_eq!(output.top_logits[0].token.as_deref(), Some("a"));
        assert_eq!(output.top_logits[1].token_id, 2);
        assert_eq!(output.logits.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn attention_uses_causal_prefix_only() {
        let attention = QwenAttention {
            projection: QwenAttentionProjection::Dense {
                q: QwenDenseQueryProjection::Plain {
                    q: test_matrix(2, 2, &[0.0; 4]),
                    q_bias: None,
                },
                k: test_matrix(2, 2, &[0.0; 4]),
                v: test_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]),
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
            },
            o: test_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            o_bias: None,
            heads: 1,
            kv_heads: 1,
            qk_head_dim: 2,
            v_head_dim: 2,
            rope_base: 1_000_000.0,
            rope_scale: 1.0,
            split_half_rope: true,
            rope_rot_dim: 2,
        };

        let output = attention
            .forward(&[vec![2.0, 0.0], vec![0.0, 2.0]], 1.0e-6)
            .unwrap();

        assert_close(output[0][0], 2.0);
        assert_close(output[0][1], 0.0);
        assert_close(output[1][0], 1.0);
        assert_close(output[1][1], 1.0);
    }

    #[test]
    fn qwen_moe_routes_to_top_expert() {
        let expert0 = QwenMlp {
            gate: test_matrix(1, 2, &[2.0, 0.0]),
            gate_bias: None,
            up: test_matrix(1, 2, &[3.0, 0.0]),
            up_bias: None,
            down: test_matrix(2, 1, &[1.0, 2.0]),
            down_bias: None,
        };
        let expert1 = QwenMlp {
            gate: test_matrix(1, 2, &[0.0, 2.0]),
            gate_bias: None,
            up: test_matrix(1, 2, &[0.0, 3.0]),
            up_bias: None,
            down: test_matrix(2, 1, &[10.0, 20.0]),
            down_bias: None,
        };
        let moe = QwenMoe {
            router: test_matrix(2, 2, &[4.0, 0.0, 0.0, 1.0]),
            router_bias: None,
            experts: vec![expert0, expert1],
            shared: None,
            shared_gate: None,
            shared_gate_bias: None,
            top_k: 1,
            norm_topk_prob: true,
        };

        let output = moe.forward(&[vec![1.0, 0.0]]).unwrap();
        let route_weight = 4.0f32.exp() / (4.0f32.exp() + 0.0f32.exp());
        let expected_hidden = route_weight * silu(2.0) * 3.0;

        assert_close(output[0][0], expected_hidden);
        assert_close(output[0][1], expected_hidden * 2.0);
    }

    #[test]
    fn qwen_moe_applies_expert_biases() {
        let expert0 = QwenMlp {
            gate: test_matrix(1, 2, &[0.0; 2]),
            gate_bias: Some(vec![2.0]),
            up: test_matrix(1, 2, &[0.0; 2]),
            up_bias: Some(vec![3.0]),
            down: test_matrix(2, 1, &[1.0, 2.0]),
            down_bias: Some(vec![0.5, -0.25]),
        };
        let expert1 = QwenMlp {
            gate: test_matrix(1, 2, &[0.0; 2]),
            gate_bias: Some(vec![1.0]),
            up: test_matrix(1, 2, &[0.0; 2]),
            up_bias: Some(vec![1.0]),
            down: test_matrix(2, 1, &[10.0, 20.0]),
            down_bias: Some(vec![1.0, 1.0]),
        };
        let moe = QwenMoe {
            router: test_matrix(2, 2, &[4.0, 0.0, 0.0, 1.0]),
            router_bias: Some(vec![1.0, 0.0]),
            experts: vec![expert0, expert1],
            shared: None,
            shared_gate: None,
            shared_gate_bias: None,
            top_k: 1,
            norm_topk_prob: true,
        };

        let output = moe.forward(&[vec![1.0, 0.0]]).unwrap();
        let route_weight = 5.0f32.exp() / (5.0f32.exp() + 0.0f32.exp());
        let hidden = silu(2.0) * 3.0;

        assert_close(output[0][0], route_weight * (hidden + 0.5));
        assert_close(output[0][1], route_weight * (hidden * 2.0 - 0.25));
    }

    #[test]
    fn qwen_moe_applies_shared_expert_gate() {
        let routed = QwenMlp {
            gate: test_matrix(1, 2, &[0.0; 2]),
            gate_bias: Some(vec![0.0]),
            up: test_matrix(1, 2, &[0.0; 2]),
            up_bias: Some(vec![0.0]),
            down: test_matrix(2, 1, &[0.0; 2]),
            down_bias: None,
        };
        let shared = QwenMlp {
            gate: test_matrix(1, 2, &[0.0; 2]),
            gate_bias: Some(vec![2.0]),
            up: test_matrix(1, 2, &[0.0; 2]),
            up_bias: Some(vec![3.0]),
            down: test_matrix(2, 1, &[1.0, 2.0]),
            down_bias: None,
        };
        let moe = QwenMoe {
            router: test_matrix(1, 2, &[0.0, 0.0]),
            router_bias: None,
            experts: vec![routed],
            shared: Some(shared),
            shared_gate: Some(test_matrix(1, 2, &[4.0, 0.0])),
            shared_gate_bias: Some(vec![1.0]),
            top_k: 1,
            norm_topk_prob: true,
        };

        let output = moe.forward(&[vec![1.0, 0.0]]).unwrap();
        let shared_scale = sigmoid(5.0);
        let hidden = silu(2.0) * 3.0;

        assert_close(output[0][0], shared_scale * hidden);
        assert_close(output[0][1], shared_scale * hidden * 2.0);
    }

    #[test]
    fn softmax_is_stable_for_large_scores() {
        let mut values = vec![1000.0, 1000.0];
        softmax_in_place(&mut values);

        assert_close(values[0], 0.5);
        assert_close(values[1], 0.5);
    }

    #[test]
    fn seeded_top_k_sampling_is_deterministic() {
        let logits = [0.0, 1.0, 2.0, 3.0];
        let left = (0..8)
            .map(|idx| sample_from_logits(&logits, 0.8, 1.0, Some(2), Some(7 + idx)).unwrap())
            .collect::<Vec<_>>();
        let right = (0..8)
            .map(|idx| sample_from_logits(&logits, 0.8, 1.0, Some(2), Some(7 + idx)).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(left, right);
        assert!(left.iter().all(|token| *token == 2 || *token == 3));
    }

    #[test]
    fn top_p_zero_sampling_selects_top_candidate() {
        let logits = [0.0, 3.0, 1.0, 2.0];

        let sampled = (0..8)
            .map(|idx| sample_from_logits(&logits, 1.0, 0.0, None, Some(17 + idx)).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(sampled, vec![1; 8]);
    }

    fn test_matrix(rows: usize, cols: usize, data: &[f32]) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        Matrix {
            rows,
            cols,
            data: data.to_vec(),
        }
    }

    fn write_reference_qwen(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_qwen_gguf(path, tensors);
    }

    fn write_reference_qwen_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_up_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_down_exps.weight", vec![2, 2, 2], &[0.0; 8]),
        ];
        write_qwen_moe_gguf(path, tensors);
    }

    fn write_reference_qwen_moe_packed_gate_up(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_up_exps.weight", vec![2, 4, 2], &[0.0; 16]),
            tensor_f16("blk.0.ffn_down_exps.weight", vec![2, 2, 2], &[0.0; 8]),
        ];
        write_qwen_moe_gguf(path, tensors);
    }

    fn write_reference_qwen_equal_custom_kv(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![4, 3],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![4, 2], &[0.0; 8]),
            tensor_f16("blk.0.attn_k.weight", vec![4, 2], &[0.0; 8]),
            tensor_f16("blk.0.attn_v.weight", vec![4, 2], &[0.0; 8]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 4], &[0.0; 8]),
            tensor_f16("blk.0.ffn_gate.weight", vec![4, 4], &[0.0; 16]),
            tensor_f16("blk.0.ffn_up.weight", vec![4, 4], &[0.0; 16]),
            tensor_f16("blk.0.ffn_down.weight", vec![4, 4], &[0.0; 16]),
        ];
        write_qwen_custom_kv_gguf(path, tensors, 2, "cpu-reference-qwen-equal-custom-kv");
    }

    fn write_reference_qwen_unequal_custom_kv(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![4, 3],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![4, 2], &[0.0; 8]),
            tensor_f16("blk.0.attn_k.weight", vec![4, 2], &[0.0; 8]),
            tensor_f16("blk.0.attn_v.weight", vec![4, 3], &[0.0; 12]),
            tensor_f16("blk.0.attn_output.weight", vec![3, 4], &[0.0; 12]),
            tensor_f16("blk.0.ffn_gate.weight", vec![4, 4], &[0.0; 16]),
            tensor_f16("blk.0.ffn_up.weight", vec![4, 4], &[0.0; 16]),
            tensor_f16("blk.0.ffn_down.weight", vec![4, 4], &[0.0; 16]),
        ];
        write_qwen_custom_kv_gguf(path, tensors, 3, "cpu-reference-qwen-unequal-custom-kv");
    }

    fn write_reference_qwen_next_dense(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_qwen_next_gguf(path, tensors);
    }

    fn write_reference_qwen_next_gated_attention(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 4], &[0.0; 8]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_qwen_next_gguf(path, tensors);
    }

    fn write_reference_qwen_next_recurrent_ssm(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_post_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.ssm_in.weight", vec![2, 8], &[0.0; 16]),
            tensor_f16("blk.0.ssm_conv1d.weight", vec![1, 6], &[0.0; 6]),
            tensor_f32("blk.0.ssm_dt.bias", vec![1], &[0.0]),
            tensor_f32("blk.0.ssm_a", vec![1], &[0.0]),
            tensor_f16("blk.0.ssm_ba.weight", vec![2, 2], &[0.0; 4]),
            tensor_f32("blk.0.ssm_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.ssm_out.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_qwen_next_recurrent_ssm_gguf(path, tensors);
    }

    fn write_reference_qwen_next_recurrent_ssm_optimized(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_post_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_qkv.weight", vec![2, 6], &[0.0; 12]),
            tensor_f16("blk.0.attn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ssm_conv1d.weight", vec![1, 6], &[0.0; 6]),
            tensor_f32("blk.0.ssm_dt.bias", vec![1], &[0.0]),
            tensor_f32("blk.0.ssm_a", vec![1], &[0.0]),
            tensor_f16("blk.0.ssm_ba.weight", vec![2, 2], &[0.0; 4]),
            tensor_f32("blk.0.ssm_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.ssm_out.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_qwen_next_recurrent_ssm_gguf(path, tensors);
    }

    fn write_reference_mistral_dense_aliases(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "model.embed_tokens.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f16(
                "lm_head.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.input_layernorm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32(
                "blk.0.post_attention_layernorm.weight",
                vec![2],
                &[1.0, 1.0],
            ),
            tensor_f16("blk.0.self_attn.q_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.self_attn.k_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.self_attn.v_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.self_attn.o_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.mlp.gate_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.mlp.up_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.mlp.down_proj.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_mistral_dense_alias_gguf(path, tensors);
    }

    fn write_reference_mistral_packed_aliases(path: &Path) {
        write_reference_mistral_packed_aliases_with_names(
            path,
            "blk.0.self_attn.qkv_proj.weight",
            "blk.0.self_attn.qkv_proj.bias",
            "blk.0.mlp.gate_up_proj.weight",
        );
    }

    fn write_reference_mistral_alternate_packed_aliases(path: &Path) {
        write_reference_mistral_packed_aliases_with_names(
            path,
            "blk.0.self_attn.W_pack.weight",
            "blk.0.self_attn.W_pack.bias",
            "blk.0.mlp.w1w3.weight",
        );
    }

    fn write_reference_mistral_packed_aliases_with_names(
        path: &Path,
        qkv_weight: &str,
        qkv_bias: &str,
        ffn_weight: &str,
    ) {
        let tensors = vec![
            tensor_f16(
                "model.embed_tokens.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f16(
                "lm_head.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.input_layernorm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32(
                "blk.0.post_attention_layernorm.weight",
                vec![2],
                &[1.0, 1.0],
            ),
            tensor_f16(
                qkv_weight,
                vec![2, 6],
                &[1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
            ),
            tensor_f32(qkv_bias, vec![6], &[1.0, 2.0, 0.0, 1.0, 2.0, 0.0]),
            tensor_f16("blk.0.self_attn.o_proj.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16(
                ffn_weight,
                vec![2, 4],
                &[1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            ),
            tensor_f16("blk.0.mlp.down_proj.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_mistral_dense_alias_gguf(path, tensors);
    }

    fn write_reference_deepseek_dense(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_deepseek_gguf(path, tensors, false);
    }

    fn write_reference_glm_dense(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_glm_gguf(path, tensors);
    }

    fn write_reference_glm_flash_dense(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_glm_flash_gguf(path, tensors);
    }

    fn write_reference_glm_flash_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_up_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_down_exps.weight", vec![2, 2, 2], &[0.0; 8]),
        ];
        write_glm_flash_moe_gguf(path, tensors);
    }

    fn write_reference_glm_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_up_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_down_exps.weight", vec![2, 2, 2], &[0.0; 8]),
        ];
        write_glm_moe_gguf(path, tensors);
    }

    fn write_reference_mixtral_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_up_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_down_exps.weight", vec![2, 2, 2], &[0.0; 8]),
        ];
        write_mixtral_moe_gguf(path, tensors);
    }

    fn write_reference_mixtral_per_expert_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.0.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.0.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.0.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate.1.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_up.1.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_down.1.weight", vec![2, 2], &[0.0; 4]),
        ];
        write_mixtral_moe_gguf(path, tensors);
    }

    fn write_reference_mixtral_alias_per_expert_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.block_sparse_moe.gate.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16(
                "blk.0.block_sparse_moe.experts.0.w1.weight",
                vec![2, 2],
                &[0.0; 4],
            ),
            tensor_f16(
                "blk.0.block_sparse_moe.experts.0.w3.weight",
                vec![2, 2],
                &[0.0; 4],
            ),
            tensor_f16(
                "blk.0.block_sparse_moe.experts.0.w2.weight",
                vec![2, 2],
                &[0.0; 4],
            ),
            tensor_f16(
                "blk.0.block_sparse_moe.experts.1.w1.weight",
                vec![2, 2],
                &[0.0; 4],
            ),
            tensor_f16(
                "blk.0.block_sparse_moe.experts.1.w3.weight",
                vec![2, 2],
                &[0.0; 4],
            ),
            tensor_f16(
                "blk.0.block_sparse_moe.experts.1.w2.weight",
                vec![2, 2],
                &[0.0; 4],
            ),
        ];
        write_mixtral_moe_gguf(path, tensors);
    }

    fn write_reference_deepseek_moe(path: &Path) {
        let tensors = vec![
            tensor_f16(
                "token_embd.weight",
                vec![2, 3],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            tensor_f32("output_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.attn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f32("blk.0.ffn_norm.weight", vec![2], &[1.0, 1.0]),
            tensor_f16("blk.0.attn_q.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_k.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_v.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.attn_output.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_inp.weight", vec![2, 2], &[0.0; 4]),
            tensor_f16("blk.0.ffn_gate_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_up_exps.weight", vec![2, 2, 2], &[0.0; 8]),
            tensor_f16("blk.0.ffn_down_exps.weight", vec![2, 2, 2], &[0.0; 8]),
        ];
        write_deepseek_gguf(path, tensors, true);
    }

    fn write_qwen_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-qwen");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 2);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen2.attention.layer_norm_rms_epsilon", 1.0e-6);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-qwen-next");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 2);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "qwen3next.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_recurrent_ssm_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 20);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(
            &mut bytes,
            "general.name",
            "cpu-reference-qwen-next-recurrent-ssm",
        );
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 128);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 2);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "qwen3next.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3next.ssm.conv_kernel", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.inner_size", 2);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 2);
        write_kv_u32(&mut bytes, "qwen3next.ssm.time_step_rank", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.group_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_u32(&mut bytes, "tokenizer.ggml.unknown_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_mistral_dense_alias_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "mistral");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-mistral-aliases");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mistral.context_length", 16);
        write_kv_u32(&mut bytes, "mistral.embedding_length", 2);
        write_kv_u32(&mut bytes, "mistral.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "mistral.block_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "mistral.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_moe_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 16);

        write_kv_string(&mut bytes, "general.architecture", "qwen3moe");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-qwen-moe");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3moe.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3moe.embedding_length", 2);
        write_kv_u32(&mut bytes, "qwen3moe.expert_feed_forward_length", 2);
        write_kv_u32(&mut bytes, "qwen3moe.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "qwen3moe.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_f32(&mut bytes, "qwen3moe.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3moe.expert_count", 2);
        write_kv_u32(&mut bytes, "qwen3moe.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_custom_kv_gguf(
        path: &Path,
        tensors: Vec<TestTensor>,
        value_head_dim: u32,
        name: &str,
    ) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 16);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", name);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 4);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.key_length", 2);
        write_kv_u32(&mut bytes, "qwen2.attention.value_length", value_head_dim);
        write_kv_f32(&mut bytes, "qwen2.attention.layer_norm_rms_epsilon", 1.0e-6);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_deepseek_gguf(path: &Path, tensors: Vec<TestTensor>, moe: bool) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, if moe { 15 } else { 13 });

        write_kv_string(&mut bytes, "general.architecture", "deepseek");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-deepseek");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "deepseek.context_length", 16);
        write_kv_u32(&mut bytes, "deepseek.embedding_length", 2);
        write_kv_u32(&mut bytes, "deepseek.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "deepseek.block_count", 1);
        write_kv_u32(&mut bytes, "deepseek.attention.head_count", 1);
        write_kv_u32(&mut bytes, "deepseek.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "deepseek.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        if moe {
            write_kv_u32(&mut bytes, "deepseek.expert_count", 2);
            write_kv_u32(&mut bytes, "deepseek.expert_used_count", 1);
        }
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_glm_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "glm4");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-glm");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "glm4.context_length", 16);
        write_kv_u32(&mut bytes, "glm4.embedding_length", 2);
        write_kv_u32(&mut bytes, "glm4.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "glm4.block_count", 1);
        write_kv_u32(&mut bytes, "glm4.attention.head_count", 1);
        write_kv_u32(&mut bytes, "glm4.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "glm4.attention.layer_norm_rms_epsilon", 1.0e-6);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_glm_moe_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "glm4moe");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-glm-moe");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "glm4moe.context_length", 16);
        write_kv_u32(&mut bytes, "glm4moe.embedding_length", 2);
        write_kv_u32(&mut bytes, "glm4moe.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "glm4moe.block_count", 1);
        write_kv_u32(&mut bytes, "glm4moe.attention.head_count", 1);
        write_kv_u32(&mut bytes, "glm4moe.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "glm4moe.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_u32(&mut bytes, "glm4moe.expert_count", 2);
        write_kv_u32(&mut bytes, "glm4moe.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_glm_flash_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "glm4flash");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-glm-flash");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "glm4flash.context_length", 16);
        write_kv_u32(&mut bytes, "glm4flash.embedding_length", 2);
        write_kv_u32(&mut bytes, "glm4flash.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "glm4flash.block_count", 1);
        write_kv_u32(&mut bytes, "glm4flash.attention.head_count", 1);
        write_kv_u32(&mut bytes, "glm4flash.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "glm4flash.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_glm_flash_moe_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "glm4flash");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-glm-flash-moe");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "glm4flash.context_length", 16);
        write_kv_u32(&mut bytes, "glm4flash.embedding_length", 2);
        write_kv_u32(&mut bytes, "glm4flash.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "glm4flash.block_count", 1);
        write_kv_u32(&mut bytes, "glm4flash.attention.head_count", 1);
        write_kv_u32(&mut bytes, "glm4flash.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "glm4flash.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_u32(&mut bytes, "glm4flash.expert_count", 2);
        write_kv_u32(&mut bytes, "glm4flash.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_mixtral_moe_gguf(path: &Path, tensors: Vec<TestTensor>) {
        let mut data = Vec::new();
        let tensors = tensors
            .into_iter()
            .map(|mut tensor| {
                pad_to_alignment(&mut data, 32);
                tensor.offset = data.len() as u64;
                data.extend_from_slice(&tensor.bytes);
                tensor
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "mixtral");
        write_kv_string(&mut bytes, "general.name", "cpu-reference-mixtral");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mixtral.context_length", 16);
        write_kv_u32(&mut bytes, "mixtral.embedding_length", 2);
        write_kv_u32(&mut bytes, "mixtral.feed_forward_length", 2);
        write_kv_u32(&mut bytes, "mixtral.block_count", 1);
        write_kv_u32(&mut bytes, "mixtral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mixtral.attention.head_count_kv", 1);
        write_kv_f32(
            &mut bytes,
            "mixtral.attention.layer_norm_rms_epsilon",
            1.0e-6,
        );
        write_kv_u32(&mut bytes, "mixtral.expert_count", 2);
        write_kv_u32(&mut bytes, "mixtral.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b", "c"]);

        for tensor in tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    struct TestTensor {
        name: String,
        dims: Vec<u64>,
        dtype: u32,
        offset: u64,
        bytes: Vec<u8>,
    }

    fn tensor_f32(name: &str, dims: Vec<u64>, values: &[f32]) -> TestTensor {
        TestTensor {
            name: name.to_string(),
            dims,
            dtype: 0,
            offset: 0,
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        }
    }

    fn tensor_f16(name: &str, dims: Vec<u64>, values: &[f32]) -> TestTensor {
        TestTensor {
            name: name.to_string(),
            dims,
            dtype: 1,
            offset: 0,
            bytes: values
                .iter()
                .flat_map(|value| f16_bits(*value).to_le_bytes())
                .collect::<Vec<_>>(),
        }
    }

    fn f16_bits(value: f32) -> u16 {
        match value {
            0.0 => 0x0000,
            1.0 => 0x3c00,
            -1.0 => 0xbc00,
            2.0 => 0x4000,
            -2.0 => 0xc000,
            _ => panic!("test fixture only supports simple f16 values, got {value}"),
        }
    }

    fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        write_string(bytes, key);
        write_u32(bytes, 8);
        write_string(bytes, value);
    }

    fn write_kv_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_string(bytes, key);
        write_u32(bytes, 9);
        write_u32(bytes, 8);
        write_u64(bytes, values.len() as u64);
        for value in values {
            write_string(bytes, value);
        }
    }

    fn write_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        write_string(bytes, key);
        write_u32(bytes, 4);
        write_u32(bytes, value);
    }

    fn write_kv_f32(bytes: &mut Vec<u8>, key: &str, value: f32) {
        write_string(bytes, key);
        write_u32(bytes, 6);
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        write_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) {
        let remainder = bytes.len() % alignment;
        if remainder != 0 {
            bytes.extend(vec![0; alignment - remainder]);
        }
    }

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-cuda-qwen-cpu-{name}-{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "actual {actual} expected {expected}"
        );
    }
}
