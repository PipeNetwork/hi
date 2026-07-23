//! DSpark drafter for DeepSeek-V4-Flash — DeepSeek's official semi-
//! autoregressive block speculator (follow-up lever in
//! `docs/deepseek-v4-spec-decode-plan.md`), selected by `HI_DSV4_SPEC=dspark`.
//!
//! Owns: loading the `mtp.0/1/2` draft blocks from the DSpark shard trio (via
//! [`crate::safetensors`]), the ONE 3-layer draft-transformer forward per
//! proposal (target-hidden conditioning through `main_proj`, a 5-token
//! anchor+noise query block, the rank-256 vanilla Markov logit bias), and the
//! [`crate::dsv4_backend::Drafter`] implementation.
//!
//! Ported from vLLM `models/deepseek_v4/nvidia/dspark.py` +
//! `v1/worker/gpu/spec_decode/dspark/speculator.py` and the DeepSpec
//! reference (`deepspec/modeling/dspark/*`):
//!
//! - **Architecture**: one 3-LAYER draft transformer (NOT three chained
//!   predictors), each layer a full compress-ratio-1 V4 decoder block — the
//!   trunk's SWA-128 latent MQA with attention sinks, hyperconnections, and
//!   the 256-expert MoE with the `gate.bias` noaux-tc selection bias plus FP8
//!   shared experts. No indexer, no compressor. `embed_tokens` and `lm_head`
//!   are absent from the shards; both bind the TARGET GGUF's
//!   `token_embd`/`output` (full vocab, no d2t).
//! - **Conditioning**: per verified context position, the target's hc-stream
//!   hiddens at layers `dspark_target_layer_ids` (post-0-based-layer taps —
//!   `[40, 41, 42]` on the real model, no index shift) are STREAM-AVERAGED
//!   ([`DsV4Taps::aux_averaged`], vLLM's mean-pooled aux states), concatenated
//!   in ascending layer order (3 x 4096 = 12288 wide) and pushed through
//!   `main_proj` then `main_norm` -> `main_x[4096]`.
//! - **Context KV** (DFlash-style incremental): every draft layer derives its
//!   context KV from the SAME `main_x` via that layer's own `wkv` + `kv_norm`
//!   + rope at the ABSOLUTE position (vLLM `precompute_and_store_context_kv`),
//!   appended to a per-layer latent list that persists across proposals —
//!   floored at the taps base after a prefix-cache restore, truncated back to
//!   the common token prefix on divergence (rejected drafts never enter it:
//!   the verify loop truncates the taps first).
//! - **Query block**: N = `dspark_block_size` = 5 tokens — the anchor (the
//!   pending token, `ctx.tokens.last()`) then N-1 noise tokens
//!   (`dspark_noise_token_id` = 128799) — embedded via the target table,
//!   expanded across the hc streams, at absolute positions `p .. p+4` where
//!   `p` is the anchor's position (`ctx.tokens.len() - 1`). Attention is
//!   NON-CAUSAL within the block: every query sees the trailing `window`
//!   context latents PLUS the ENTIRE query block including future positions
//!   (vLLM: "include the future query tokens ... for each query token"; the
//!   training mask is `kv < anchor_pos | same block`). The 3 layers run
//!   sequentially with ordinary hc residual handoff, then the `mtp.2`
//!   hc_head collapse yields the PRE-norm head hidden per position.
//! - **Logits + Markov**: base logits = target `lm_head` over
//!   `mtp.2.norm`-normed hiddens at ALL 5 positions; then a sequential
//!   vanilla-Markov pass, left to right — `prev = anchor`; per position
//!   `logits_i += markov_w2 @ markov_w1[prev]`, `draft_i = argmax`,
//!   `prev = draft_i`. Position i predicts ABSOLUTE `p + 1 + i`, so the
//!   drafts continue directly after the pending token. `min(5, ctx.k)`
//!   drafts are returned (`sample_from_anchor` layout — every query position
//!   is a prediction; the checkpoint has no `dspark_bonus_anchor`).
//! - **Weight remap** (vLLM `_remap_dspark_name`, verbatim):
//!   `mtp.{i}.confidence_head.*` -> DROPPED (vLLM does not wire it into
//!   inference; a future dynamic-draft-length lever);
//!   `mtp.0.{main_proj,main_norm}.*` and
//!   `mtp.2.{norm,hc_head_*,markov_head.*}` -> model level; everything else
//!   -> draft layer i. E8M0 scale bytes stay raw exponent bytes
//!   (`.scale` siblings resolve as multipliers inside
//!   [`crate::safetensors`]'s fp8/fp4 dequant).
//!
//! # Weights and residency
//!
//! `HI_DSV4_DSPARK_PATH` (default `~/.hi/models/deepseek-v4-flash/dspark`, a
//! DIRECTORY holding the shard trio + `config.json`) loads through
//! [`crate::safetensors`]: FP8-e4m3 + ue8m0 dense weights dequantize to f16
//! (~0.9 GiB resident across the three blocks + `main_proj`), BF16 routers
//! and `markov_w2` stay bf16 (~66 MiB), small norms/mixers/sinks materialize
//! f32 host-side, `markov_w1` stays a host-side embedding table (row gathers
//! only), and the 3 x 256 fp4 experts repack bit-exactly into the GGUF MXFP4
//! layout (~10.3 GiB) and register in the trunk's expert pool as PINNED draft
//! layers `trunk_layers + 1 + i` (44/45/46 on the real model; 43 is the MTP
//! module's slot).
//!
//! Execution split mirrors `dsv4_mtp`: heavy linears go through a
//! [`DsV4Linear`] provider (the trunk's GPU handle in production, an exact
//! host-f32 overlay in tests) while norms/rope/attention-softmax/hc math stay
//! in host f32 shared verbatim between both, so CPU==GPU parity is a pure
//! matmul-precision comparison.

// The production consumer (the drafter) is native-cuda-gated and the CPU
// reference paths are exercised by tests; a bare default build uses little
// here, which is expected rather than a smell.
#![cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{GgufFile, GgufTensorType};

use crate::dsv4_cpu::{
    DsV4Engine, DsV4Geometry, DsV4HcFunc, DsV4Layer, DsV4Linear, DsV4MoeBlockCtx, DsV4MoeShared,
    DsV4Taps, HcParams, HcWeights, RawExperts, RawMatrix, SharedExpertWeights, TensorKey,
    dsv4_embed_row, hc_post, hc_pre_math, hyper_head_math, v4_rope_sincos,
};
use crate::qwen_cpu::{argmax, dot, rms_norm_in_place};
use crate::safetensors::{SafetensorsDtype, SafetensorsFile, bf16_to_f32, f16_to_f32};

#[cfg(feature = "native-cuda")]
use crate::dsv4_backend::{DraftContext, Drafter};
#[cfg(feature = "native-cuda")]
use crate::dsv4_cpu::DsV4TapConfig;
#[cfg(feature = "native-cuda")]
use crate::dsv4_gpu::{DeepSeekV4GpuEngine, DsV4GpuLinear, HostDenseData};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// DSpark-specific knobs. Baked defaults match the byte-verified real
/// checkpoint; `apply_config_json` overrides them from the shard directory's
/// `config.json` (the `dspark_*` keys), and [`DsparkDims::from_engine`]
/// validates the final values against the trunk.
#[derive(Clone, Debug)]
pub(crate) struct DsparkConfig {
    /// Draft decoder layers (`n_mtp_layers`, default 3 like vLLM).
    layers: usize,
    /// Query-block width; every proposal drafts up to this many tokens.
    block_size: usize,
    /// Filler token embedded at the non-anchor query slots.
    noise_token: u32,
    /// TARGET layers to tap (post-0-based-layer indices, strictly ascending;
    /// concat order = this order). NO off-by-one shift: the checkpoint's
    /// `dspark_target_layer_ids` already name post-layer hiddens (DeepSpec's
    /// `extract_context_feature` reads `hidden_states[layer_id + 1]`, the
    /// output OF layer `layer_id`, which is exactly our tap index).
    target_layers: Vec<usize>,
    /// Vanilla Markov head rank (`markov_w1`/`markov_w2` inner dim).
    markov_rank: usize,
}

impl DsparkConfig {
    /// Census-verified values for the real DeepSeek-V4-Flash DSpark shards.
    fn real_default() -> Self {
        Self {
            layers: 3,
            block_size: 5,
            noise_token: 128_799,
            target_layers: vec![40, 41, 42],
            markov_rank: 256,
        }
    }

    /// Override the baked defaults from the checkpoint's `config.json`
    /// (`dspark_*` keys at the top level, mirroring vLLM's config reads).
    fn apply_config_json(&mut self, path: &Path) -> Result<()> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let root: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if let Some(v) = json_usize(&root, "n_mtp_layers") {
            self.layers = v;
        }
        if let Some(v) = json_usize(&root, "dspark_block_size") {
            self.block_size = v;
        }
        if let Some(v) = json_usize(&root, "dspark_noise_token_id") {
            self.noise_token =
                u32::try_from(v).context("dspark_noise_token_id does not fit u32")?;
        }
        if let Some(v) = json_usize(&root, "dspark_markov_rank") {
            self.markov_rank = v;
        }
        if let Some(layers) = json_usize_array(&root, "dspark_target_layer_ids") {
            self.target_layers = layers;
        }
        // The 1+N "bonus anchor" query layout (speculators-format checkpoints)
        // is a different sampling alignment; this port implements only the
        // official sample-from-anchor layout the real checkpoint uses.
        if root
            .get("dspark_bonus_anchor")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            bail!("dspark_bonus_anchor checkpoints (1+N query layout) are not supported");
        }
        Ok(())
    }

    /// Validate against the loaded trunk (layer count for the taps, vocab for
    /// the noise token).
    fn validate(&self, trunk_layers: usize, vocab: usize) -> Result<()> {
        if self.layers == 0 || self.block_size == 0 || self.markov_rank == 0 {
            bail!("DSpark config has a zero dimension: {self:?}");
        }
        if self.target_layers.is_empty() {
            bail!("DSpark target_layer_ids is empty");
        }
        if !self.target_layers.windows(2).all(|w| w[0] < w[1]) {
            bail!(
                "DSpark target_layer_ids {:?} must be strictly ascending (main_proj consumes the concat in layer order)",
                self.target_layers
            );
        }
        if let Some(&last) = self.target_layers.last()
            && last >= trunk_layers
        {
            bail!("DSpark target layer {last} is outside the trunk's {trunk_layers} layers");
        }
        if (self.noise_token as usize) >= vocab {
            bail!(
                "DSpark noise token {} is outside the target vocab {vocab}",
                self.noise_token
            );
        }
        Ok(())
    }
}

fn json_usize(value: &serde_json::Value, key: &str) -> Option<usize> {
    value
        .get(key)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
}

fn json_usize_array(value: &serde_json::Value, key: &str) -> Option<Vec<usize>> {
    let items = value.get(key)?.as_array()?;
    items
        .iter()
        .map(|item| item.as_u64().and_then(|v| usize::try_from(v).ok()))
        .collect()
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Every dimension the loader validates shard shapes against and the module
/// forwards with. Trunk-shared geometry comes from the loaded engine (the
/// draft blocks share it exactly); constructible directly for the real-shard
/// census test.
pub(crate) struct DsparkDims {
    pub(crate) geometry: DsV4Geometry,
    /// Routed-expert FFN width (`gate_exps.out_dim` on the trunk).
    pub(crate) expert_ff: usize,
    /// Shared-expert FFN width; V4 draft blocks always carry shared experts.
    pub(crate) shared_ff: usize,
    /// Trunk decoder layer count. Draft layer i registers its experts as pool
    /// layer `trunk_layers + 1 + i` (44/45/46 on the real model — 43 is the
    /// MTP module's slot).
    pub(crate) trunk_layers: usize,
    /// Non-compress rope base (`rope.freq_base`; the draft blocks are
    /// compress-ratio-1 layers, so they carry the trunk's raw base — 10000).
    pub(crate) rope_base: f32,
    /// Trunk swiglu clamp (all layers carry 10.0 on the real model).
    pub(crate) swiglu_clamp: f32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_eps: f32,
    config: DsparkConfig,
}

impl DsparkDims {
    /// Derive from a loaded trunk engine and a validated DSpark config.
    pub(crate) fn from_engine<L: DsV4Linear>(
        engine: &DsV4Engine<L>,
        config: DsparkConfig,
    ) -> Result<Self> {
        let layers = engine.layers();
        let first = layers
            .first()
            .ok_or_else(|| anyhow!("trunk engine has no layers"))?;
        let last = layers.last().expect("non-empty checked above");
        if first.gate_exps.out_dim == 0 {
            bail!("trunk expert tensors have zero FFN width");
        }
        let shared = last.shared.as_ref().ok_or_else(|| {
            anyhow!("DSpark draft blocks require the trunk's shared expert shape")
        })?;
        let geometry = engine.geometry().clone();
        if geometry.window.is_none() {
            bail!("DSpark draft blocks require a sliding window (attention.sliding_window)");
        }
        config.validate(layers.len(), geometry.vocab)?;
        Ok(Self {
            geometry,
            expert_ff: first.gate_exps.out_dim,
            shared_ff: shared.gate.rows,
            trunk_layers: layers.len(),
            // Layer 0 is a ratio-0 layer, so it carries the non-compress base.
            rope_base: first.rope_base,
            swiglu_clamp: last.swiglu_clamp,
            rms_eps: engine.rms_eps(),
            hc_eps: engine.hc_eps(),
            config,
        })
    }

    fn hc_mix_rows(&self) -> usize {
        let hc = self.geometry.hc;
        hc * hc + 2 * hc
    }

    fn q_dim(&self) -> usize {
        self.geometry.heads * self.geometry.head_dim
    }

    /// `main_proj` input width: one stream-averaged row per tapped layer.
    fn main_in(&self) -> usize {
        self.geometry.embed * self.config.target_layers.len()
    }

    /// Pool layer index for draft layer `i` (44/45/46 on the real model).
    fn pool_layer(&self, layer: usize) -> usize {
        self.trunk_layers + 1 + layer
    }

    /// Synthesized GGUF-style name for draft layer `layer`'s pooled expert
    /// tensor — the key the expert pool slices it under.
    fn expert_name(&self, layer: usize, proj: &str) -> String {
        format!("blk.{}.ffn_{proj}_exps.weight", self.pool_layer(layer))
    }
}

// ---------------------------------------------------------------------------
// Shard tensor names (census-verified against the real trio)
// ---------------------------------------------------------------------------

mod names {
    /// `mtp.{layer}.{part}` — the per-layer decoder-block names.
    pub(super) fn layer(layer: usize, part: &str) -> String {
        format!("mtp.{layer}.{part}")
    }

    /// `mtp.{layer}.hc_{which}_{part}` (no `.weight` suffix in the shards).
    pub(super) fn hc(layer: usize, which: &str, part: &str) -> String {
        format!("mtp.{layer}.hc_{which}_{part}")
    }

    pub(super) fn expert(layer: usize, index: usize, proj: &str) -> String {
        format!("mtp.{layer}.ffn.experts.{index}.{proj}.weight")
    }

    pub(super) const MAIN_PROJ: &str = "mtp.0.main_proj.weight";
    pub(super) const MAIN_NORM: &str = "mtp.0.main_norm.weight";

    /// Model-level head-stack names live on the LAST draft layer.
    pub(super) fn head(last_layer: usize, part: &str) -> String {
        format!("mtp.{last_layer}.{part}")
    }
}

// ---------------------------------------------------------------------------
// Shard trio access
// ---------------------------------------------------------------------------

/// The DSpark checkpoint directory's `.safetensors` files, opened together so
/// tensors resolve by name regardless of which shard holds them (`mtp.0/1/2`
/// live in shards 46/47/48 of the real split; the synthetic fixture mirrors
/// the trio).
pub(crate) struct DsparkShards {
    files: Vec<SafetensorsFile>,
}

impl DsparkShards {
    pub(crate) fn open_dir(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading DSpark checkpoint directory {}", dir.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!(
                "DSpark checkpoint directory {} contains no .safetensors shards",
                dir.display()
            );
        }
        let files = paths
            .iter()
            .map(SafetensorsFile::open)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { files })
    }

    /// The shard holding `name` (scale siblings always sit next to their
    /// weight in the same shard, so per-file fp8/fp4 dequant just works).
    fn file_for(&self, name: &str) -> Result<&SafetensorsFile> {
        self.files
            .iter()
            .find(|file| file.info(name).is_some())
            .ok_or_else(|| anyhow!("DSpark shards are missing tensor {name}"))
    }

    fn info(&self, name: &str) -> Option<&crate::safetensors::SafetensorsTensor> {
        self.files.iter().find_map(|file| file.info(name))
    }

    /// Every tensor name across the trio (the census test's iteration).
    #[cfg(test)]
    fn names(&self) -> Vec<String> {
        self.files
            .iter()
            .flat_map(|file| file.names().map(str::to_string))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Loaded payloads
// ---------------------------------------------------------------------------

/// A dense weight payload in the dtype it should be served in: F32 stays
/// exact (the synthetic fixture and the CPU host reference), F16 carries
/// fp8-block-dequantized shard weights, Bf16 carries the shard's bf16
/// tensors (router, markov_w2) verbatim.
pub(crate) enum DsparkPayload {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
}

impl DsparkPayload {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(bits) | Self::Bf16(bits) => bits.len(),
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            Self::F32(values) => values.len() * 4,
            Self::F16(bits) | Self::Bf16(bits) => bits.len() * 2,
        }
    }

    /// Exact f32 view (bf16/f16 widen losslessly) — the CPU host reference.
    #[cfg_attr(not(test), allow(dead_code))]
    fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values.clone(),
            Self::F16(bits) => bits.iter().map(|&bits| f16_to_f32(bits)).collect(),
            Self::Bf16(bits) => bits.iter().map(|&bits| bf16_to_f32(bits)).collect(),
        }
    }
}

/// One dense matrix destined for provider residency.
pub(crate) struct DsparkDenseTensor {
    pub(crate) matrix: RawMatrix,
    /// `Some(rank)` uploads block-diagonally (the grouped output projection).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) grouped_rank: Option<usize>,
    pub(crate) payload: DsparkPayload,
}

/// One packed rank-3 expert tensor in the GGUF layout (expert-major slices,
/// innermost = the in dim): MXFP4 from the shard's fp4 (bit-exact repack), or
/// raw F32 for the synthetic fixture.
pub(crate) struct DsparkExpertTensor {
    pub(crate) experts: RawExperts,
    pub(crate) expert_count: usize,
    /// Draft layer index (0..layers) — its pool layer is
    /// [`DsparkDims::pool_layer`].
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) layer: usize,
    /// Pool projection id: 0 = gate, 1 = up, 2 = down.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) proj: u8,
    pub(crate) dtype: GgufTensorType,
    pub(crate) bytes: Vec<u8>,
}

/// The vanilla Markov head's embedding half (`markov_w1`): host-resident, row
/// gathers only (one rank-wide row per previously drafted token).
pub(crate) struct MarkovTable {
    vocab: usize,
    rank: usize,
    payload: DsparkPayload,
}

impl MarkovTable {
    /// `markov_w1[token]` as exact f32.
    fn row(&self, token: u32) -> Result<Vec<f32>> {
        let token = token as usize;
        if token >= self.vocab {
            bail!("Markov table row {token} outside vocab {}", self.vocab);
        }
        let range = token * self.rank..(token + 1) * self.rank;
        Ok(match &self.payload {
            DsparkPayload::F32(values) => values[range].to_vec(),
            DsparkPayload::F16(bits) => bits[range].iter().map(|&b| f16_to_f32(b)).collect(),
            DsparkPayload::Bf16(bits) => bits[range].iter().map(|&b| bf16_to_f32(b)).collect(),
        })
    }
}

/// The DSpark module's weight handles: `config.layers` full [`DsV4Layer`]s
/// (compressor/indexer `None`) plus the model-level attachments. Heavy
/// matrices are name-keyed [`RawMatrix`] handles served by whatever provider
/// took residency of the payloads.
pub(crate) struct DsparkWeights {
    pub(crate) layers: Vec<DsV4Layer>,
    pub(crate) main_proj: RawMatrix,
    pub(crate) main_norm: Vec<f32>,
    /// `mtp.2.hc_head_*`: collapses the hc streams after the last layer.
    pub(crate) hc_head: HcWeights,
    /// `mtp.2.norm`: the final norm before the target's lm head.
    pub(crate) norm: Vec<f32>,
    pub(crate) markov_w1: MarkovTable,
    /// `markov_w2` (`[vocab, rank]`, provider-resident): rank -> vocab bias.
    pub(crate) markov_w2: RawMatrix,
}

/// Everything [`load_dspark`] produces: the module's weight handles + small
/// host tensors, and the heavy payloads a provider takes residency of.
pub(crate) struct DsparkLoad {
    pub(crate) weights: DsparkWeights,
    pub(crate) dense: Vec<DsparkDenseTensor>,
    pub(crate) experts: Vec<DsparkExpertTensor>,
}

impl DsparkLoad {
    /// Resident (dense f16/f32/bf16) payload bytes, markov_w1 included.
    pub(crate) fn resident_bytes(&self) -> usize {
        self.dense
            .iter()
            .map(|entry| entry.payload.byte_len())
            .sum::<usize>()
            + self.weights.markov_w1.payload.byte_len()
    }

    /// Packed expert payload bytes.
    pub(crate) fn expert_bytes(&self) -> usize {
        self.experts.iter().map(|entry| entry.bytes.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load and shape-validate the DSpark shard trio against the trunk-derived
/// dims. Dense fp8 dequantizes to f16 (scale siblings auto-resolved,
/// MULTIPLIER semantics), bf16/f32 pass through, fp4 experts repack
/// bit-exactly into the GGUF MXFP4 stream the expert pool consumes.
/// `mtp.{i}.confidence_head.*` is deliberately never read (vLLM drops it).
pub(crate) fn load_dspark(shards: &DsparkShards, dims: &DsparkDims) -> Result<DsparkLoad> {
    let g = &dims.geometry;
    let cfg = &dims.config;
    let mut dense = Vec::new();
    let mut load_dense = |name: String, rows: usize, cols: usize, grouped_rank: Option<usize>| {
        let payload = dense_payload(shards, &name, rows, cols)?;
        let matrix = RawMatrix { name, rows, cols };
        dense.push(DsparkDenseTensor {
            matrix: matrix.clone(),
            grouped_rank,
            payload,
        });
        Ok::<RawMatrix, anyhow::Error>(matrix)
    };

    let mut layers = Vec::with_capacity(cfg.layers);
    let mut experts = Vec::with_capacity(cfg.layers * 3);
    for i in 0..cfg.layers {
        let q_a = load_dense(names::layer(i, "attn.wq_a.weight"), g.q_lora, g.embed, None)?;
        let q_b = load_dense(
            names::layer(i, "attn.wq_b.weight"),
            dims.q_dim(),
            g.q_lora,
            None,
        )?;
        let kv = load_dense(
            names::layer(i, "attn.wkv.weight"),
            g.head_dim,
            g.embed,
            None,
        )?;
        let out_a = load_dense(
            names::layer(i, "attn.wo_a.weight"),
            g.o_groups * g.o_rank,
            dims.q_dim() / g.o_groups,
            Some(g.o_rank),
        )?;
        let out_b = load_dense(
            names::layer(i, "attn.wo_b.weight"),
            g.embed,
            g.o_groups * g.o_rank,
            None,
        )?;
        let router = load_dense(names::layer(i, "ffn.gate.weight"), g.experts, g.embed, None)?;
        let shared_gate = load_dense(
            names::layer(i, "ffn.shared_experts.w1.weight"),
            dims.shared_ff,
            g.embed,
            None,
        )?;
        let shared_down = load_dense(
            names::layer(i, "ffn.shared_experts.w2.weight"),
            g.embed,
            dims.shared_ff,
            None,
        )?;
        let shared_up = load_dense(
            names::layer(i, "ffn.shared_experts.w3.weight"),
            dims.shared_ff,
            g.embed,
            None,
        )?;

        layers.push(DsV4Layer {
            attn_norm: small_vector(shards, &names::layer(i, "attn_norm.weight"), g.embed)?,
            ffn_norm: small_vector(shards, &names::layer(i, "ffn_norm.weight"), g.embed)?,
            hc_attn: load_hc(shards, dims, i, "attn", dims.hc_mix_rows(), 3)?,
            hc_ffn: load_hc(shards, dims, i, "ffn", dims.hc_mix_rows(), 3)?,
            q_a,
            q_a_norm: small_vector(shards, &names::layer(i, "attn.q_norm.weight"), g.q_lora)?,
            q_b,
            kv,
            kv_norm: small_vector(shards, &names::layer(i, "attn.kv_norm.weight"), g.head_dim)?,
            sinks: Some(small_vector(
                shards,
                &names::layer(i, "attn.attn_sink"),
                g.heads,
            )?),
            out_a,
            out_b,
            rope_base: dims.rope_base,
            compressor: None,
            indexer: None,
            router,
            probs_bias: Some(small_vector(
                shards,
                &names::layer(i, "ffn.gate.bias"),
                g.experts,
            )?),
            tid2eid: None,
            gate_exps: RawExperts {
                name: dims.expert_name(i, "gate"),
                in_dim: g.embed,
                out_dim: dims.expert_ff,
            },
            up_exps: RawExperts {
                name: dims.expert_name(i, "up"),
                in_dim: g.embed,
                out_dim: dims.expert_ff,
            },
            down_exps: RawExperts {
                name: dims.expert_name(i, "down"),
                in_dim: dims.expert_ff,
                out_dim: g.embed,
            },
            shared: Some(SharedExpertWeights {
                gate: shared_gate,
                up: shared_up,
                down: shared_down,
            }),
            swiglu_clamp: dims.swiglu_clamp,
        });

        experts.push(load_expert_tensor(
            shards,
            dims,
            i,
            "w1",
            0,
            dims.expert_ff,
            g.embed,
        )?);
        experts.push(load_expert_tensor(
            shards,
            dims,
            i,
            "w3",
            1,
            dims.expert_ff,
            g.embed,
        )?);
        experts.push(load_expert_tensor(
            shards,
            dims,
            i,
            "w2",
            2,
            g.embed,
            dims.expert_ff,
        )?);
    }

    let last = cfg.layers - 1;
    let main_proj = load_dense(names::MAIN_PROJ.to_string(), g.embed, dims.main_in(), None)?;
    let markov_w2 = load_dense(
        names::head(last, "markov_head.markov_w2.weight"),
        g.vocab,
        cfg.markov_rank,
        None,
    )?;
    let markov_w1_name = names::head(last, "markov_head.markov_w1.weight");
    let markov_w1 = MarkovTable {
        vocab: g.vocab,
        rank: cfg.markov_rank,
        payload: dense_payload(shards, &markov_w1_name, g.vocab, cfg.markov_rank)?,
    };

    let weights = DsparkWeights {
        layers,
        main_proj,
        main_norm: small_vector(shards, names::MAIN_NORM, g.embed)?,
        hc_head: load_hc(shards, dims, last, "head", g.hc, 1)?,
        norm: small_vector(shards, &names::head(last, "norm.weight"), g.embed)?,
        markov_w1,
        markov_w2,
    };
    Ok(DsparkLoad {
        weights,
        dense,
        experts,
    })
}

/// Read a dense 2-D weight in serving dtype, validating the safetensors
/// `[out, in]` shape (row-major — identical memory layout to the GGUF's
/// `[ne0 = in, ne1 = out]`).
fn dense_payload(
    shards: &DsparkShards,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DsparkPayload> {
    let file = shards.file_for(name)?;
    let info = file.info(name).expect("file_for checked presence");
    if info.shape != [rows, cols] {
        bail!(
            "DSpark tensor {name} has shape {:?}; expected [{rows}, {cols}]",
            info.shape
        );
    }
    let payload = match info.dtype {
        SafetensorsDtype::F8E4M3 => DsparkPayload::F16(file.fp8_block_scaled_f16(name)?),
        SafetensorsDtype::F32 => DsparkPayload::F32(file.tensor_f32(name)?),
        SafetensorsDtype::F16 => DsparkPayload::F16(file.tensor_f16(name)?),
        SafetensorsDtype::BF16 => DsparkPayload::Bf16(
            file.bytes(name)?
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect(),
        ),
        other => bail!(
            "DSpark tensor {name} has unsupported dtype {}",
            other.name()
        ),
    };
    if payload.len() != rows * cols {
        bail!(
            "DSpark tensor {name} dequantized to {} values; expected {}",
            payload.len(),
            rows * cols
        );
    }
    Ok(payload)
}

/// Read a small 1-D tensor (norms, sinks, bias) as exact f32.
fn small_vector(shards: &DsparkShards, name: &str, len: usize) -> Result<Vec<f32>> {
    let file = shards.file_for(name)?;
    let info = file.info(name).expect("file_for checked presence");
    if info.shape != [len] {
        bail!(
            "DSpark tensor {name} has shape {:?}; expected [{len}]",
            info.shape
        );
    }
    file.tensor_f32(name)
}

/// Load one hyper-connection mixer triple (`hc_{attn,ffn,head}_*`; F32 in the
/// shards, unsuffixed names).
fn load_hc(
    shards: &DsparkShards,
    dims: &DsparkDims,
    layer: usize,
    which: &str,
    rows: usize,
    scale_len: usize,
) -> Result<HcWeights> {
    let cols = dims.geometry.hc * dims.geometry.embed;
    let fn_name = names::hc(layer, which, "fn");
    let file = shards.file_for(&fn_name)?;
    let info = file.info(&fn_name).expect("file_for checked presence");
    if info.shape != [rows, cols] {
        bail!(
            "DSpark tensor {fn_name} has shape {:?}; expected [{rows}, {cols}]",
            info.shape
        );
    }
    Ok(HcWeights {
        func: DsV4HcFunc::from_parts(rows, cols, file.tensor_f32(&fn_name)?)?,
        base: small_vector(shards, &names::hc(layer, which, "base"), rows)?,
        scale: small_vector(shards, &names::hc(layer, which, "scale"), scale_len)?,
    })
}

/// Load one draft layer's per-expert `w{1,3,2}` weights of one projection
/// into a single packed rank-3 blob in the GGUF layout: fp4 shards repack
/// bit-exactly to MXFP4 (one 17-byte block per 32 in-dim values,
/// expert-major), the f32 fixture concatenates raw little-endian rows.
fn load_expert_tensor(
    shards: &DsparkShards,
    dims: &DsparkDims,
    layer: usize,
    shard_proj: &str,
    proj: u8,
    out_dim: usize,
    in_dim: usize,
) -> Result<DsparkExpertTensor> {
    let gguf_proj = match proj {
        0 => "gate",
        1 => "up",
        _ => "down",
    };
    let expert_count = dims.geometry.experts;
    let mut bytes = Vec::new();
    let mut dtype = None;
    for index in 0..expert_count {
        let name = names::expert(layer, index, shard_proj);
        let file = shards.file_for(&name)?;
        let info = file.info(&name).expect("file_for checked presence");
        match info.dtype {
            SafetensorsDtype::I8 => {
                // Packed fp4 [out, in/2] + ue8m0 scale sibling.
                if info.shape != [out_dim, in_dim / 2] {
                    bail!(
                        "DSpark tensor {name} has shape {:?}; expected [{out_dim}, {}] (packed fp4)",
                        info.shape,
                        in_dim / 2
                    );
                }
                if dtype.replace(GgufTensorType::MXFP4) == Some(GgufTensorType::F32) {
                    bail!("DSpark expert tensors mix fp4 and f32 payloads");
                }
                bytes.extend_from_slice(&file.fp4_to_gguf_mxfp4(&name)?);
            }
            SafetensorsDtype::F32 => {
                if info.shape != [out_dim, in_dim] {
                    bail!(
                        "DSpark tensor {name} has shape {:?}; expected [{out_dim}, {in_dim}]",
                        info.shape
                    );
                }
                if dtype.replace(GgufTensorType::F32) == Some(GgufTensorType::MXFP4) {
                    bail!("DSpark expert tensors mix fp4 and f32 payloads");
                }
                bytes.extend_from_slice(file.bytes(&name)?);
            }
            other => bail!(
                "DSpark tensor {name} has unsupported dtype {}",
                other.name()
            ),
        }
    }
    Ok(DsparkExpertTensor {
        experts: RawExperts {
            name: dims.expert_name(layer, gguf_proj),
            in_dim,
            out_dim,
        },
        expert_count,
        layer,
        proj,
        dtype: dtype.ok_or_else(|| anyhow!("DSpark shards have zero experts"))?,
        bytes,
    })
}

// ---------------------------------------------------------------------------
// Host math shared by both providers
// ---------------------------------------------------------------------------

/// V4 rope on the trailing `rope_dims` of `values` — interleaved pairs, no
/// YARN — via [`v4_rope_sincos`], whose expressions are the exact ones the
/// trunk's `v4_rope_tail` evaluates (same `powf`/`sin_cos` libm calls), so
/// draft KV latents rotate bit-identically to trunk latents at equal
/// positions.
fn rope_tail(values: &mut [f32], rope_dims: usize, pos: usize, base: f32, inverse: bool) {
    if rope_dims == 0 {
        return;
    }
    let sincos = v4_rope_sincos(rope_dims, pos, base, inverse);
    let start = values.len() - rope_dims;
    let tail = &mut values[start..];
    for (pair, &(sin, cos)) in sincos.iter().enumerate() {
        let x0 = tail[2 * pair];
        let x1 = tail[2 * pair + 1];
        tail[2 * pair] = x0 * cos - x1 * sin;
        tail[2 * pair + 1] = x0 * sin + x1 * cos;
    }
}

/// Unweighted per-head RMS scaling on a contiguous multi-head row (the V4
/// query path applies it after `wq_b`, before rope — no learned weight).
fn per_head_rms(row: &mut [f32], head_dim: usize, rms_eps: f32) {
    for head in row.chunks_mut(head_dim) {
        let mean_square = head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let inv = (mean_square + rms_eps).sqrt().recip();
        for value in head.iter_mut() {
            *value *= inv;
        }
    }
}

/// The query block's MQA latent attention, pure host f32: every query row
/// attends to EVERY key (the caller passes the trailing-window context
/// latents plus the whole in-block key set — DSpark's non-causal block
/// semantics leave nothing to mask), with the layer's per-head attention
/// sinks adding softmax mass but no value. `keys` are shared K=V latents
/// (`head_dim` wide), exactly like the trunk's raw ring; per-head math
/// mirrors `dsv4_attention_step` operation for operation.
fn dspark_block_attention(
    q_rows: &[Vec<f32>],
    keys: &[&[f32]],
    sinks: Option<&[f32]>,
    heads: usize,
    head_dim: usize,
) -> Vec<Vec<f32>> {
    let scale = (head_dim as f32).powf(-0.5);
    let attend = |q_head: &[f32], sink: Option<f32>, out_head: &mut [f32]| {
        let mut weights: Vec<f32> = keys.iter().map(|key| dot(q_head, key) * scale).collect();
        // Stable softmax over [keys, sink]: the sink adds exp() mass to the
        // denominator but contributes no value.
        let mut max = sink.unwrap_or(f32::NEG_INFINITY);
        for weight in &weights {
            max = max.max(*weight);
        }
        let mut denom = sink.map(|sink| (sink - max).exp()).unwrap_or(0.0);
        for weight in &mut weights {
            *weight = (*weight - max).exp();
            denom += *weight;
        }
        for (weight, value) in weights.iter().zip(keys) {
            let weight = weight / denom;
            for (out, value) in out_head.iter_mut().zip(*value) {
                *out += weight * value;
            }
        }
    };
    let mut out = vec![vec![0.0f32; heads * head_dim]; q_rows.len()];
    // Each (row, head) task is independent and internally sequential, so the
    // parallel and serial paths produce bit-identical results.
    if q_rows.len() * heads * keys.len() * head_dim >= 1 << 18 {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(r, out_row)| {
            out_row
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(h, out_head)| {
                    attend(
                        &q_rows[r][h * head_dim..][..head_dim],
                        sinks.map(|sinks| sinks[h]),
                        out_head,
                    );
                });
        });
    } else {
        for (r, out_row) in out.iter_mut().enumerate() {
            for (h, out_head) in out_row.chunks_mut(head_dim).enumerate() {
                attend(
                    &q_rows[r][h * head_dim..][..head_dim],
                    sinks.map(|sinks| sinks[h]),
                    out_head,
                );
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Aux-hidden source (decouples the module from DsV4Taps for unit tests)
// ---------------------------------------------------------------------------

/// The module's view of captured target hiddens: one stream-averaged row
/// (embed wide) per (tapped layer, absolute position), for positions
/// `base() .. positions()`. `base()` is non-zero when the request resumed
/// from a prefix-cache restore — the restored prefix's activations were never
/// recomputed, so the context KV floors there.
pub(crate) trait DsparkAux {
    fn base(&self) -> usize;
    fn positions(&self) -> usize;
    fn aux_avg(&self, layer: usize, position: usize) -> Option<Vec<f32>>;
}

impl DsparkAux for DsV4Taps {
    fn base(&self) -> usize {
        DsV4Taps::base(self)
    }

    fn positions(&self) -> usize {
        DsV4Taps::positions(self)
    }

    fn aux_avg(&self, layer: usize, position: usize) -> Option<Vec<f32>> {
        self.aux_averaged(layer, position)
    }
}

// ---------------------------------------------------------------------------
// Mutable drafter state
// ---------------------------------------------------------------------------

/// The drafter's own mutable state: per-layer context-KV latents plus the
/// token of every inserted position (`inserted[i]` is the token at absolute
/// position `ctx_base + i`; tap position `p` is captured while forwarding
/// token `p`, so a matching token prefix implies matching taps under greedy
/// decode). Mismatch or a new request truncates back to the common prefix.
pub(crate) struct DsparkContext {
    /// Per draft layer: flat context latents, `head_dim` floats per position,
    /// row `i` holding ABSOLUTE position `ctx_base + i` (kv_norm'd + rope'd).
    ctx: Vec<Vec<f32>>,
    head_dim: usize,
    inserted: Vec<u32>,
    /// Absolute position of `inserted[0]` — 0 for full-history requests, the
    /// taps base after a prefix-cache restore floored the buildable context
    /// (positions below it stay uncovered: a quality-only approximation,
    /// invisible once the anchor moves a full SWA window past the base).
    ctx_base: usize,
    proposals: u64,
    rows_inserted: u64,
    truncations: u64,
    propose_nanos: u128,
    /// `accepted_hist[n]` counts verify steps whose accepted prefix was `n`
    /// drafts long (tail-capped). Slot `i >= 1` implies draft position `i`
    /// matched the target argmax. Fed by the native drafter's
    /// `observe_accepted`, reported at drop.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    accepted_hist: [u64; 6],
    /// Base (pre-Markov) logits of the last proposal, one row per query
    /// position — the Markov-chain exactness tests recompute the bias chain
    /// against these.
    #[cfg(test)]
    pub(crate) last_base_logits: Vec<Vec<f32>>,
}

impl DsparkContext {
    pub(crate) fn new(layers: usize, head_dim: usize) -> Self {
        Self {
            ctx: vec![Vec::new(); layers],
            head_dim,
            inserted: Vec::new(),
            ctx_base: 0,
            proposals: 0,
            rows_inserted: 0,
            truncations: 0,
            propose_nanos: 0,
            accepted_hist: [0; 6],
            #[cfg(test)]
            last_base_logits: Vec::new(),
        }
    }

    /// Drop every context row at or beyond `keep` (rejected drafts, request
    /// switches, prompt divergence).
    fn truncate(&mut self, keep: usize) {
        for latents in &mut self.ctx {
            latents.truncate(keep * self.head_dim);
        }
        self.inserted.truncate(keep);
        self.truncations += 1;
    }

    /// Inserted context length (tests).
    #[cfg(test)]
    pub(crate) fn inserted_len(&self) -> usize {
        self.inserted.len()
    }

    /// Absolute position of the first context row (tests).
    #[cfg(test)]
    pub(crate) fn ctx_base(&self) -> usize {
        self.ctx_base
    }
}

// ---------------------------------------------------------------------------
// The module (provider-agnostic forward)
// ---------------------------------------------------------------------------

/// The DSpark draft transformer: weights + the trunk scalars its forward
/// needs. The heavy linears go through whatever [`DsV4Linear`] provider
/// registered the payloads ([`DsparkHostLinear`] for the CPU reference, the
/// trunk's `DsV4GpuLinear` handle in production), passed per call so the
/// module itself stays provider-agnostic.
pub(crate) struct DsparkModule {
    dims: DsparkDims,
    weights: DsparkWeights,
    /// Target-shared embedding + lm head (the shards have neither).
    gguf: Arc<GgufFile>,
    token_embd: RawMatrix,
    output_head: RawMatrix,
    /// The noise token's embedding row, gathered once at construction (the
    /// anchor row is gathered per proposal).
    noise_embed: Vec<f32>,
}

impl DsparkModule {
    pub(crate) fn new(
        dims: DsparkDims,
        weights: DsparkWeights,
        gguf: Arc<GgufFile>,
        token_embd: RawMatrix,
        output_head: RawMatrix,
    ) -> Result<Self> {
        if weights.layers.len() != dims.config.layers {
            bail!(
                "DSpark weights carry {} layers; the config declares {}",
                weights.layers.len(),
                dims.config.layers
            );
        }
        let noise_embed = dsv4_embed_row(
            &gguf,
            &token_embd,
            dims.geometry.vocab,
            dims.config.noise_token,
        )?;
        Ok(Self {
            dims,
            weights,
            gguf,
            token_embd,
            output_head,
            noise_embed,
        })
    }

    pub(crate) fn context(&self) -> DsparkContext {
        DsparkContext::new(self.dims.config.layers, self.dims.geometry.head_dim)
    }

    /// The tap layers this module conditions on (the drafter's tap config).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn target_layers(&self) -> &[usize] {
        &self.dims.config.target_layers
    }

    fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        dsv4_embed_row(
            &self.gguf,
            &self.token_embd,
            self.dims.geometry.vocab,
            token,
        )
    }

    fn hc_params(&self) -> HcParams {
        HcParams {
            hc: self.dims.geometry.hc,
            embed: self.dims.geometry.embed,
            rms_eps: self.dims.rms_eps,
            hc_eps: self.dims.hc_eps,
            sinkhorn_iterations: self.dims.geometry.sinkhorn_iterations,
        }
    }

    fn moe_ctx<'a>(&'a self, layer: &'a DsV4Layer) -> DsV4MoeBlockCtx<'a> {
        let g = &self.dims.geometry;
        DsV4MoeBlockCtx {
            router: &layer.router,
            probs_bias: layer.probs_bias.as_deref(),
            tid2eid: None,
            gate: &layer.gate_exps,
            up: &layer.up_exps,
            down: &layer.down_exps,
            shared: layer.shared.as_ref().map(|shared| DsV4MoeShared {
                gate: &shared.gate,
                up: &shared.up,
                down: &shared.down,
            }),
            swiglu_clamp: layer.swiglu_clamp,
            experts: g.experts,
            top_k: g.moe_top_k,
            weights_norm: g.expert_weights_norm,
            weights_scale: g.expert_weights_scale,
            embed: g.embed,
        }
    }

    /// Condition on the target hiddens for positions `start..end`: per
    /// position, concat the stream-averaged taps in layer order, project
    /// through `main_proj`, THEN `main_norm` (vLLM `combine_hidden_states`:
    /// `main_norm(main_proj(concat))`), and append every layer's context-KV
    /// latent (`kv_norm(wkv(main_x))` rope'd at the ABSOLUTE position — vLLM
    /// `precompute_and_store_context_kv`).
    fn append_context<L: DsV4Linear + ?Sized>(
        &self,
        linear: &L,
        state: &mut DsparkContext,
        aux: &dyn DsparkAux,
        start: usize,
        end: usize,
    ) -> Result<()> {
        let g = &self.dims.geometry;
        let m = end - start;
        let mut rows = Vec::with_capacity(m);
        for pos in start..end {
            let mut row = Vec::with_capacity(self.dims.main_in());
            for &layer in &self.dims.config.target_layers {
                let avg = aux.aux_avg(layer, pos).ok_or_else(|| {
                    anyhow!("missing aux tap for target layer {layer} position {pos}")
                })?;
                if avg.len() != g.embed {
                    bail!(
                        "aux tap rows are {} wide but the DSpark main_proj expects {} per layer",
                        avg.len(),
                        g.embed
                    );
                }
                row.extend_from_slice(&avg);
            }
            rows.push(row);
        }
        let mut main_x = linear.mul_mat(TensorKey::Dense(&self.weights.main_proj), &rows)?;
        for row in &mut main_x {
            rms_norm_in_place(row, &self.weights.main_norm, self.dims.rms_eps)?;
        }
        for (l, layer) in self.weights.layers.iter().enumerate() {
            let kvs = linear.mul_mat(TensorKey::Dense(&layer.kv), &main_x)?;
            for (r, mut kv) in kvs.into_iter().enumerate() {
                rms_norm_in_place(&mut kv, &layer.kv_norm, self.dims.rms_eps)?;
                rope_tail(&mut kv, g.rope_dims, start + r, layer.rope_base, false);
                state.ctx[l].extend_from_slice(&kv);
            }
        }
        state.rows_inserted += m as u64;
        Ok(())
    }

    /// The query-block forward: `block_size` tokens `[anchor, noise, ...]` at
    /// absolute positions `pos0 ..`, non-causal within the block against the
    /// trailing-window context latents. Returns one PRE-norm hc_head-collapsed
    /// hidden per query position.
    fn run_block<L: DsV4Linear + ?Sized>(
        &self,
        linear: &L,
        state: &DsparkContext,
        anchor: u32,
        pos0: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let g = &self.dims.geometry;
        let rms_eps = self.dims.rms_eps;
        let n = self.dims.config.block_size;
        let head_dim = g.head_dim;

        // Query ids: the anchor then noise fill. Embeddings expand across the
        // hc streams (vLLM: `inputs_embeds.unsqueeze(-2).repeat(1, hc, 1)`).
        let mut ids = vec![self.dims.config.noise_token; n];
        ids[0] = anchor;
        let anchor_embed = self.embed_row(anchor)?;
        let mut streams: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
        for slot in 0..n {
            let embed = if slot == 0 {
                anchor_embed.clone()
            } else {
                self.noise_embed.clone()
            };
            streams.push(vec![embed; g.hc]);
        }

        // Context keys: the trailing `window` latents — the draft layers'
        // SWA-128 ring. Every query sees the same context slice plus the
        // whole in-block key set (non-causal).
        let ctx_len = state.inserted.len();
        let take = ctx_len.min(g.window.unwrap_or(ctx_len));
        let ctx_start = (ctx_len - take) * head_dim;

        for (l, layer) in self.weights.layers.iter().enumerate() {
            // --- attention sub-block ---
            let residual = streams.clone();
            let mut ys = Vec::with_capacity(n);
            let mut posts = Vec::with_capacity(n);
            let mut combs = Vec::with_capacity(n);
            for slot_streams in &streams {
                let (mut y, post, comb) =
                    hc_pre_math(&layer.hc_attn, slot_streams, self.hc_params())?;
                rms_norm_in_place(&mut y, &layer.attn_norm, rms_eps)?;
                ys.push(y);
                posts.push(post);
                combs.push(comb);
            }

            // Queries: wq_a -> q_norm -> wq_b -> per-head RMS -> rope at the
            // slot's absolute position.
            let mut qr = linear.mul_mat(TensorKey::Dense(&layer.q_a), &ys)?;
            for row in &mut qr {
                rms_norm_in_place(row, &layer.q_a_norm, rms_eps)?;
            }
            let mut q = linear.mul_mat(TensorKey::Dense(&layer.q_b), &qr)?;
            for (slot, row) in q.iter_mut().enumerate() {
                per_head_rms(row, head_dim, rms_eps);
                for head in row.chunks_mut(head_dim) {
                    rope_tail(head, g.rope_dims, pos0 + slot, layer.rope_base, false);
                }
            }

            // In-block KV latents from the block's own activations, rope'd at
            // their absolute positions; NEVER persisted into the context.
            let mut blk_kv = linear.mul_mat(TensorKey::Dense(&layer.kv), &ys)?;
            for (slot, kv) in blk_kv.iter_mut().enumerate() {
                rms_norm_in_place(kv, &layer.kv_norm, rms_eps)?;
                rope_tail(kv, g.rope_dims, pos0 + slot, layer.rope_base, false);
            }

            // K = V: trailing-window context latents then the block latents.
            let mut keys: Vec<&[f32]> = state.ctx[l][ctx_start..].chunks(head_dim).collect();
            keys.extend(blk_kv.iter().map(Vec::as_slice));
            let mut attn =
                dspark_block_attention(&q, &keys, layer.sinks.as_deref(), g.heads, head_dim);
            for (slot, row) in attn.iter_mut().enumerate() {
                // Inverse rope (negated angle) on each head's output tail at
                // the query position, exactly like the trunk step.
                for head in row.chunks_mut(head_dim) {
                    rope_tail(head, g.rope_dims, pos0 + slot, layer.rope_base, true);
                }
            }
            let projected = linear.mul_mat(
                TensorKey::Grouped {
                    matrix: &layer.out_a,
                    rank: g.o_rank,
                },
                &attn,
            )?;
            let o = linear.mul_mat(TensorKey::Dense(&layer.out_b), &projected)?;
            for (slot, o_row) in o.iter().enumerate() {
                streams[slot] = hc_post(o_row, &residual[slot], &posts[slot], &combs[slot]);
            }

            // --- MoE sub-block ---
            let residual = streams.clone();
            let mut ys = Vec::with_capacity(n);
            let mut posts = Vec::with_capacity(n);
            let mut combs = Vec::with_capacity(n);
            for slot_streams in &streams {
                let (mut y, post, comb) =
                    hc_pre_math(&layer.hc_ffn, slot_streams, self.hc_params())?;
                rms_norm_in_place(&mut y, &layer.ffn_norm, rms_eps)?;
                ys.push(y);
                posts.push(post);
                combs.push(comb);
            }
            let ffn = linear.moe_block(&self.moe_ctx(layer), &ys, &ids)?;
            if ffn.len() != n {
                bail!("DSpark moe_block returned {} rows for {n} slots", ffn.len());
            }
            for (slot, ffn_row) in ffn.iter().enumerate() {
                streams[slot] = hc_post(ffn_row, &residual[slot], &posts[slot], &combs[slot]);
            }
        }

        // hc_head collapse -> PRE-norm head hidden per position (vLLM's
        // forward returns exactly this; norm applies in compute_logits).
        streams
            .iter()
            .map(|slot_streams| {
                hyper_head_math(
                    &self.weights.hc_head,
                    slot_streams,
                    self.dims.geometry.embed,
                    rms_eps,
                    self.dims.hc_eps,
                )
            })
            .collect()
    }

    /// Base logits per query position: the TARGET's lm head over
    /// `mtp.2.norm`-normed hiddens (vLLM `compute_logits`).
    fn block_logits<L: DsV4Linear + ?Sized>(
        &self,
        linear: &L,
        mut hidden: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>> {
        for row in &mut hidden {
            rms_norm_in_place(row, &self.weights.norm, self.dims.rms_eps)?;
        }
        linear.mul_mat(TensorKey::Dense(&self.output_head), &hidden)
    }

    /// The vanilla Markov bias for one previous token:
    /// `markov_w2 @ markov_w1[prev]` (full vocab).
    pub(crate) fn markov_bias<L: DsV4Linear + ?Sized>(
        &self,
        linear: &L,
        prev: u32,
    ) -> Result<Vec<f32>> {
        let row = self.weights.markov_w1.row(prev)?;
        linear.mul_vec(TensorKey::Dense(&self.weights.markov_w2), &row)
    }

    /// The full proposal step: reconcile the context KV with the taps
    /// (truncate on divergence, floor at the taps base, append newly verified
    /// positions), run ONE query-block forward after the pending token, and
    /// chain the Markov-biased argmaxes left to right. Returns
    /// `min(block_size, k)` drafts; position i predicts absolute
    /// `pos0 + 1 + i` — directly continuing `tokens`.
    pub(crate) fn propose_tokens<L: DsV4Linear + ?Sized>(
        &self,
        linear: &L,
        state: &mut DsparkContext,
        tokens: &[u32],
        aux: &dyn DsparkAux,
        k: usize,
    ) -> Result<Vec<u32>> {
        let started = Instant::now();
        let g = &self.dims.geometry;
        let n_ctx = aux.positions();
        if tokens.len() != n_ctx + 1 {
            bail!(
                "DSpark drafter got {} context tokens but {n_ctx} tap positions (want tokens = taps + pending)",
                tokens.len()
            );
        }
        let k = k.min(self.dims.config.block_size);
        if k == 0 {
            return Ok(Vec::new());
        }
        let anchor = tokens[n_ctx];
        if anchor as usize >= g.vocab {
            bail!(
                "DSpark anchor token {anchor} outside the target vocab {}",
                g.vocab
            );
        }

        // Longest verified prefix both sides agree on, compared at the
        // context's own absolute offset. Within one request this is
        // everything (taps only ever extend by accepted tokens); across
        // requests it reuses a shared conversation prefix — rows below the
        // new request's taps base stay valid (their taps existed when they
        // were built) — and truncates the rest. Rows that would have to be
        // REBUILT below the base cannot be (no taps there), so the context
        // rebases: building floors at the taps base.
        let base = aux.base();
        let matched = if state.ctx_base <= n_ctx {
            state
                .inserted
                .iter()
                .zip(&tokens[state.ctx_base..n_ctx])
                .take_while(|(a, b)| a == b)
                .count()
        } else {
            0
        };
        if state.ctx_base > n_ctx || state.ctx_base + matched < base {
            // Nothing reusable at or above the base: restart the context
            // there (an already-empty context just moves — no truncation to
            // count).
            if !state.inserted.is_empty() {
                state.truncate(0);
            }
            state.ctx_base = base;
        } else if matched < state.inserted.len() {
            state.truncate(matched);
        }
        let keep = state.ctx_base + state.inserted.len();
        debug_assert!(keep >= base);
        if keep < n_ctx {
            self.append_context(linear, state, aux, keep, n_ctx)?;
            state.inserted.extend_from_slice(&tokens[keep..n_ctx]);
        }

        // One parallel block forward, then the sequential Markov chain.
        let hidden = self.run_block(linear, state, anchor, n_ctx)?;
        let mut logits = self.block_logits(linear, hidden)?;
        #[cfg(test)]
        {
            state.last_base_logits = logits.clone();
        }
        let mut drafts = Vec::with_capacity(k);
        let mut prev = anchor;
        for row in logits.iter_mut().take(k) {
            let bias = self.markov_bias(linear, prev)?;
            if bias.len() != row.len() {
                bail!(
                    "Markov bias width {} does not match the vocab logits {}",
                    bias.len(),
                    row.len()
                );
            }
            for (logit, bias) in row.iter_mut().zip(&bias) {
                *logit += *bias;
            }
            let draft = argmax(row)?;
            drafts.push(draft);
            prev = draft;
        }
        state.proposals += 1;
        state.propose_nanos += started.elapsed().as_nanos();
        Ok(drafts)
    }
}

// ---------------------------------------------------------------------------
// CPU host reference provider
// ---------------------------------------------------------------------------

/// The CPU host reference [`DsV4Linear`] for the DSpark module: the shards'
/// dense payloads and packed expert blobs served with exact f32 host math,
/// and everything else (the target-shared embedding + lm head) falling
/// through to the plain GGUF-streaming CPU provider. The CPU==GPU parity gate
/// drives the SAME [`DsparkModule`] through this and the CUDA provider.
/// Test-only today.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct DsparkHostLinear {
    fallback: crate::dsv4_cpu::DsV4CpuLinear,
    mats: HashMap<String, DsparkHostMat>,
    experts: HashMap<String, DsparkHostExperts>,
}

struct DsparkHostMat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

struct DsparkHostExperts {
    dtype: GgufTensorType,
    bytes: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DsparkHostLinear {
    pub(crate) fn new(gguf: Arc<GgufFile>, load: &DsparkLoad) -> Self {
        let mats = load
            .dense
            .iter()
            .map(|entry| {
                (
                    entry.matrix.name.clone(),
                    DsparkHostMat {
                        rows: entry.matrix.rows,
                        cols: entry.matrix.cols,
                        data: entry.payload.to_f32(),
                    },
                )
            })
            .collect();
        let experts = load
            .experts
            .iter()
            .map(|entry| {
                (
                    entry.experts.name.clone(),
                    DsparkHostExperts {
                        dtype: entry.dtype,
                        bytes: entry.bytes.clone(),
                    },
                )
            })
            .collect();
        Self {
            fallback: crate::dsv4_cpu::DsV4CpuLinear::from_gguf(gguf),
            mats,
            experts,
        }
    }
}

impl DsV4Linear for DsparkHostLinear {
    fn mul_vec(&self, key: TensorKey<'_>, x: &[f32]) -> Result<Vec<f32>> {
        match key {
            TensorKey::Dense(matrix) => {
                let Some(mat) = self.mats.get(&matrix.name) else {
                    return self.fallback.mul_vec(key, x);
                };
                if x.len() != mat.cols || matrix.rows != mat.rows {
                    bail!(
                        "matvec shapes do not match host DSpark tensor {} [{}, {}]",
                        matrix.name,
                        mat.rows,
                        mat.cols
                    );
                }
                Ok(mat
                    .data
                    .chunks_exact(mat.cols)
                    .map(|row| dot(row, x))
                    .collect())
            }
            TensorKey::Grouped { matrix, rank } => {
                let Some(mat) = self.mats.get(&matrix.name) else {
                    return self.fallback.mul_vec(key, x);
                };
                // Mirror the CPU provider's block-diagonal loop exactly.
                let group_features = mat.cols;
                if rank == 0 || !mat.rows.is_multiple_of(rank) {
                    bail!(
                        "grouped matvec tensor {} does not fit rank {rank}",
                        matrix.name
                    );
                }
                let groups = mat.rows / rank;
                if x.len() != groups * group_features {
                    bail!(
                        "grouped matvec input length {} does not match {groups} groups of {group_features}",
                        x.len()
                    );
                }
                Ok((0..mat.rows)
                    .map(|row| {
                        let group = row / rank;
                        let x_group = &x[group * group_features..(group + 1) * group_features];
                        dot(&mat.data[row * mat.cols..(row + 1) * mat.cols], x_group)
                    })
                    .collect())
            }
            TensorKey::Expert { experts, expert } => {
                let Some(blob) = self.experts.get(&experts.name) else {
                    return self.fallback.mul_vec(key, x);
                };
                if x.len() != experts.in_dim {
                    bail!(
                        "expert matvec input length {} does not match tensor {} input dim {}",
                        x.len(),
                        experts.name,
                        experts.in_dim
                    );
                }
                let per_expert = experts.in_dim * experts.out_dim;
                let start = usize::try_from(blob.dtype.byte_len((expert * per_expert) as u64)?)
                    .context("expert byte offset does not fit usize")?;
                let len = usize::try_from(blob.dtype.byte_len(per_expert as u64)?)
                    .context("expert byte length does not fit usize")?;
                let bytes = blob
                    .bytes
                    .get(start..start + len)
                    .ok_or_else(|| anyhow!("expert blob {} slice out of range", experts.name))?;
                let data = hi_gguf::dequantize_tensor_as_f32(bytes, blob.dtype, per_expert)?;
                Ok((0..experts.out_dim)
                    .map(|row| dot(&data[row * experts.in_dim..(row + 1) * experts.in_dim], x))
                    .collect())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Production drafter (native-cuda): shares the trunk engine's provider handle.
// ---------------------------------------------------------------------------

/// `HI_DSV4_DSPARK_PATH` (a DIRECTORY), defaulting to the documented local
/// checkpoint location.
#[cfg(feature = "native-cuda")]
fn dspark_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HI_DSV4_DSPARK_PATH") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".hi/models/deepseek-v4-flash/dspark"))
}

/// The `HI_DSV4_SPEC=dspark` drafter: the module + the trunk provider handle
/// + its own incremental context state.
#[cfg(feature = "native-cuda")]
pub(crate) struct DsparkDrafter {
    module: DsparkModule,
    linear: DsV4GpuLinear,
    state: DsparkContext,
    error_logged: bool,
}

#[cfg(feature = "native-cuda")]
impl DsparkDrafter {
    /// Build the drafter over an already-loaded trunk engine: resolve the
    /// config (`config.json` overrides the baked real defaults), derive dims,
    /// load + validate the shard trio, take GPU residency of the dense
    /// payloads, and register the packed experts in the trunk's pool as
    /// pinned draft layers. Construction runs on the engine worker thread.
    pub(crate) fn from_dir(engine: &DeepSeekV4GpuEngine, dir: &Path) -> Result<Self> {
        let inner = engine.engine();
        let mut config = DsparkConfig::real_default();
        let config_json = dir.join("config.json");
        if config_json.exists() {
            config
                .apply_config_json(&config_json)
                .with_context(|| format!("applying {}", config_json.display()))?;
        }
        let dims = DsparkDims::from_engine(inner, config)?;
        let shards = DsparkShards::open_dir(dir)?;
        let mut load = load_dspark(&shards, &dims)
            .with_context(|| format!("loading DSpark shards from {}", dir.display()))?;
        let resident_bytes = load.resident_bytes();
        let expert_bytes = load.expert_bytes();
        let linear = inner.linear().clone();

        for entry in load.dense.drain(..) {
            let payload = match entry.payload {
                DsparkPayload::F32(values) => HostDenseData::F32(values),
                DsparkPayload::F16(bits) => HostDenseData::F16(bits),
                DsparkPayload::Bf16(bits) => HostDenseData::Bf16(bits),
            };
            match entry.grouped_rank {
                Some(rank) => linear.register_host_grouped(&entry.matrix, rank, &payload)?,
                None => linear.register_host_dense(&entry.matrix, &payload)?,
            }
        }
        let (pinned, pooled) = register_pool_experts(&linear, &dims, &mut load)?;
        eprintln!(
            "dsv4 dspark drafter: loaded {} — {} draft layers, block {} (noise token {}), \
             taps {:?}, markov rank {}; {:.0} MiB resident dense, {:.2} GiB experts \
             ({pinned}/{pooled} expert slices pool-pinned as layers {:?})",
            dir.display(),
            dims.config.layers,
            dims.config.block_size,
            dims.config.noise_token,
            dims.config.target_layers,
            dims.config.markov_rank,
            resident_bytes as f64 / (1u64 << 20) as f64,
            expert_bytes as f64 / (1u64 << 30) as f64,
            (0..dims.config.layers)
                .map(|i| dims.pool_layer(i))
                .collect::<Vec<_>>(),
        );

        let module = DsparkModule::new(
            dims,
            load.weights,
            inner.gguf().clone(),
            inner.token_embd_matrix().clone(),
            inner.output_head_matrix().clone(),
        )?;
        let state = module.context();
        Ok(Self {
            module,
            linear,
            state,
            error_logged: false,
        })
    }

    fn log_error_once(&mut self, message: &str) {
        if !self.error_logged {
            self.error_logged = true;
            eprintln!("DSpark drafter disabled itself for this run: {message}");
        }
    }
}

/// POOL INTEGRATION POINT: register every draft layer's packed experts under
/// its own pool layer index (`trunk_layers + 1 + i` = 44/45/46 on the real
/// model) with pinned residency — every proposal touches all three layers'
/// routed experts, so LRU churn against the trunk's working set would thrash.
/// Blob ownership moves into the provider (no 10-GiB transient clones).
#[cfg(feature = "native-cuda")]
fn register_pool_experts(
    linear: &DsV4GpuLinear,
    dims: &DsparkDims,
    load: &mut DsparkLoad,
) -> Result<(usize, usize)> {
    let mut pinned = 0usize;
    let mut pooled = 0usize;
    for entry in load.experts.drain(..) {
        let layer = u32::try_from(dims.pool_layer(entry.layer))
            .context("draft pool layer index does not fit u32")?;
        let pin = linear.register_host_experts(
            &entry.experts,
            entry.expert_count,
            layer,
            entry.proj,
            entry.dtype,
            entry.bytes,
            true,
        )?;
        pooled += entry.expert_count;
        if pin {
            pinned += entry.expert_count;
        }
    }
    Ok((pinned, pooled))
}

#[cfg(feature = "native-cuda")]
impl Drafter for DsparkDrafter {
    fn tap_config(&self) -> DsV4TapConfig {
        DsV4TapConfig {
            pre_hc_head: false,
            aux_layers: self.module.target_layers().to_vec(),
        }
    }

    fn propose(&mut self, ctx: &DraftContext<'_>) -> Vec<u32> {
        let Some(taps) = ctx.taps else {
            self.log_error_once("the verify loop supplied no hidden taps");
            return Vec::new();
        };
        match self
            .module
            .propose_tokens(&self.linear, &mut self.state, ctx.tokens, taps, ctx.k)
        {
            Ok(drafts) => drafts,
            Err(err) => {
                self.log_error_once(&format!("{err:#}"));
                Vec::new()
            }
        }
    }

    fn observe_accepted(&mut self, proposed: usize, accepted: usize, emitted: u32) {
        let _ = emitted;
        if proposed > 0 {
            let slot = accepted.min(self.state.accepted_hist.len() - 1);
            self.state.accepted_hist[slot] += 1;
        }
    }
}

#[cfg(feature = "native-cuda")]
impl Drop for DsparkDrafter {
    fn drop(&mut self) {
        let state = &self.state;
        if state.proposals > 0 {
            let steps: u64 = state.accepted_hist.iter().sum();
            let weighted: u64 = state
                .accepted_hist
                .iter()
                .enumerate()
                .map(|(n, &count)| n as u64 * count)
                .sum();
            eprintln!(
                "DSpark drafter: {} proposals, {:.2} ms/propose avg, {} context rows inserted, \
                 {} truncations, accepted-prefix histogram {:?} (mean accepted {:.2}/block)",
                state.proposals,
                state.propose_nanos as f64 / state.proposals as f64 / 1e6,
                state.rows_inserted,
                state.truncations,
                state.accepted_hist,
                weighted as f64 / steps.max(1) as f64,
            );
        }
    }
}

/// `HI_DSV4_SPEC=dspark` entry point, constructed on the engine worker thread
/// (device resources are allowed here). Returning `None` leaves speculative
/// decoding off; every failure path explains itself on stderr.
#[cfg(feature = "native-cuda")]
pub(crate) fn dspark_drafter_from_env(engine: &DeepSeekV4GpuEngine) -> Option<Box<dyn Drafter>> {
    let Some(dir) = dspark_dir() else {
        eprintln!("HI_DSV4_SPEC=dspark: HOME is unset and HI_DSV4_DSPARK_PATH not given; spec off");
        return None;
    };
    if !dir.is_dir() {
        eprintln!(
            "HI_DSV4_SPEC=dspark: checkpoint directory {} not found (set HI_DSV4_DSPARK_PATH); \
             speculative decoding stays off",
            dir.display()
        );
        return None;
    }
    match DsparkDrafter::from_dir(engine, &dir) {
        Ok(drafter) => Some(Box::new(drafter)),
        Err(err) => {
            eprintln!(
                "HI_DSV4_SPEC=dspark: failed to load {}: {err:#}; speculative decoding stays off",
                dir.display()
            );
            None
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::dsv4_cpu::fixture::{tempfile_path, write_deepseek4_spec_gguf};
    use crate::dsv4_cpu::{DsV4CpuLinear, DsV4State};

    // -----------------------------------------------------------------------
    // Fixture plumbing: a shard TRIO + config.json in a temp directory,
    // mirroring the real checkpoint layout (mtp.{i} in its own file).
    // -----------------------------------------------------------------------

    pub(super) fn dspark_tempdir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-cuda-dsv4-dspark-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    /// Deterministic pseudo-random weights in roughly [-0.35, 0.35] (the
    /// engine fixture's generator, fresh seeds).
    fn vals(seed: u32, count: usize) -> Vec<f32> {
        (0..count)
            .map(|idx| {
                let mixed = seed
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(idx as u32)
                    .wrapping_mul(0x85eb_ca6b);
                (((mixed >> 8) % 2001) as f32 / 1000.0 - 1.0) * 0.35
            })
            .collect()
    }

    /// Write a syntactically valid F32 safetensors file (the writer pattern
    /// from `safetensors.rs` tests).
    fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut entries = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            assert_eq!(
                shape.iter().product::<usize>(),
                values.len(),
                "fixture tensor {name} value count does not match shape"
            );
            let begin = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            entries.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [begin, data.len()],
                }),
            );
        }
        let header = serde_json::Value::Object(entries).to_string().into_bytes();
        let mut out = Vec::with_capacity(8 + header.len() + data.len());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    /// Fixture DSpark config matching the spec-GGUF fixture trunk (3 layers,
    /// vocab 4): 3 draft layers, block 3, noise token 1, taps [0,1,2],
    /// markov rank 3.
    pub(super) fn fixture_config() -> DsparkConfig {
        DsparkConfig {
            layers: 3,
            block_size: 3,
            noise_token: 1,
            target_layers: vec![0, 1, 2],
            markov_rank: 3,
        }
    }

    /// Crafting knobs for the DSpark fixture trio (dims mirror
    /// `write_deepseek4_spec_gguf`: embed 4, hc 2, heads 2, head_dim 8,
    /// rope 4, q_lora 4, groups 2x4, experts 4 top-2 ff 4 + shared 4,
    /// window 4, vocab 4).
    pub(super) struct Craft {
        /// wo_b = 0 in every layer: attention contributes EXACTLY nothing.
        pub(super) zero_attn_out: bool,
        /// down experts + shared w2 = 0: the MoE contributes EXACTLY nothing.
        pub(super) zero_moe_down: bool,
        /// markov_w2 = 0: the Markov bias contributes EXACTLY nothing.
        pub(super) zero_markov: bool,
        /// Explicit `mtp.0.main_proj.weight` values ([4, 12]).
        pub(super) main_proj: Option<Vec<f32>>,
        /// Explicit `mtp.0.attn.wkv.weight` values ([8, 4]; layer 0 only).
        pub(super) wkv0: Option<Vec<f32>>,
        /// Explicit markov tables ([4, 3] each).
        pub(super) markov_w1: Option<Vec<f32>>,
        pub(super) markov_w2: Option<Vec<f32>>,
    }

    impl Craft {
        pub(super) fn random() -> Self {
            Self {
                zero_attn_out: false,
                zero_moe_down: false,
                zero_markov: false,
                main_proj: None,
                wkv0: None,
                markov_w1: None,
                markov_w2: None,
            }
        }
    }

    /// Write the fixture trio + config.json into `dir`.
    pub(super) fn write_dspark_fixture(dir: &Path, craft: &Craft, config: &DsparkConfig) {
        let embed = 4usize;
        let hc = 2usize;
        let q_lora = 4usize;
        let q_dim = 16usize; // 2 heads x 8
        let head_dim = 8usize;
        let groups = 2usize;
        let rank = 4usize;
        let experts = 4usize;
        let ff = 4usize;
        let shared_ff = 4usize;
        let mix = hc * hc + 2 * hc;
        let main_in = embed * config.target_layers.len();

        for i in 0..config.layers {
            let s = |k: u32| 500 + i as u32 * 100 + k;
            let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
                (
                    names::layer(i, "attn.wq_a.weight"),
                    vec![q_lora, embed],
                    vals(s(0), q_lora * embed),
                ),
                (
                    names::layer(i, "attn.q_norm.weight"),
                    vec![q_lora],
                    vec![1.0; q_lora],
                ),
                (
                    names::layer(i, "attn.wq_b.weight"),
                    vec![q_dim, q_lora],
                    vals(s(1), q_dim * q_lora),
                ),
                (
                    names::layer(i, "attn.wkv.weight"),
                    vec![head_dim, embed],
                    match (&craft.wkv0, i) {
                        (Some(values), 0) => values.clone(),
                        _ => vals(s(2), head_dim * embed),
                    },
                ),
                (
                    names::layer(i, "attn.kv_norm.weight"),
                    vec![head_dim],
                    vec![1.0; head_dim],
                ),
                (names::layer(i, "attn.attn_sink"), vec![2], vals(s(3), 2)),
                (
                    names::layer(i, "attn.wo_a.weight"),
                    vec![groups * rank, q_dim / groups],
                    vals(s(4), groups * rank * (q_dim / groups)),
                ),
                (
                    names::layer(i, "attn.wo_b.weight"),
                    vec![embed, groups * rank],
                    if craft.zero_attn_out {
                        vec![0.0; embed * groups * rank]
                    } else {
                        vals(s(5), embed * groups * rank)
                    },
                ),
                (
                    names::layer(i, "attn_norm.weight"),
                    vec![embed],
                    vec![1.0; embed],
                ),
                (
                    names::layer(i, "ffn_norm.weight"),
                    vec![embed],
                    vec![1.0; embed],
                ),
                (
                    names::hc(i, "attn", "fn"),
                    vec![mix, hc * embed],
                    vals(s(6), mix * hc * embed),
                ),
                (names::hc(i, "attn", "base"), vec![mix], vals(s(7), mix)),
                (names::hc(i, "attn", "scale"), vec![3], vec![0.6, 0.4, 0.8]),
                (
                    names::hc(i, "ffn", "fn"),
                    vec![mix, hc * embed],
                    vals(s(8), mix * hc * embed),
                ),
                (names::hc(i, "ffn", "base"), vec![mix], vals(s(9), mix)),
                (names::hc(i, "ffn", "scale"), vec![3], vec![0.5, 0.7, 0.3]),
                (
                    names::layer(i, "ffn.gate.weight"),
                    vec![experts, embed],
                    vals(s(10), experts * embed),
                ),
                (
                    names::layer(i, "ffn.gate.bias"),
                    vec![experts],
                    vals(s(11), experts),
                ),
                (
                    names::layer(i, "ffn.shared_experts.w1.weight"),
                    vec![shared_ff, embed],
                    vals(s(12), shared_ff * embed),
                ),
                (
                    names::layer(i, "ffn.shared_experts.w2.weight"),
                    vec![embed, shared_ff],
                    if craft.zero_moe_down {
                        vec![0.0; embed * shared_ff]
                    } else {
                        vals(s(13), embed * shared_ff)
                    },
                ),
                (
                    names::layer(i, "ffn.shared_experts.w3.weight"),
                    vec![shared_ff, embed],
                    vals(s(14), shared_ff * embed),
                ),
            ];
            for index in 0..experts {
                let e = s(20 + index as u32 * 3);
                tensors.push((
                    names::expert(i, index, "w1"),
                    vec![ff, embed],
                    vals(e, ff * embed),
                ));
                tensors.push((
                    names::expert(i, index, "w3"),
                    vec![ff, embed],
                    vals(e + 1, ff * embed),
                ));
                tensors.push((
                    names::expert(i, index, "w2"),
                    vec![embed, ff],
                    if craft.zero_moe_down {
                        vec![0.0; embed * ff]
                    } else {
                        vals(e + 2, embed * ff)
                    },
                ));
            }
            if i == 0 {
                tensors.push((
                    names::MAIN_PROJ.to_string(),
                    vec![embed, main_in],
                    match &craft.main_proj {
                        Some(values) => values.clone(),
                        None => vals(900, embed * main_in),
                    },
                ));
                tensors.push((names::MAIN_NORM.to_string(), vec![embed], vec![1.0; embed]));
            }
            if i == config.layers - 1 {
                tensors.push((names::head(i, "norm.weight"), vec![embed], vec![1.0; embed]));
                tensors.push((
                    names::hc(i, "head", "fn"),
                    vec![hc, hc * embed],
                    vals(910, hc * hc * embed),
                ));
                tensors.push((names::hc(i, "head", "base"), vec![hc], vals(911, hc)));
                tensors.push((names::hc(i, "head", "scale"), vec![1], vec![0.7]));
                tensors.push((
                    names::head(i, "markov_head.markov_w1.weight"),
                    vec![4, config.markov_rank],
                    match &craft.markov_w1 {
                        Some(values) => values.clone(),
                        None => vals(912, 4 * config.markov_rank),
                    },
                ));
                tensors.push((
                    names::head(i, "markov_head.markov_w2.weight"),
                    vec![4, config.markov_rank],
                    match (&craft.markov_w2, craft.zero_markov) {
                        (_, true) => vec![0.0; 4 * config.markov_rank],
                        (Some(values), _) => values.clone(),
                        (None, _) => vals(913, 4 * config.markov_rank),
                    },
                ));
                // Present in the real shards, deliberately never loaded.
                tensors.push((
                    names::head(i, "confidence_head.proj.weight"),
                    vec![1, embed + config.markov_rank],
                    vals(914, embed + config.markov_rank),
                ));
            }
            write_safetensors(
                &dir.join(format!("dspark-{:05}.safetensors", i + 1)),
                &tensors,
            );
        }
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "n_mtp_layers": config.layers,
                "dspark_block_size": config.block_size,
                "dspark_noise_token_id": config.noise_token,
                "dspark_target_layer_ids": config.target_layers,
                "dspark_markov_rank": config.markov_rank,
            })
            .to_string(),
        )
        .unwrap();
    }

    /// A CPU trunk engine + DSpark module + host provider over fresh fixtures.
    pub(super) struct CpuRig {
        pub(super) engine: DsV4Engine<DsV4CpuLinear>,
        pub(super) module: DsparkModule,
        pub(super) host: DsparkHostLinear,
        #[allow(dead_code)]
        pub(super) dir: PathBuf,
    }

    pub(super) fn cpu_rig(name: &str, craft: &Craft) -> CpuRig {
        cpu_rig_with(name, craft, &fixture_config())
    }

    pub(super) fn cpu_rig_with(name: &str, craft: &Craft, config: &DsparkConfig) -> CpuRig {
        let gguf_path = tempfile_path(&format!("dspark-{name}"));
        write_deepseek4_spec_gguf(&gguf_path);
        let dir = dspark_tempdir(name);
        write_dspark_fixture(&dir, craft, config);
        let gguf = Arc::new(hi_gguf::GgufFile::open(&gguf_path).unwrap());
        let engine = DsV4Engine::new(
            gguf.clone(),
            DsV4CpuLinear::from_gguf(gguf.clone()),
            "cpu-reference",
        )
        .unwrap();
        // The production config path: baked defaults overridden by the
        // directory's config.json.
        let mut resolved = DsparkConfig::real_default();
        resolved
            .apply_config_json(&dir.join("config.json"))
            .unwrap();
        let dims = DsparkDims::from_engine(&engine, resolved).unwrap();
        let shards = DsparkShards::open_dir(&dir).unwrap();
        let load = load_dspark(&shards, &dims).unwrap();
        let host = DsparkHostLinear::new(gguf.clone(), &load);
        let module = DsparkModule::new(
            dims,
            load.weights,
            gguf,
            engine.token_embd_matrix().clone(),
            engine.output_head_matrix().clone(),
        )
        .unwrap();
        CpuRig {
            engine,
            module,
            host,
            dir,
        }
    }

    /// Prefill `context` through the trunk with aux capture at the fixture's
    /// tap layers.
    pub(super) fn taps_for(engine: &DsV4Engine<DsV4CpuLinear>, context: &[u32]) -> DsV4Taps {
        taps_for_at(engine, context, 0)
    }

    /// [`taps_for`] with the first `base` positions forwarded UNTAPPED — a
    /// prefix-cache restore's view of the same sequence.
    pub(super) fn taps_for_at(
        engine: &DsV4Engine<DsV4CpuLinear>,
        context: &[u32],
        base: usize,
    ) -> DsV4Taps {
        let mut taps = engine
            .new_taps_at(
                crate::dsv4_cpu::DsV4TapConfig {
                    pre_hc_head: false,
                    aux_layers: fixture_config().target_layers,
                },
                base,
            )
            .unwrap();
        let mut state: DsV4State = engine.new_state();
        if base > 0 {
            engine.prefill(&mut state, &context[..base]).unwrap();
        }
        engine
            .prefill_with_taps(&mut state, &context[base..], Some(&mut taps))
            .unwrap();
        taps
    }

    /// Test double for the engine taps: one stream-averaged row per
    /// (position, layer), rows starting at absolute `base` like a
    /// restore-attached [`DsV4Taps`].
    pub(super) struct OwnedAux {
        layers: Vec<usize>,
        base: usize,
        /// `rows[position - base][layer group]` — embed-wide averaged rows.
        rows: Vec<Vec<Vec<f32>>>,
    }

    impl DsparkAux for OwnedAux {
        fn base(&self) -> usize {
            self.base
        }

        fn positions(&self) -> usize {
            self.base + self.rows.len()
        }

        fn aux_avg(&self, layer: usize, position: usize) -> Option<Vec<f32>> {
            let group = self.layers.iter().position(|&l| l == layer)?;
            self.rows
                .get(position.checked_sub(self.base)?)
                .map(|row| row[group].clone())
        }
    }

    fn fnv(s: &str) -> u32 {
        s.bytes().fold(0x811c_9dc5u32, |h, b| {
            (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
        })
    }

    /// Rows keyed off the token at each position, so "same tokens => same
    /// taps" holds across reconstructions like it does for the real
    /// (deterministic, greedy) engine.
    pub(super) fn owned_aux(layers: &[usize], width: usize, context: &[u32]) -> OwnedAux {
        owned_aux_at(layers, width, context, 0)
    }

    pub(super) fn owned_aux_at(
        layers: &[usize],
        width: usize,
        context: &[u32],
        base: usize,
    ) -> OwnedAux {
        let rows = context[base..]
            .iter()
            .map(|&token| {
                layers
                    .iter()
                    .map(|&layer| vals(fnv(&format!("aux-{layer}-{token}")), width))
                    .collect()
            })
            .collect();
        OwnedAux {
            layers: layers.to_vec(),
            base,
            rows,
        }
    }

    pub(super) fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "{what}[{i}]: got {a}, expected {e} (tol {tol})"
            );
        }
    }

    // Independent reference math for the hand-computed tests (explicit
    // formulas, NOT the module's helpers).

    fn ref_rms(values: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        values
            .iter()
            .zip(weight)
            .map(|(v, w)| v * inv * w)
            .collect()
    }

    /// V4 interleaved rope on the trailing `rope_dims` values: pair (2i,
    /// 2i+1) rotates by angle `pos / base^(2i/rope_dims)`.
    fn ref_rope_tail(values: &[f32], rope_dims: usize, pos: usize, base: f32) -> Vec<f32> {
        let mut out = values.to_vec();
        let start = values.len() - rope_dims;
        for pair in 0..rope_dims / 2 {
            let freq = 1.0 / base.powf((2 * pair) as f32 / rope_dims as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = out[start + 2 * pair];
            let x1 = out[start + 2 * pair + 1];
            out[start + 2 * pair] = x0 * cos - x1 * sin;
            out[start + 2 * pair + 1] = x0 * sin + x1 * cos;
        }
        out
    }

    // -----------------------------------------------------------------------
    // Loader + config validation
    // -----------------------------------------------------------------------

    #[test]
    fn dspark_loader_validates_shapes_and_remap() {
        let rig = cpu_rig("loader", &Craft::random());
        let dims = &rig.module.dims;
        assert_eq!(dims.trunk_layers, 3);
        assert_eq!(dims.expert_ff, 4);
        assert_eq!(dims.shared_ff, 4);
        assert_eq!(dims.geometry.hc, 2);
        assert_eq!(dims.rope_base, 10_000.0);
        assert_eq!(dims.swiglu_clamp, 10.0);
        assert_eq!(dims.config.layers, 3);
        assert_eq!(dims.config.block_size, 3);
        assert_eq!(dims.config.noise_token, 1);
        assert_eq!(dims.config.target_layers, vec![0, 1, 2]);
        assert_eq!(dims.config.markov_rank, 3);
        // Draft pool layers sit ABOVE the MTP slot (trunk_layers + 1 + i).
        assert_eq!(dims.pool_layer(0), 4);
        assert_eq!(dims.pool_layer(2), 6);
        let weights = &rig.module.weights;
        assert_eq!(weights.layers.len(), 3);
        for (i, layer) in weights.layers.iter().enumerate() {
            assert!(layer.compressor.is_none());
            assert!(layer.indexer.is_none());
            assert!(layer.tid2eid.is_none());
            assert!(layer.probs_bias.is_some());
            assert!(layer.sinks.is_some());
            assert_eq!(
                layer.gate_exps.name,
                format!("blk.{}.ffn_gate_exps.weight", 4 + i)
            );
        }
        assert_eq!(weights.main_proj.rows, 4);
        assert_eq!(weights.main_proj.cols, 12);
        assert_eq!(weights.markov_w2.rows, 4);
        assert_eq!(weights.markov_w2.cols, 3);

        // The confidence head ships in the shards and is deliberately never
        // loaded (vLLM drops it; a future dynamic-length lever).
        let shards = DsparkShards::open_dir(&rig.dir).unwrap();
        assert!(
            shards.info("mtp.2.confidence_head.proj.weight").is_some(),
            "fixture must carry the confidence head for the load-skip coverage"
        );

        // A wrong-shaped tensor must fail loudly, naming the tensor.
        let bad_dir = dspark_tempdir("loader-bad");
        write_dspark_fixture(&bad_dir, &Craft::random(), &fixture_config());
        let shard1 = bad_dir.join("dspark-00001.safetensors");
        let mut file_bytes = std::fs::read(&shard1).unwrap();
        let header_len = u64::from_le_bytes(file_bytes[0..8].try_into().unwrap()) as usize;
        let header = String::from_utf8(file_bytes[8..8 + header_len].to_vec()).unwrap();
        // Rewrite the wq_a shape [4,4] -> [2,8] (same byte span). The header
        // stores tensors in insertion order, so wq_a's shape is the first
        // "[4,4]" after its name.
        let marker = "mtp.0.attn.wq_a.weight";
        let at = header.find(marker).expect("wq_a in header");
        let shape_at = at + header[at..].find("\"shape\":[4,4]").expect("wq_a shape");
        let mut bad_header = header.clone();
        bad_header.replace_range(
            shape_at.."\"shape\":[4,4]".len() + shape_at,
            "\"shape\":[2,8]",
        );
        assert_eq!(header.len(), bad_header.len());
        file_bytes[8..8 + header_len].copy_from_slice(bad_header.as_bytes());
        std::fs::write(&shard1, file_bytes).unwrap();
        let shards = DsparkShards::open_dir(&bad_dir).unwrap();
        let err = match load_dspark(&shards, &rig.module.dims) {
            Ok(_) => panic!("loader accepted a mis-shaped wq_a"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("wq_a") && err.contains("expected [4, 4]"),
            "unhelpful shape error: {err}"
        );

        // A missing tensor names itself too.
        let sparse_dir = dspark_tempdir("loader-missing");
        let mut two_layer = fixture_config();
        two_layer.layers = 2;
        write_dspark_fixture(&sparse_dir, &Craft::random(), &two_layer);
        let shards = DsparkShards::open_dir(&sparse_dir).unwrap();
        let err = load_dspark(&shards, &rig.module.dims)
            .err()
            .expect("missing mtp.2 tensors must fail")
            .to_string();
        assert!(err.contains("mtp.2."), "got: {err}");
    }

    #[test]
    fn dspark_config_json_overrides_and_validates() {
        let dir = dspark_tempdir("config");
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "dspark_block_size": 7,
                "dspark_noise_token_id": 42,
                "dspark_target_layer_ids": [1, 2],
                "dspark_markov_rank": 8,
                "n_mtp_layers": 2,
            })
            .to_string(),
        )
        .unwrap();
        let mut config = DsparkConfig::real_default();
        config.apply_config_json(&dir.join("config.json")).unwrap();
        assert_eq!(config.layers, 2);
        assert_eq!(config.block_size, 7);
        assert_eq!(config.noise_token, 42);
        assert_eq!(config.target_layers, vec![1, 2]);
        assert_eq!(config.markov_rank, 8);

        // The unsupported 1+N layout is rejected loudly.
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({"dspark_bonus_anchor": true}).to_string(),
        )
        .unwrap();
        let err = config
            .apply_config_json(&dir.join("config.json"))
            .expect_err("bonus-anchor checkpoints are unsupported")
            .to_string();
        assert!(err.contains("bonus_anchor"), "got: {err}");

        // Validation against the trunk.
        let bad = |mutate: &dyn Fn(&mut DsparkConfig)| {
            let mut config = fixture_config();
            mutate(&mut config);
            config.validate(3, 4).expect_err("must fail validation")
        };
        assert!(
            bad(&|c| c.target_layers = vec![0, 2, 1])
                .to_string()
                .contains("ascending")
        );
        assert!(
            bad(&|c| c.target_layers = vec![0, 3])
                .to_string()
                .contains("outside the trunk")
        );
        assert!(
            bad(&|c| c.noise_token = 4)
                .to_string()
                .contains("outside the target vocab")
        );
        assert!(bad(&|c| c.block_size = 0).to_string().contains("zero"));
        assert!(fixture_config().validate(3, 4).is_ok());
    }

    // -----------------------------------------------------------------------
    // Context conditioning: main_proj -> main_norm -> per-layer wkv/kv_norm/rope
    // -----------------------------------------------------------------------

    #[test]
    fn dspark_main_conditioning_projects_then_norms_hand_computed() {
        // main_proj rows select single concat entries with scales, so the
        // pre-norm projection is hand-readable; wkv (layer 0) stacks two 4x4
        // identities so the kv latent is [main_x | main_x].
        let mut main_proj = vec![0.0f32; 4 * 12];
        main_proj[0] = 2.0; // row 0 <- 2 * layer0[0]
        main_proj[1 * 12 + 4] = 1.0; // row 1 <- layer1[0]
        main_proj[2 * 12 + 8] = -1.0; // row 2 <- -layer2[0]
        main_proj[3 * 12 + 1] = 1.0; // row 3 <- layer0[1]
        let mut wkv0 = vec![0.0f32; 8 * 4];
        for j in 0..4 {
            wkv0[j * 4 + j] = 1.0;
            wkv0[(4 + j) * 4 + j] = 1.0;
        }
        let craft = Craft {
            main_proj: Some(main_proj),
            wkv0: Some(wkv0),
            ..Craft::random()
        };
        let rig = cpu_rig("main-cond", &craft);

        // One context position with explicit averaged rows.
        let a = vec![0.5f32, -1.0, 2.0, 0.25];
        let b = vec![1.0f32, 1.0, -0.5, 0.0];
        let c = vec![-2.0f32, 0.5, 1.0, 3.0];
        let aux = OwnedAux {
            layers: vec![0, 1, 2],
            base: 0,
            rows: vec![vec![a.clone(), b.clone(), c.clone()]],
        };
        let tokens = [0u32, 1];
        let mut state = rig.module.context();
        rig.module
            .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
            .unwrap();

        // Hand pipeline: proj -> main_norm(ones) -> wkv -> kv_norm(ones) ->
        // rope at position 0 (identity rotation).
        let pre = [2.0 * a[0], b[0], -c[0], a[1]];
        let main_x = ref_rms(&pre, &[1.0; 4], 1e-6);
        let mut kv_pre = main_x.clone();
        kv_pre.extend_from_slice(&main_x);
        let expected = ref_rms(&kv_pre, &[1.0; 8], 1e-6);
        assert_eq!(state.inserted_len(), 1);
        assert_close(&state.ctx[0][..8], &expected, 1e-6, "layer 0 latent");

        // Had the order been norm-then-proj, row 0 would differ (2*a0 is not
        // preserved by a pre-projection norm of the concat): guard the order
        // by checking the pre-norm ratio structure survived — element 2 of
        // the latent is -c0 scaled by the SAME factors as element 0's 2*a0.
        let ratio = state.ctx[0][2] / state.ctx[0][0];
        assert!(
            (ratio - (-c[0] / (2.0 * a[0]))).abs() < 1e-5,
            "proj-then-norm must preserve projection ratios, got {ratio}"
        );
    }

    #[test]
    fn dspark_context_kv_ropes_absolute_positions_and_incremental_matches_batch() {
        // wkv0 = [I; I] again, and IDENTICAL aux rows at every position, so
        // context latents differ ONLY by their rope rotation.
        let mut wkv0 = vec![0.0f32; 8 * 4];
        for j in 0..4 {
            wkv0[j * 4 + j] = 1.0;
            wkv0[(4 + j) * 4 + j] = 1.0;
        }
        let craft = Craft {
            wkv0: Some(wkv0),
            ..Craft::random()
        };
        let rig = cpu_rig("rope-pos", &craft);
        let row = vec![0.7f32, -0.3, 1.1, 0.4];
        let aux = OwnedAux {
            layers: vec![0, 1, 2],
            base: 0,
            rows: vec![vec![row.clone(); 3]; 4],
        };
        let tokens = [2u32, 2, 2, 2, 3];
        let mut state = rig.module.context();
        rig.module
            .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
            .unwrap();
        assert_eq!(state.inserted_len(), 4);

        // Position 0's rotation is the identity, so latent 0 is the pre-rope
        // latent; every later latent must be exactly its rotation at the
        // ABSOLUTE position (fixture rope_dims 4, base 10000).
        let latent0 = state.ctx[0][..8].to_vec();
        for pos in 1..4 {
            let expected = ref_rope_tail(&latent0, 4, pos, 10_000.0);
            assert_close(
                &state.ctx[0][pos * 8..(pos + 1) * 8],
                &expected,
                1e-5,
                &format!("latent at position {pos}"),
            );
        }

        // Incremental two-call build == one-shot build, bit for bit.
        let mut split = rig.module.context();
        let aux_half = OwnedAux {
            layers: vec![0, 1, 2],
            base: 0,
            rows: vec![vec![row.clone(); 3]; 2],
        };
        rig.module
            .propose_tokens(&rig.host, &mut split, &tokens[..3], &aux_half, 3)
            .unwrap();
        assert_eq!(split.inserted_len(), 2);
        rig.module
            .propose_tokens(&rig.host, &mut split, &tokens, &aux, 3)
            .unwrap();
        assert_eq!(split.inserted_len(), 4);
        for l in 0..3 {
            assert_eq!(state.ctx[l], split.ctx[l], "layer {l} incremental context");
        }
    }

    // -----------------------------------------------------------------------
    // Block attention: hand-computed core + visibility semantics
    // -----------------------------------------------------------------------

    #[test]
    fn dspark_block_attention_hand_computed() {
        // One head, head_dim 2, two keys; scale = 2^-0.5.
        let q = vec![vec![2.0f32, 0.0]];
        let k0 = [1.0f32, 0.0];
        let k1 = [0.0f32, 1.0];
        let keys: Vec<&[f32]> = vec![&k0, &k1];
        let scale = (2.0f32).powf(-0.5);
        let (s0, s1) = (2.0 * scale, 0.0);
        let (e0, e1) = (0.0f32.exp(), (s1 - s0).exp()); // max = s0
        let denom = e0 + e1;
        let expected = [
            (e0 / denom) * k0[0] + (e1 / denom) * k1[0],
            (e0 / denom) * k0[1] + (e1 / denom) * k1[1],
        ];
        let out = dspark_block_attention(&q, &keys, None, 1, 2);
        assert_close(&out[0], &expected, 1e-6, "no sink");

        // A sink adds exp mass to the denominator but no value.
        let sink = 1.0f32;
        let max = s0.max(sink);
        let (e0, e1, es) = ((s0 - max).exp(), (s1 - max).exp(), (sink - max).exp());
        let denom = e0 + e1 + es;
        let expected = [
            (e0 / denom) * k0[0] + (e1 / denom) * k1[0],
            (e0 / denom) * k0[1] + (e1 / denom) * k1[1],
        ];
        let out = dspark_block_attention(&q, &keys, Some(&[sink]), 1, 2);
        assert_close(&out[0], &expected, 1e-6, "with sink");

        // Two heads attend independently with their own sinks.
        let q2 = vec![vec![2.0f32, 0.0, 0.0, 3.0]];
        let k0 = [1.0f32, 0.0, 0.5, 0.5];
        let keys: Vec<&[f32]> = vec![&k0];
        let out = dspark_block_attention(&q2, &keys, Some(&[0.0, 10.0]), 2, 2);
        // Head 0: score 2*scale vs sink 0. Head 1: score (3*0.5)*scale vs
        // sink 10 — the sink soaks almost all mass, so head 1's output is
        // tiny relative to head 0's.
        assert!(out[0][0] > 0.5, "head 0 keeps most mass: {:?}", out[0]);
        assert!(
            out[0][2].abs() < 0.01 && out[0][3].abs() < 0.01,
            "head 1's sink absorbs the mass: {:?}",
            out[0]
        );
    }

    #[test]
    fn dspark_queries_see_future_block_and_windowed_context() {
        let layers = [0usize, 1, 2];
        let context = [0u32, 1, 2, 0, 1, 2, 0, 1];
        let tokens: Vec<u32> = context.iter().copied().chain([2]).collect();

        // (a) SWA window: with n_ctx = 8 and window 4, positions 0..4 are
        // outside every query's context slice. Perturbing an out-of-window
        // row leaves the drafts and base logits bit-identical; perturbing an
        // in-window row changes them.
        let rig = cpu_rig("window", &Craft::random());
        let run = |mutate_pos: Option<usize>| {
            let mut aux = owned_aux(&layers, 4, &context);
            if let Some(pos) = mutate_pos {
                for group in &mut aux.rows[pos] {
                    for value in group.iter_mut() {
                        *value += 1.0;
                    }
                }
            }
            let mut state = rig.module.context();
            let drafts = rig
                .module
                .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
                .unwrap();
            (drafts, state.last_base_logits.clone())
        };
        let (drafts_base, logits_base) = run(None);
        assert_eq!(drafts_base.len(), 3);
        let (drafts_out, logits_out) = run(Some(3));
        assert_eq!(drafts_out, drafts_base, "position 3 is outside the window");
        assert_eq!(
            logits_out, logits_base,
            "out-of-window context must be invisible bit-for-bit"
        );
        let (_, logits_in) = run(Some(7));
        assert_ne!(
            logits_in, logits_base,
            "position 7 is inside every query's window"
        );

        // (b) Non-causal block: the ONLY cross-slot path is attention, so a
        // different noise token changing the ANCHOR slot's logits proves the
        // anchor query attends to the FUTURE noise keys.
        let mut noisy = fixture_config();
        noisy.noise_token = 2;
        let rig_noise1 = cpu_rig("noise-1", &Craft::random());
        let rig_noise2 = cpu_rig_with("noise-2", &Craft::random(), &noisy);
        let propose = |rig: &CpuRig| {
            let aux = owned_aux(&layers, 4, &context);
            let mut state = rig.module.context();
            rig.module
                .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
                .unwrap();
            state.last_base_logits.clone()
        };
        let logits_n1 = propose(&rig_noise1);
        let logits_n2 = propose(&rig_noise2);
        assert_ne!(
            logits_n1[0], logits_n2[0],
            "the anchor's own logits must see the future noise keys (non-causal block)"
        );
        // Determinism sanity: identical rigs reproduce identical logits.
        let rig_again = cpu_rig("noise-1-again", &Craft::random());
        assert_eq!(logits_n1, propose(&rig_again));
    }

    // -----------------------------------------------------------------------
    // Markov head
    // -----------------------------------------------------------------------

    #[test]
    fn dspark_markov_bias_and_chain_hand_computed() {
        // w1 rows pick w2 columns: bias(0) = w2[:,0], bias(1) = w2[:,1],
        // bias(2) = w2[:,2], bias(3) = w2[:,0]+w2[:,1]+w2[:,2].
        let w1 = vec![
            1.0f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ];
        let w2 = vec![
            0.5f32, -1.0, 2.0, //
            1.5, 0.25, -0.75, //
            -2.0, 3.0, 0.125, //
            0.0, 1.0, -1.0,
        ];
        let craft = Craft {
            markov_w1: Some(w1),
            markov_w2: Some(w2.clone()),
            ..Craft::random()
        };
        let rig = cpu_rig("markov", &craft);
        for prev in 0..3u32 {
            let bias = rig.module.markov_bias(&rig.host, prev).unwrap();
            let expected: Vec<f32> = (0..4).map(|row| w2[row * 3 + prev as usize]).collect();
            assert_close(&bias, &expected, 1e-6, &format!("bias(prev={prev})"));
        }
        let bias3 = rig.module.markov_bias(&rig.host, 3).unwrap();
        let expected: Vec<f32> = (0..4)
            .map(|row| w2[row * 3] + w2[row * 3 + 1] + w2[row * 3 + 2])
            .collect();
        assert_close(&bias3, &expected, 1e-6, "bias(prev=3)");

        // The chain: prev starts at the ANCHOR; each draft becomes the next
        // prev. Recompute from the captured base logits.
        let context = [0u32, 1, 2, 0, 1, 2];
        let tokens: Vec<u32> = context.iter().copied().chain([2]).collect();
        let aux = owned_aux(&[0, 1, 2], 4, &context);
        let mut state = rig.module.context();
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
            .unwrap();
        assert_eq!(drafts.len(), 3);
        let mut prev = *tokens.last().unwrap();
        for (i, &draft) in drafts.iter().enumerate() {
            let bias = rig.module.markov_bias(&rig.host, prev).unwrap();
            let biased: Vec<f32> = state.last_base_logits[i]
                .iter()
                .zip(&bias)
                .map(|(l, b)| l + b)
                .collect();
            assert_eq!(
                draft,
                argmax(&biased).unwrap(),
                "draft {i} must be the Markov-biased argmax chained from prev {prev}"
            );
            prev = draft;
        }
    }

    // -----------------------------------------------------------------------
    // Anchor/noise layout + emission contract
    // -----------------------------------------------------------------------

    #[test]
    fn dspark_anchor_noise_slot_layout() {
        // Attention + MoE zeroed: slots cannot see each other and position
        // enters nowhere, so each slot's logits are a pure function of its
        // own token id. Slot 0 must carry the anchor; slots 1.. the noise
        // token (equal logits, anchor-independent).
        let craft = Craft {
            zero_attn_out: true,
            zero_moe_down: true,
            zero_markov: true,
            ..Craft::random()
        };
        let rig = cpu_rig("layout", &craft);
        let context = [0u32, 1, 2, 0];
        let propose = |anchor: u32| {
            let tokens: Vec<u32> = context.iter().copied().chain([anchor]).collect();
            let aux = owned_aux(&[0, 1, 2], 4, &context);
            let mut state = rig.module.context();
            let drafts = rig
                .module
                .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
                .unwrap();
            (drafts, state.last_base_logits.clone())
        };
        let (drafts_a, logits_a) = propose(0);
        let (drafts_b, logits_b) = propose(2);
        assert_eq!(
            logits_a[1], logits_a[2],
            "both noise slots embed the same token and cannot differ"
        );
        assert_ne!(logits_a[0], logits_b[0], "slot 0 must carry the anchor");
        assert_eq!(
            logits_a[1], logits_b[1],
            "noise slots must not depend on the anchor"
        );
        assert_eq!(drafts_a[1..], drafts_b[1..]);
    }

    #[test]
    fn dspark_emission_count_alignment_and_k_cap() {
        let rig = cpu_rig("kcap", &Craft::random());
        let context = [0u32, 1, 2, 0, 1];
        let tokens: Vec<u32> = context.iter().copied().chain([2]).collect();
        let propose_k = |k: usize| {
            let aux = owned_aux(&[0, 1, 2], 4, &context);
            let mut state = rig.module.context();
            rig.module
                .propose_tokens(&rig.host, &mut state, &tokens, &aux, k)
                .unwrap()
        };
        assert!(propose_k(0).is_empty(), "k = 0 proposes nothing");
        let full = propose_k(3);
        assert_eq!(full.len(), 3, "the full block emits block_size drafts");
        assert!(full.iter().all(|&d| (d as usize) < 4), "drafts in vocab");
        // The Markov chain is left-to-right, so smaller k returns an exact
        // prefix; k beyond the block truncates to block_size.
        assert_eq!(propose_k(1), full[..1]);
        assert_eq!(propose_k(2), full[..2]);
        assert_eq!(propose_k(10), full);
    }

    // -----------------------------------------------------------------------
    // Incremental reuse, divergence, taps-base flooring, error paths
    // -----------------------------------------------------------------------

    #[test]
    fn dspark_incremental_reuse_divergence_and_base_floor() {
        let rig = cpu_rig("reuse", &Craft::random());
        let layers = [0usize, 1, 2];
        let context = [0u32, 1, 2, 0, 1, 2, 0];
        let tokens: Vec<u32> = context.iter().copied().chain([1]).collect();

        let mut state = rig.module.context();
        let aux = owned_aux(&layers, 4, &context);
        rig.module
            .propose_tokens(&rig.host, &mut state, &tokens, &aux, 3)
            .unwrap();
        assert_eq!(state.inserted_len(), 7);
        assert_eq!(state.ctx_base(), 0);
        assert_eq!(state.rows_inserted, 7);
        assert_eq!(state.truncations, 0);

        // The verify loop accepted one token: context grows by 1, no rebuild.
        let grown: Vec<u32> = context.iter().copied().chain([1, 2]).collect();
        let aux2 = owned_aux(&layers, 4, &grown[..8]);
        rig.module
            .propose_tokens(&rig.host, &mut state, &grown, &aux2, 3)
            .unwrap();
        assert_eq!(state.inserted_len(), 8);
        assert_eq!(state.rows_inserted, 8, "only the new position was built");
        assert_eq!(state.truncations, 0);

        // A different request sharing tokens[..4]: truncate to the common
        // prefix, rebuild the rest.
        let diverged = [0u32, 1, 2, 0, 2, 2, 0, 1, 0];
        let aux3 = owned_aux(&layers, 4, &diverged[..8]);
        rig.module
            .propose_tokens(&rig.host, &mut state, &diverged, &aux3, 3)
            .unwrap();
        assert_eq!(state.truncations, 1);
        assert_eq!(state.inserted_len(), 8);
        assert_eq!(state.rows_inserted, 8 + 4, "positions 4..8 were rebuilt");

        // Prefix-cache restore: taps floored at base 4 — the context rebases
        // there (positions below stay uncovered) and proposals still work.
        let mut floored = rig.module.context();
        let aux4 = owned_aux_at(&layers, 4, &context, 4);
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut floored, &tokens, &aux4, 3)
            .unwrap();
        assert_eq!(drafts.len(), 3);
        assert_eq!(floored.ctx_base(), 4);
        assert_eq!(floored.inserted_len(), 3);

        // A later request whose taps base moved past the reusable prefix:
        // everything resets at the new base.
        let unrelated = [3u32, 3, 3, 3, 3, 3, 3, 0];
        let aux5 = owned_aux_at(&layers, 4, &unrelated[..7], 6);
        rig.module
            .propose_tokens(&rig.host, &mut floored, &unrelated, &aux5, 3)
            .unwrap();
        assert_eq!(floored.ctx_base(), 6);
        assert_eq!(floored.inserted_len(), 1);
    }

    #[test]
    fn dspark_propose_rejects_mismatched_tokens_and_anchor() {
        let rig = cpu_rig("rejects", &Craft::random());
        let layers = [0usize, 1, 2];
        let context = [0u32, 1, 2];
        let aux = owned_aux(&layers, 4, &context);

        // tokens must be taps + pending, exactly.
        let mut state = rig.module.context();
        let err = rig
            .module
            .propose_tokens(&rig.host, &mut state, &[0, 1, 2], &aux, 3)
            .expect_err("length mismatch must fail")
            .to_string();
        assert!(err.contains("tap positions"), "got: {err}");

        // Out-of-vocab anchors fail loudly (vocab 4).
        let mut state = rig.module.context();
        let err = rig
            .module
            .propose_tokens(&rig.host, &mut state, &[0, 1, 2, 99], &aux, 3)
            .expect_err("out-of-vocab anchor must fail")
            .to_string();
        assert!(err.contains("anchor"), "got: {err}");
    }

    /// Engine-taps smoke: the module consumes REAL trunk taps (stream
    /// averaging, base handling) end to end and reuses its context across
    /// growing proposals.
    #[test]
    fn dspark_cpu_forward_with_engine_taps() {
        let rig = cpu_rig("engine-taps", &Craft::random());
        let tokens = [0u32, 1, 2, 0, 1, 2, 0, 1];
        let taps = taps_for(&rig.engine, &tokens[..7]);
        assert_eq!(DsparkAux::positions(&taps), 7);
        let mut state = rig.module.context();
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &taps, 3)
            .unwrap();
        assert_eq!(drafts.len(), 3);
        assert!(drafts.iter().all(|&d| (d as usize) < 4));

        // Accept the first draft: the deterministic engine reproduces the
        // same prefix taps, so the context extends without truncation.
        let mut grown = tokens.to_vec();
        grown.push(drafts[0]);
        let taps2 = taps_for(&rig.engine, &grown[..8]);
        let drafts2 = rig
            .module
            .propose_tokens(&rig.host, &mut state, &grown, &taps2, 3)
            .unwrap();
        assert_eq!(drafts2.len(), 3);
        assert_eq!(state.truncations, 0);
        assert_eq!(state.inserted_len(), 8);

        // A restore-floored taps buffer floors the rebuilt context.
        let mut floored = rig.module.context();
        let taps3 = taps_for_at(&rig.engine, &tokens[..7], 4);
        assert_eq!(DsparkAux::base(&taps3), 4);
        rig.module
            .propose_tokens(&rig.host, &mut floored, &tokens, &taps3, 3)
            .unwrap();
        assert_eq!(floored.ctx_base(), 4);
        assert_eq!(floored.inserted_len(), 3);
    }

    // -----------------------------------------------------------------------
    // Real-shard census (ignored: needs the downloaded checkpoint; CPU-only)
    // -----------------------------------------------------------------------

    pub(super) fn real_dspark_dir() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("HI_DSV4_DSPARK_PATH") {
            let path = PathBuf::from(path);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home).join(".hi/models/deepseek-v4-flash/dspark");
        path.is_dir().then_some(path)
    }

    /// Hand-built real-model dims (the census runs without loading the 685B
    /// trunk GGUF; values are the byte-verified checkpoint contract).
    fn real_dims(config: DsparkConfig) -> DsparkDims {
        DsparkDims {
            geometry: DsV4Geometry {
                embed: 4096,
                heads: 64,
                head_dim: 512,
                rope_dims: 64,
                q_lora: 1024,
                o_groups: 8,
                o_rank: 1024,
                window: Some(128),
                hc: 4,
                sinkhorn_iterations: 20,
                idx_heads: 64,
                idx_key: 128,
                idx_top_k: 512,
                experts: 256,
                moe_top_k: 6,
                expert_weights_norm: true,
                expert_weights_scale: 1.5,
                vocab: 129_280,
                context: 1 << 20,
            },
            expert_ff: 2048,
            shared_ff: 2048,
            trunk_layers: 43,
            rope_base: 10_000.0,
            swiglu_clamp: 10.0,
            rms_eps: 1e-6,
            hc_eps: 1e-6,
            config,
        }
    }

    /// The remap contract as a name classifier: every real shard tensor must
    /// be a per-layer weight, a model-level attachment, a `.scale` sibling
    /// (consumed implicitly by the fp8/fp4 dequant), or the dropped
    /// confidence head. "unknown" fails the census.
    fn classify_real_tensor(name: &str) -> &'static str {
        let Some(rest) = name.strip_prefix("mtp.") else {
            return "unknown";
        };
        let Some((layer, rest)) = rest.split_once('.') else {
            return "unknown";
        };
        if layer.parse::<usize>().is_err() {
            return "unknown";
        }
        if rest.starts_with("confidence_head.") {
            return "skip:confidence";
        }
        if rest.ends_with(".scale") {
            return "scale-sibling";
        }
        const MODEL_LEVEL: [&str; 8] = [
            "main_proj.weight",
            "main_norm.weight",
            "norm.weight",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
        ];
        if MODEL_LEVEL.contains(&rest) {
            return "model";
        }
        const LAYER_LEVEL: [&str; 21] = [
            "attn.wq_a.weight",
            "attn.q_norm.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.kv_norm.weight",
            "attn.attn_sink",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
            "ffn.gate.weight",
            "ffn.gate.bias",
            "ffn.shared_experts.w1.weight",
            "ffn.shared_experts.w2.weight",
            "ffn.shared_experts.w3.weight",
        ];
        if LAYER_LEVEL.contains(&rest) {
            return "layer";
        }
        if let Some(expert) = rest.strip_prefix("ffn.experts.")
            && let Some((index, proj)) = expert.split_once('.')
            && index.parse::<usize>().is_ok()
            && matches!(proj, "w1.weight" | "w2.weight" | "w3.weight")
        {
            return "expert";
        }
        "unknown"
    }

    /// Census + FULL load of the real shard trio: every tensor classified,
    /// every shape validated against the contract dims, fp8 dense dequant +
    /// fp4->MXFP4 repack exercised end to end, payload sanity checked.
    /// `cargo test -p hi-cuda --release dspark_real_shard -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the downloaded DeepSeek-V4-Flash DSpark shard trio (~10.9 GB)"]
    fn dspark_real_shard_census_and_load() {
        let Some(dir) = real_dspark_dir() else {
            eprintln!("skipping: DSpark checkpoint directory not found");
            return;
        };
        let mut config = DsparkConfig::real_default();
        let config_json = dir.join("config.json");
        if config_json.exists() {
            config.apply_config_json(&config_json).unwrap();
        }
        assert_eq!(config.layers, 3);
        assert_eq!(config.block_size, 5);
        assert_eq!(config.noise_token, 128_799);
        assert_eq!(config.target_layers, vec![40, 41, 42]);
        assert_eq!(config.markov_rank, 256);
        let dims = real_dims(config);

        let shards = DsparkShards::open_dir(&dir).unwrap();
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        let mut unknown = Vec::new();
        for name in shards.names() {
            let class = classify_real_tensor(&name);
            *counts.entry(class).or_default() += 1;
            if class == "unknown" {
                unknown.push(name);
            }
        }
        eprintln!("census: {counts:?}");
        assert!(
            unknown.is_empty(),
            "unclassified shard tensors (remap out of date): {unknown:?}"
        );
        // 3 layers x 256 experts x 3 projections.
        assert_eq!(counts.get("expert").copied().unwrap_or(0), 3 * 256 * 3);
        assert_eq!(counts.get("skip:confidence").copied().unwrap_or(0), 1);

        let started = Instant::now();
        let load = load_dspark(&shards, &dims).unwrap();
        eprintln!(
            "loaded in {:.0}s: {:.0} MiB resident dense, {:.2} GiB packed experts ({} tensors)",
            started.elapsed().as_secs_f64(),
            load.resident_bytes() as f64 / (1u64 << 20) as f64,
            load.expert_bytes() as f64 / (1u64 << 30) as f64,
            load.experts.len(),
        );
        assert_eq!(load.experts.len(), 9);
        for entry in &load.experts {
            assert_eq!(entry.expert_count, 256);
            assert_eq!(entry.dtype, GgufTensorType::MXFP4);
        }
        // Dequant sanity: finite, non-degenerate payloads.
        let row = load.weights.markov_w1.row(0).unwrap();
        assert!(row.iter().all(|value| value.is_finite()));
        assert!(row.iter().any(|value| *value != 0.0));
        let main_proj = load
            .dense
            .iter()
            .find(|entry| entry.matrix.name == names::MAIN_PROJ)
            .unwrap();
        let values = main_proj.payload.to_f32();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
        assert_eq!(values.len(), 4096 * 12288);
    }
}

/// GPU-provider parity on the fixture trio, the Drafter wiring, the serving
/// loop with the REAL DsparkDrafter end to end (lossless by construction),
/// and the ignored real-model acceptance measurement.
#[cfg(all(test, feature = "native-cuda"))]
mod native_tests {
    use futures_util::StreamExt;
    use hi_local_core::backend::{GenerationEvent, GenerationRequest, InferenceBackend};

    use super::tests::{
        Craft, assert_close, cpu_rig, dspark_tempdir, fixture_config, real_dspark_dir, taps_for,
        write_dspark_fixture,
    };
    use super::*;
    use crate::dsv4_backend::DeepSeekV4Backend;
    use crate::dsv4_cpu::fixture::{tempfile_path, write_deepseek4_spec_gguf};

    /// A GPU engine + DsparkDrafter over fixtures byte-identical to
    /// `cpu_rig`'s (the writers are deterministic).
    fn gpu_drafter(name: &str, craft: &Craft) -> (DeepSeekV4GpuEngine, DsparkDrafter) {
        let gguf_path = tempfile_path(&format!("dspark-{name}"));
        write_deepseek4_spec_gguf(&gguf_path);
        let dir = dspark_tempdir(&format!("{name}-gpu"));
        write_dspark_fixture(&dir, craft, &fixture_config());
        let gpu = DeepSeekV4GpuEngine::load(&gguf_path).unwrap();
        let drafter = DsparkDrafter::from_dir(&gpu, &dir).unwrap();
        (gpu, drafter)
    }

    /// The module forward must agree between the CPU host reference and the
    /// CUDA provider — same tokens, same taps (the CPU engine's, so trunk
    /// GEMV reduction differences cannot leak into the comparison).
    #[test]
    fn dspark_fixture_cpu_gpu_parity() {
        let craft = Craft::random();
        let rig = cpu_rig("gpu-parity", &craft);
        let (_gpu, mut drafter) = gpu_drafter("gpu-parity", &craft);

        let tokens = [0u32, 1, 2, 0, 1, 2, 0, 1];
        let taps = taps_for(&rig.engine, &tokens[..7]);

        let mut cpu_state = rig.module.context();
        let cpu_drafts = rig
            .module
            .propose_tokens(&rig.host, &mut cpu_state, &tokens, &taps, 3)
            .unwrap();
        let ctx = DraftContext {
            tokens: &tokens,
            taps: Some(&taps),
            k: 3,
        };
        let gpu_drafts = drafter.propose(&ctx);
        assert_eq!(cpu_drafts, gpu_drafts, "draft tokens diverged");
        assert_eq!(gpu_drafts.len(), 3);

        // Context latents and base logits agree within GEMV-vs-host f32
        // reduction tolerance (both providers hold exact-f32 fixtures).
        for l in 0..3 {
            assert_close(
                &cpu_state.ctx[l],
                &drafter.state.ctx[l],
                1e-4,
                &format!("layer {l} context latents"),
            );
        }
        assert_eq!(
            cpu_state.last_base_logits.len(),
            drafter.state.last_base_logits.len()
        );
        for (slot, (cpu, gpu)) in cpu_state
            .last_base_logits
            .iter()
            .zip(&drafter.state.last_base_logits)
            .enumerate()
        {
            assert_close(cpu, gpu, 1e-4, &format!("slot {slot} base logits"));
        }
    }

    /// The drafter asks the engine for exactly the configured target-layer
    /// taps, stream-averaged (pre_hc_head off).
    #[test]
    fn dspark_tap_config_requests_target_layers() {
        let (_gpu, drafter) = gpu_drafter("tap-config", &Craft::random());
        let config = drafter.tap_config();
        assert!(!config.pre_hc_head);
        assert_eq!(config.aux_layers, vec![0, 1, 2]);
    }

    fn generation_request(prompt: &str, max_tokens: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
            messages: Vec::new(),
        }
    }

    async fn collect_generation(
        backend: &DeepSeekV4Backend,
        request: GenerationRequest,
    ) -> (Vec<u32>, String) {
        let mut stream = backend.stream_generate(request).await.unwrap();
        let mut ids = Vec::new();
        let mut text = String::new();
        let mut finished = false;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                GenerationEvent::TokenDelta { token_id, text: t } => {
                    ids.push(token_id);
                    text.push_str(&t);
                }
                GenerationEvent::Finished { output } => {
                    assert_eq!(text, output.text);
                    finished = true;
                }
            }
        }
        assert!(finished, "stream must end with Finished");
        (ids, text)
    }

    fn health_counter(backend: &DeepSeekV4Backend, key: &str) -> u64 {
        let health = backend.health();
        let marker = format!("{key}=");
        let start = health
            .quantization
            .find(&marker)
            .unwrap_or_else(|| panic!("{key} missing from health: {}", health.quantization))
            + marker.len();
        health.quantization[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap()
    }

    /// The full serving loop with the REAL DsparkDrafter over the fixture:
    /// the emitted stream must be byte-identical to the sequential baseline
    /// (the verify loop guarantees losslessness regardless of draft quality),
    /// the drafter must actually run, and a warm rerun (prefix restore) must
    /// reproduce it again through the taps-base flooring path.
    #[tokio::test]
    async fn dspark_fixture_backend_output_identical_with_dspark_drafter() {
        let prompt = "abcabcab";
        let max_tokens = 8u32;

        let baseline_path = tempfile_path("dspark-e2e-baseline");
        write_deepseek4_spec_gguf(&baseline_path);
        let baseline = DeepSeekV4Backend::load_with_prefix_config(
            &baseline_path,
            Some("dsv4-fixture".to_string()),
            4,
            1 << 20,
        )
        .unwrap();
        let (base_ids, base_text) =
            collect_generation(&baseline, generation_request(prompt, max_tokens)).await;

        let spec_gguf = tempfile_path("dspark-e2e-spec");
        write_deepseek4_spec_gguf(&spec_gguf);
        let dir = dspark_tempdir("e2e");
        write_dspark_fixture(&dir, &Craft::random(), &fixture_config());
        let factory_dir = dir.clone();
        let backend = DeepSeekV4Backend::load_with_drafter(
            &spec_gguf,
            Some("dsv4-fixture".to_string()),
            4,
            1 << 20,
            Box::new(move |engine| {
                Some(Box::new(
                    DsparkDrafter::from_dir(engine, &factory_dir)
                        .expect("fixture DSpark drafter must build"),
                ) as Box<dyn Drafter>)
            }),
        )
        .unwrap();

        let (ids, text) =
            collect_generation(&backend, generation_request(prompt, max_tokens)).await;
        assert_eq!(ids, base_ids, "speculative stream must be lossless");
        assert_eq!(text, base_text);
        assert!(health_counter(&backend, "spec_verify_steps") >= 1);
        assert!(
            health_counter(&backend, "spec_proposed") >= 1,
            "the DSpark drafter never proposed"
        );

        // Warm rerun: same conversation again (prefix snapshots were written
        // at accepted boundaries only). The rerun MUST reuse cached blocks,
        // the drafter floors its context at the taps base, and the stream
        // stays byte-identical.
        let proposed_cold = health_counter(&backend, "spec_proposed");
        let (rerun_ids, rerun_text) =
            collect_generation(&backend, generation_request(prompt, max_tokens)).await;
        assert_eq!(rerun_ids, base_ids);
        assert_eq!(rerun_text, base_text);
        assert!(
            health_counter(&backend, "reused_tokens") > 0,
            "the tap-requesting drafter must not block prefix restores"
        );
        assert!(
            health_counter(&backend, "spec_proposed") > proposed_cold,
            "the drafter must keep proposing after a restore"
        );
    }

    // ---- Real model (ignored: shared GPU + multi-minute load) -------------

    /// Real-model DSpark end-to-end: identical output with the drafter on vs
    /// off, printed acceptance stats. GPU 0 is SHARED with a training job —
    /// budget the pool for 10.3 GiB of pinned draft experts plus trunk
    /// headroom, and run explicitly:
    /// `HI_DSV4_EXPERT_POOL_GB=16 HI_DSV4_SPEC_K=5 CUDA_VISIBLE_DEVICES=0 \
    ///  cargo test -p hi-cuda --release --features native-cuda \
    ///  dspark_real_model_e2e -- --ignored --nocapture --test-threads=1`
    #[tokio::test]
    #[ignore = "needs the real checkpoint + DSpark shard trio and tens of GB of VRAM"]
    async fn dspark_real_model_e2e_lossless_and_acceptance() {
        let Some(gguf_path) = crate::dsv4_gpu::tests::real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let Some(dir) = real_dspark_dir() else {
            eprintln!("skipping: DSpark checkpoint directory not found");
            return;
        };
        let prompt = "The capital of France is Paris. The capital of Japan is";
        let max_tokens = 48u32;

        eprintln!("loading baseline backend...");
        let started = std::time::Instant::now();
        let baseline = DeepSeekV4Backend::load_with_prefix_config(
            &gguf_path,
            Some("dsv4-real".to_string()),
            256,
            1 << 30,
        )
        .unwrap();
        eprintln!("baseline loaded in {:.0}s", started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let (base_ids, base_text) =
            collect_generation(&baseline, generation_request(prompt, max_tokens)).await;
        let base_elapsed = started.elapsed();
        eprintln!(
            "baseline: {} tokens in {:.1}s ({:.2} tok/s): {base_text:?}",
            base_ids.len(),
            base_elapsed.as_secs_f64(),
            base_ids.len() as f64 / base_elapsed.as_secs_f64(),
        );
        drop(baseline); // release VRAM before the second engine

        eprintln!("loading speculative backend (DSpark drafter)...");
        let started = std::time::Instant::now();
        let factory_dir = dir.clone();
        let backend = DeepSeekV4Backend::load_with_drafter(
            &gguf_path,
            Some("dsv4-real".to_string()),
            256,
            1 << 30,
            Box::new(move |engine| {
                Some(Box::new(
                    DsparkDrafter::from_dir(engine, &factory_dir)
                        .expect("real DSpark drafter must build"),
                ) as Box<dyn Drafter>)
            }),
        )
        .unwrap();
        eprintln!(
            "spec backend loaded in {:.0}s",
            started.elapsed().as_secs_f64()
        );
        let started = std::time::Instant::now();
        let (ids, text) =
            collect_generation(&backend, generation_request(prompt, max_tokens)).await;
        let spec_elapsed = started.elapsed();

        let proposed = health_counter(&backend, "spec_proposed");
        let accepted = health_counter(&backend, "spec_accepted");
        let steps = health_counter(&backend, "spec_verify_steps");
        eprintln!(
            "dspark spec: {} tokens in {:.1}s ({:.2} tok/s); proposed {proposed} accepted {accepted} ({:.1}%) over {steps} verify steps",
            ids.len(),
            spec_elapsed.as_secs_f64(),
            ids.len() as f64 / spec_elapsed.as_secs_f64(),
            100.0 * accepted as f64 / proposed.max(1) as f64,
        );
        assert_eq!(ids, base_ids, "speculative output must be lossless");
        assert_eq!(text, base_text);
        assert!(steps >= 1 && proposed >= 1, "the drafter never engaged");
        // DeepSeek reports 60-85% per-position acceptance over MTP-1;
        // near-zero acceptance means a porting bug (tap averaging, main_norm
        // placement, noise token, non-causal mask, Markov chaining, or a
        // position off-by-one — see the module docs).
        assert!(
            accepted * 5 >= proposed,
            "acceptance {accepted}/{proposed} is bug-level low"
        );

        // The backend worker thread tears the real engine down asynchronously
        // after the drop; give it a moment so process exit does not race CUDA
        // driver teardown mid-free (observed as a post-"ok" SIGSEGV on the
        // 29-GB real engine).
        drop(backend);
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
