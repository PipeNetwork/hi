//! CPU reference forward pass for DeepSeek-V4-Flash (`deepseek4` GGUF arch).
//!
//! Mirrors the working MLX reference captured in
//! `docs/deepseek-v4-flash-port-spec.md`: hyper-connection residual streams
//! (sinkhorn-normalized stream mixing), latent-MQA attention (one shared
//! 512-wide KV latent per layer, K = V) over a 128-token raw sliding window
//! plus block-compressed long-range KV (per-layer APE compressors and a
//! lightning indexer on ratio-4 layers), grouped low-rank output projection,
//! attention sinks, and sqrt-softplus MoE routing with hash-routed leading
//! layers and per-layer SwiGLU clamps.
//!
//! This is a correctness oracle, not a fast path. Tokens are processed
//! strictly one at a time (S=1, prompt included), which keeps every mask
//! trivial: the raw window is exactly the ring of the last `sliding_window`
//! cached latents, and in decode form every complete compressed block is
//! causally visible, so the indexer only narrows the block set once it holds
//! more than `top_k` blocks. Large weight matrices stay as raw (mmap'd) GGUF
//! bytes and are dequantized transiently per use; packed expert tensors are
//! dequantized one expert slice at a time and the embedding/lm-head rows in
//! chunks. Only small tensors (norms, hyper-connection mixers, sinks, APEs,
//! routers, biases) are materialized as f32 up front, so a 284B model loads
//! without materializing weights in RAM.
//!
//! The per-token state machine itself is provider-agnostic: [`DsV4Engine`]
//! holds all small host-side math (hyper-connections/sinkhorn, rope, sink
//! softmax, ring cache, compressors, indexer, routing, clamps) exactly once,
//! and routes every heavy linear op — dense matvecs, the block-diagonal output
//! projection, and packed expert slices — through the [`DsV4Linear`] trait.
//! This module's [`DsV4CpuLinear`] implements the trait with transient mmap
//! dequantization (the oracle path); `dsv4_gpu` supplies the CUDA-resident
//! implementation against the same engine, so any CPU/GPU divergence is a GPU
//! bug by definition.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{
    GgufFile, GgufTensorType, GgufTokenizer, QwenGgufConfig, TensorView, dequantize_tensor_as_f32,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::qwen_cpu::{
    QwenCpuRunOptions, QwenCpuRunOutput, argmax, dot, load_vector, optional_vector,
    rms_norm_in_place, sample_from_logits_with_rng, sigmoid, silu, softmax_in_place, softplus,
    top_logits,
};

/// Rows dequantized per chunk when streaming a matrix through the CPU matvec
/// (keeps the transient f32 buffer at a few MB even for the vocab-sized
/// lm-head). 512 is a multiple of every GGUF quant block size (32 and 256), so
/// chunk boundaries always land on block boundaries and chunked dequantization
/// is bit-identical to dequantizing the whole matrix at once.
const MATVEC_CHUNK_ROWS: usize = 512;

/// Prompt tokens processed per batched-prefill pass when
/// `HI_DSV4_PREFILL_CHUNK` is unset. 1 selects the legacy strictly-sequential
/// path (one [`DsV4Engine::step`] per prompt token).
const DSV4_DEFAULT_PREFILL_CHUNK: usize = 64;

/// `HI_DSV4_PREFILL_CHUNK`: batched-prefill chunk size, resolved once at
/// engine load. Invalid values fall back to the default with a warning.
fn prefill_chunk_from_env() -> usize {
    let Ok(raw) = std::env::var("HI_DSV4_PREFILL_CHUNK") else {
        return DSV4_DEFAULT_PREFILL_CHUNK;
    };
    match raw.trim().parse::<usize>() {
        Ok(chunk) if chunk >= 1 => chunk,
        _ => {
            eprintln!(
                "ignoring invalid HI_DSV4_PREFILL_CHUNK '{raw}' (want an integer >= 1); using {DSV4_DEFAULT_PREFILL_CHUNK}"
            );
            DSV4_DEFAULT_PREFILL_CHUNK
        }
    }
}

/// Identifies one heavy-linear operand for a [`DsV4Linear::mul_vec`] call:
/// a dense mmap'd matrix, the block-diagonal grouped output projection, or a
/// single expert's slice of a rank-3 packed tensor.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TensorKey<'a> {
    /// y[rows] = W[rows, cols] · x[cols].
    Dense(&'a RawMatrix),
    /// Block-diagonal (attn_output_a): output rows g*rank..(g+1)*rank read only
    /// input slice g of `matrix.cols` elements; x is (rows/rank)*cols long.
    Grouped { matrix: &'a RawMatrix, rank: usize },
    /// y[out] = W_e[out, in] · x[in] over expert e's contiguous slice.
    Expert {
        experts: &'a RawExperts,
        expert: usize,
    },
}

/// Provider of the heavy linear ops for [`DsV4Engine`]. Implementations own
/// where the weight bytes live (mmap'd host memory, GPU-resident f16, streamed
/// expert slices); the engine owns everything else.
pub(crate) trait DsV4Linear {
    fn mul_vec(&self, key: TensorKey<'_>, x: &[f32]) -> Result<Vec<f32>>;

    /// One layer's ENTIRE MoE block for a batch of tokens: router logits,
    /// sqrt-softplus routing (hash tables / selection bias / top-k with the
    /// lower-index tie-break), the routed expert SwiGLU matmuls with the
    /// per-layer clamps, weighted accumulation, and the shared expert. The
    /// default implementation is the exact host path ([`host_moe_block`]) —
    /// per-token router matvecs, host routing math, per-expert matmuls through
    /// [`DsV4Linear::mul_mat`] — so a provider that does not override it
    /// behaves bit-identically to the pre-seam engine code. The GPU provider
    /// overrides it to run the whole block device-side with a single interior
    /// host sync (Wave-2 Stage 2a).
    fn moe_block(
        &self,
        ctx: &DsV4MoeBlockCtx<'_>,
        xs: &[Vec<f32>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        host_moe_block(self, ctx, xs, tokens)
    }

    /// Batched form used by the chunked prefill: one output row per input row.
    /// The default loops [`DsV4Linear::mul_vec`], which keeps a provider
    /// bit-identical to its sequential path (the CPU oracle relies on this);
    /// GPU providers override it with true GEMM batching.
    fn mul_mat(&self, key: TensorKey<'_>, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        xs.iter().map(|x| self.mul_vec(key, x)).collect()
    }

    /// Optional prefetch hook: called once per MoE layer after routing is known
    /// and before the layer's expert matmuls, with the layer's three expert
    /// tensors and the deduplicated set of expert ids the layer will use. A GPU
    /// provider issues the missing expert-slice H2D copies on a copy stream so
    /// the subsequent expert GEMVs hit; the default is a no-op (the CPU oracle
    /// and the synchronous fallback need nothing here). Purely an optimization —
    /// results are identical whether or not it runs.
    fn prefetch_experts(
        &self,
        _tensors: DsV4ExpertTensors<'_>,
        _expert_ids: &[usize],
    ) -> Result<()> {
        Ok(())
    }

    /// Scoped speculative-verify hint: while set, [`DsV4Linear::mul_mat`]
    /// must be bit-identical to looping [`DsV4Linear::mul_vec`] over the rows.
    /// The default is a no-op because the default `mul_mat` already loops (the
    /// CPU oracle is always exact); the GPU provider uses it to suspend its
    /// opt-in `HI_DSV4_PREFILL_GEMM=1` batching for the duration of a verify
    /// chunk — greedy speculative acceptance is lossless only because verify
    /// logits reproduce the sequential step path exactly.
    fn set_exact_batching(&self, _exact: bool) {}

    /// Wave-2 Stage 2b hook: serve one WHOLE decode step (S=1) device-side.
    /// `None` declines — the engine then runs the host step, today's exact
    /// path (the default, the CPU oracle, and the `HI_DSV4_HOST_STEP=1` kill
    /// switch all land there). A provider that accepts must (a) return the
    /// same logits the host step would, (b) advance `state` exactly as the
    /// host step would — including the raw ring, compressor/indexer pending
    /// and blocks, and `pos` — so snapshots cloned from the state stay exact,
    /// and (c) leave `state` untouched on error.
    fn try_device_step(
        &self,
        _engine: &DsV4Engine<Self>,
        _state: &mut DsV4State,
        _token: u32,
    ) -> Option<Result<Vec<f32>>>
    where
        Self: Sized,
    {
        None
    }

    /// Wave-3 hook: serve a whole verify/re-feed CHUNK of up to
    /// [`DSV4_DEVICE_VERIFY_CAP`] tokens device-side — the S>1 analog of
    /// [`DsV4Linear::try_device_step`], with the same contract per position:
    /// `want_logits` returns one logits row per token (bit-exact with running
    /// [`DsV4Engine::host_step`] over the same tokens — the speculative
    /// losslessness requirement), `state` advances over every token exactly
    /// as sequential host steps would, `taps` (when given) captures the same
    /// rows a tapped host step would, and `state` is untouched on error.
    /// `want_logits = false` is the rewind re-feed form (no output heads).
    /// `None` declines to the host chunk path.
    fn try_device_verify(
        &self,
        _engine: &DsV4Engine<Self>,
        _state: &mut DsV4State,
        _tokens: &[u32],
        _taps: Option<&mut DsV4Taps>,
        _want_logits: bool,
    ) -> Option<Result<Vec<Vec<f32>>>>
    where
        Self: Sized,
    {
        None
    }
}

/// Maximum tokens per [`DsV4Linear::try_device_verify`] chunk. Sized for the
/// largest verify the spec loop issues (pending + K drafts, K <= 7 for
/// DFlash/DSpark block drafters); the GPU provider's chunk arena carries this
/// many per-position slots. Longer verifies loop chunks of this size.
pub(crate) const DSV4_DEVICE_VERIFY_CAP: usize = 8;

/// The CPU-oracle [`DsV4Linear`]: every matrix stays as raw GGUF bytes in the
/// shared mmap and is dequantized transiently per use, in row chunks.
#[derive(Debug)]
pub(crate) struct DsV4CpuLinear {
    gguf: Arc<GgufFile>,
}

/// Public CPU reference entry point: [`DsV4Engine`] driven by
/// [`DsV4CpuLinear`]. Kept as a thin wrapper so the engine internals stay
/// crate-private while this type's API is unchanged.
#[derive(Debug)]
pub struct DeepSeekV4CpuReference {
    engine: DsV4Engine<DsV4CpuLinear>,
}

impl DeepSeekV4CpuReference {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(GgufFile::open(path)?)
    }

    /// Takes ownership of the GGUF: the reference keeps weights as raw mmap'd
    /// bytes and dequantizes them per use, so the file must stay open.
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        let gguf = Arc::new(gguf);
        let linear = DsV4CpuLinear { gguf: gguf.clone() };
        Ok(Self {
            engine: DsV4Engine::new(gguf, linear, "cpu-reference")?,
        })
    }

    pub fn config(&self) -> &QwenGgufConfig {
        self.engine.config()
    }

    pub fn tokenizer(&self) -> &GgufTokenizer {
        self.engine.tokenizer()
    }

    /// Logits after the last input token, from a fresh per-run state.
    pub fn last_logits(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.engine.last_logits(input_ids)
    }

    pub fn run_tokens(
        &self,
        input_ids: &[u32],
        options: QwenCpuRunOptions,
    ) -> Result<QwenCpuRunOutput> {
        self.engine.run_tokens(input_ids, options)
    }

    pub fn run_prompt(&self, prompt: &str, options: QwenCpuRunOptions) -> Result<QwenCpuRunOutput> {
        self.engine.run_prompt(prompt, options)
    }

    #[cfg(test)]
    fn new_state(&self) -> DsV4State {
        self.engine.new_state()
    }

    #[cfg(test)]
    fn step(&self, state: &mut DsV4State, token: u32) -> Result<Vec<f32>> {
        self.engine.step(state, token)
    }
}

/// The shared per-token forward state machine, generic over the heavy-linear
/// provider. All small host math lives here; nothing in this struct touches
/// device memory.
#[derive(Debug)]
pub(crate) struct DsV4Engine<L: DsV4Linear> {
    linear: L,
    gguf: Arc<GgufFile>,
    config: QwenGgufConfig,
    tokenizer: GgufTokenizer,
    geometry: DsV4Geometry,
    layers: Vec<DsV4Layer>,
    output_norm: Vec<f32>,
    /// `output.weight`, or `token_embd.weight` when the lm head is tied.
    output_head: RawMatrix,
    token_embd: RawMatrix,
    hyper_head: HcWeights,
    rms_eps: f32,
    hc_eps: f32,
    /// Batched-prefill chunk size (`HI_DSV4_PREFILL_CHUNK`, resolved at load;
    /// 1 = legacy sequential prefill).
    prefill_chunk: usize,
    /// Reported as `QwenCpuRunOutput::backend` by the run entry points.
    backend: &'static str,
}

/// Model-wide dimensions resolved (and validated) from GGUF metadata once at
/// load time so the forward pass never re-derives them. Fields are
/// crate-visible for the GPU device-step provider (`dsv4_gpu`), which mirrors
/// this exact geometry on device. `Clone` lets the MTP drafter (`dsv4_mtp`)
/// carry its own copy — the MTP block shares the trunk's geometry exactly.
#[derive(Clone, Debug)]
pub(crate) struct DsV4Geometry {
    pub(crate) embed: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rope_dims: usize,
    pub(crate) q_lora: usize,
    pub(crate) o_groups: usize,
    pub(crate) o_rank: usize,
    /// Raw-attention sliding window (ring capacity); `None` = unbounded.
    pub(crate) window: Option<usize>,
    pub(crate) hc: usize,
    pub(crate) sinkhorn_iterations: usize,
    pub(crate) idx_heads: usize,
    pub(crate) idx_key: usize,
    pub(crate) idx_top_k: usize,
    pub(crate) experts: usize,
    pub(crate) moe_top_k: usize,
    pub(crate) expert_weights_norm: bool,
    pub(crate) expert_weights_scale: f32,
    pub(crate) vocab: usize,
    pub(crate) context: usize,
}

#[derive(Debug)]
pub(crate) struct DsV4Layer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) hc_attn: HcWeights,
    pub(crate) hc_ffn: HcWeights,
    pub(crate) q_a: RawMatrix,
    pub(crate) q_a_norm: Vec<f32>,
    pub(crate) q_b: RawMatrix,
    pub(crate) kv: RawMatrix,
    pub(crate) kv_norm: Vec<f32>,
    pub(crate) sinks: Option<Vec<f32>>,
    pub(crate) out_a: RawMatrix,
    pub(crate) out_b: RawMatrix,
    pub(crate) rope_base: f32,
    pub(crate) compressor: Option<CompressorWeights>,
    pub(crate) indexer: Option<IndexerWeights>,
    pub(crate) router: RawMatrix,
    pub(crate) probs_bias: Option<Vec<f32>>,
    /// Hash-routing lookup table (layers < hash_layer_count); raw I32, sliced
    /// per token id.
    pub(crate) tid2eid: Option<Tid2Eid>,
    pub(crate) gate_exps: RawExperts,
    pub(crate) up_exps: RawExperts,
    pub(crate) down_exps: RawExperts,
    pub(crate) shared: Option<SharedExpertWeights>,
    pub(crate) swiglu_clamp: f32,
}

/// Hyper-connection mixer (`hc_attn_*` / `hc_ffn_*` / `output_hc_*`): `func`
/// has hc²+2hc rows for block mixers (pre | post | comb) and hc rows for the
/// output head.
#[derive(Debug)]
pub(crate) struct HcWeights {
    pub(crate) func: DsV4HcFunc,
    pub(crate) base: Vec<f32>,
    pub(crate) scale: Vec<f32>,
}

/// Materialized f32 mixer matrix for [`HcWeights`]. A local replacement for
/// `qwen_cpu::Matrix` (same `mul_vec`/`shape`/`data` surface) so weights that
/// do not live in a GGUF — the MTP module's safetensors-sourced mixers — can
/// construct one from raw parts.
#[derive(Debug)]
pub(crate) struct DsV4HcFunc {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DsV4HcFunc {
    /// Build from a row-major `[rows, cols]` f32 payload.
    pub(crate) fn from_parts(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows * cols {
            bail!(
                "hc mixer payload has {} values; expected {rows} x {cols}",
                data.len()
            );
        }
        Ok(Self { rows, cols, data })
    }

    fn load(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let matrix = raw_matrix(gguf, name, rows, cols)?;
        let view = gguf
            .tensor(name)
            .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
        let data = dequantize_elem_range(&view, 0, matrix.rows * matrix.cols)?;
        Self::from_parts(rows, cols, data)
    }

    pub(crate) fn mul_vec(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.cols {
            bail!(
                "hc mixer input length {} does not match cols {}",
                input.len(),
                self.cols
            );
        }
        Ok(self
            .data
            .chunks_exact(self.cols)
            .map(|row| dot(row, input))
            .collect())
    }

    /// (rows, cols) of the materialized matrix (GPU device-step upload).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Row-major f32 payload (see [`Self::shape`]).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }
}

/// APE block compressor: `gate`/`kv` project embed -> `width`, `ape` is a
/// per-block-position additive bias laid out `[ratio][width]`, and `norm`
/// RMS-normalizes each `dim`-wide compressed half. `width` takes two forms in
/// the real GGUF: 2*dim (split K|V pair; ratio-4 layers and every indexer
/// compressor) or dim (a single shared latent serving as both K and V;
/// ratio-128 layers — mirroring the raw path's K=V design).
#[derive(Debug)]
pub(crate) struct CompressorWeights {
    pub(crate) gate: RawMatrix,
    pub(crate) kv: RawMatrix,
    pub(crate) ape: Vec<f32>,
    pub(crate) norm: Vec<f32>,
    pub(crate) ratio: usize,
    pub(crate) dim: usize,
    pub(crate) width: usize,
}

#[derive(Debug)]
pub(crate) struct IndexerWeights {
    pub(crate) q_b: RawMatrix,
    pub(crate) proj: RawMatrix,
    pub(crate) compressor: CompressorWeights,
}

#[derive(Debug)]
pub(crate) struct SharedExpertWeights {
    pub(crate) gate: RawMatrix,
    pub(crate) up: RawMatrix,
    pub(crate) down: RawMatrix,
}

/// Handle to an un-dequantized GGUF matrix; `rows` is the output dimension
/// (GGUF ne1) and `cols` the input dimension (GGUF ne0). The bytes stay in the
/// mmap and are dequantized (or uploaded) per provider policy.
#[derive(Clone, Debug)]
pub(crate) struct RawMatrix {
    pub(crate) name: String,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

/// Handle to a rank-3 packed expert tensor `[in, out, experts]`; expert `e`
/// occupies the contiguous element range `e*in*out .. (e+1)*in*out`.
#[derive(Debug)]
pub(crate) struct RawExperts {
    pub(crate) name: String,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

/// A MoE layer's three packed expert tensors, passed to the optional
/// [`DsV4Linear::prefetch_experts`] hook so a provider can prefetch the routed
/// experts' slices before the layer's expert matmuls. Only the native-cuda
/// provider reads the fields (the CPU default hook is a no-op).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DsV4ExpertTensors<'a> {
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) gate: &'a RawExperts,
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) up: &'a RawExperts,
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) down: &'a RawExperts,
}

/// Everything one layer's MoE block needs, bundled for
/// [`DsV4Linear::moe_block`]: tensor handles, routing tables/bias, the
/// per-layer SwiGLU clamp, and the routing scalars the engine resolved at
/// load. Built per call by the engine (cheap — all references).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DsV4MoeBlockCtx<'a> {
    pub(crate) router: &'a RawMatrix,
    /// `exp_probs_b.bias`: added to the sqrt-softplus scores for SELECTION
    /// only; mixture weights use the raw scores.
    pub(crate) probs_bias: Option<&'a [f32]>,
    /// Hash-routing table (layers < hash_layer_count); selection bypasses the
    /// scores entirely but the weights still come from them.
    pub(crate) tid2eid: Option<&'a Tid2Eid>,
    pub(crate) gate: &'a RawExperts,
    pub(crate) up: &'a RawExperts,
    pub(crate) down: &'a RawExperts,
    pub(crate) shared: Option<DsV4MoeShared<'a>>,
    /// Per-layer SwiGLU clamp: gate ceiled at +clamp, up clamped to ±clamp
    /// (<= 0 disables); the shared expert is always unclamped.
    pub(crate) swiglu_clamp: f32,
    pub(crate) experts: usize,
    pub(crate) top_k: usize,
    pub(crate) weights_norm: bool,
    pub(crate) weights_scale: f32,
    pub(crate) embed: usize,
}

/// The shared expert's three dense matrices (plain SwiGLU, no clamp).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DsV4MoeShared<'a> {
    pub(crate) gate: &'a RawMatrix,
    pub(crate) up: &'a RawMatrix,
    pub(crate) down: &'a RawMatrix,
}

#[derive(Debug)]
pub(crate) struct Tid2Eid {
    /// GGUF tensor name; the GPU provider keys its device copy of the table
    /// by it (unused by the host path once values are materialized).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) name: String,
    /// Experts per token (GGUF ne0, equals expert_used_count).
    pub(crate) stride: usize,
    /// Token rows in the table (GGUF ne1); lookups clamp to it like MLX.
    pub(crate) tokens: usize,
    /// The table itself, materialized (and range-validated) at load: row
    /// `t` selects experts `values[t*stride..(t+1)*stride]`. Tiny (~3 MB per
    /// hash layer on the real model); materializing lets the host path avoid
    /// per-token mmap reads and gives the GPU provider bytes to upload.
    pub(crate) values: Vec<i32>,
}

/// Per-run mutable state; reset for every prompt. Opaque outside the engine:
/// callers obtain one from [`DsV4Engine::new_state`] and advance it through
/// [`DsV4Engine::step`] / [`DsV4Engine::prefill`] (the serving backend drives
/// exactly that pair). `Clone` supports the serving backend's prefix-reuse
/// snapshot; all fields are host-side plain vectors.
#[derive(Clone)]
pub(crate) struct DsV4State {
    pub(crate) layers: Vec<DsV4LayerState>,
    pub(crate) pos: usize,
    /// Extra raw-ring entries retained beyond the attention window so the
    /// state can later be truncated back to an earlier position (prefix
    /// reuse). Attention always reads only the trailing `window` entries, so
    /// any slack value yields bit-identical logits; 0 (the default) keeps the
    /// original evict-at-window behavior.
    pub(crate) ring_slack: usize,
    /// Wave-2 Stage 2b: identity of the GPU-resident copy of this state.
    /// 0 = never device-mirrored. A device decode step whose provider-side
    /// device state carries the same (tag, pos) continues device-resident
    /// without re-uploading; a forward host mutation (host_step /
    /// step_chunk_heads, prefill, snapshot resume onto other content) zeroes
    /// the tag and forces a full host→device restore under a fresh tag.
    /// Host-side content is ALWAYS authoritative — the device step downloads
    /// its per-step state delta and replays it into these host vectors before
    /// returning, so cloning a state (prefix-cache snapshots) is valid at any
    /// time. The lineage invariant: any state carrying tag T at position p is
    /// bit-identical to the tag-T device trajectory's prefix at p. Clones
    /// share the tag (equal content); truncation keeps it (exact prefix
    /// reconstruction) and the provider then either ADOPTS the rewind
    /// device-side — re-tagging both sides fresh, so a divergent sibling
    /// clone at the old (tag, pos) can never false-match — or restores.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) device_tag: u64,
}

impl DsV4State {
    /// Number of tokens this state has processed.
    // Consumed by the native-cuda serving backend (and tests).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// Approximate host-heap footprint of this snapshot in bytes: the raw-KV
    /// ring latents plus every compressor/indexer's pending + compressed
    /// key/value latents (f32 payload bytes, the dominant term). Consumed by
    /// the serving backend's block-hash prefix cache to bound its LRU budget.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn snapshot_bytes(&self) -> usize {
        const F32: usize = std::mem::size_of::<f32>();
        self.layers
            .iter()
            .map(|layer| {
                let ring: usize = layer.ring.iter().map(|latent| latent.len() * F32).sum();
                let compressor = layer
                    .compressor
                    .as_ref()
                    .map_or(0, CompressorState::payload_bytes);
                let indexer = layer
                    .indexer
                    .as_ref()
                    .map_or(0, CompressorState::payload_bytes);
                ring + compressor + indexer
            })
            .sum()
    }
}

#[derive(Clone)]
pub(crate) struct DsV4LayerState {
    /// Raw KV ring: rope'd shared latents of the last `window + ring_slack`
    /// positions (attention reads only the trailing `window`).
    pub(crate) ring: VecDeque<Vec<f32>>,
    pub(crate) compressor: Option<CompressorState>,
    pub(crate) indexer: Option<CompressorState>,
}

#[derive(Clone)]
pub(crate) struct CompressorState {
    /// Tokens buffered until a full block of `ratio` accumulates.
    pub(crate) pending: Vec<Vec<f32>>,
    pub(crate) keys: Vec<Vec<f32>>,
    pub(crate) values: Vec<Vec<f32>>,
}

/// One decode step's device→host state delta (Wave-2 Stage 2b): exactly what
/// a host step would have appended to [`DsV4State`], computed on device and
/// downloaded once at the end of the step. Slices borrow the downloaded
/// arena; [`DsV4Engine::apply_device_step_mirror`] replays them so the host
/// mirror stays authoritative after every device step (prefix-cache snapshots
/// can clone the state at any boundary).
#[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
pub(crate) struct DsV4StepMirror<'a> {
    /// Per layer: the rope'd shared KV latent pushed onto the raw ring.
    pub(crate) kv: Vec<&'a [f32]>,
    /// Per layer: the attention input (post attn-norm) the compressor and
    /// indexer pending lists buffer; `None` for compressor-less layers.
    pub(crate) x: Vec<Option<&'a [f32]>>,
    /// Per layer: the compressed (key, value) block the layer's compressor
    /// emitted this step, when a block completed.
    pub(crate) comp_block: Vec<Option<(&'a [f32], &'a [f32])>>,
    /// Same for the indexer's private compressor.
    pub(crate) idx_block: Vec<Option<(&'a [f32], &'a [f32])>>,
}

/// Which positions of a chunked forward pass get the output head (hyper-head
/// stream collapse + final norm + lm head) applied.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChunkHeads {
    /// No logits (interior prefill chunks).
    None,
    /// Last position only (the prefill tail).
    Last,
    /// Every position (speculative verify).
    All,
}

/// RAII wrapper around [`DsV4Linear::set_exact_batching`] so an early `?`
/// return inside a verify chunk cannot leave the provider pinned to exact
/// batching.
struct ExactBatchingGuard<'a, L: DsV4Linear>(&'a L);

impl<'a, L: DsV4Linear> ExactBatchingGuard<'a, L> {
    fn new(linear: &'a L) -> Self {
        linear.set_exact_batching(true);
        Self(linear)
    }
}

impl<L: DsV4Linear> Drop for ExactBatchingGuard<'_, L> {
    fn drop(&mut self) {
        self.0.set_exact_batching(false);
    }
}

/// Which hidden activations a [`DsV4Taps`] buffer captures per position. An
/// empty config captures nothing; every forward entry point takes
/// `Option<&mut DsV4Taps>` and `None` skips all capture work, so taps are
/// zero-cost unless a drafter asks for them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
pub(crate) struct DsV4TapConfig {
    /// Capture the flat (hc*embed) residual immediately before the output
    /// hyper-head + final norm + lm head are applied — the MTP drafter's
    /// `prev_hidden` input (Stage B of the spec-decode plan).
    pub(crate) pre_hc_head: bool,
    /// Capture post-layer hc-stream hidden states at these layer indices —
    /// the DFlash drafter's aux-hidden conditioning (Stage C). Normalized
    /// (sorted, deduplicated, validated) by [`DsV4Engine::new_taps`].
    pub(crate) aux_layers: Vec<usize>,
}

impl DsV4TapConfig {
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn is_empty(&self) -> bool {
        !self.pre_hc_head && self.aux_layers.is_empty()
    }
}

/// Per-request hidden-state capture buffer for speculative drafters. Rows are
/// appended by every tapped forward path — prefill chunks, verify chunks, and
/// single (host) steps — for consecutive ABSOLUTE positions starting at the
/// buffer's `base` (the state position the buffer was attached at — a
/// prefix-cache restore point skips the restored prefix, whose activations
/// were never recomputed). Captured positions stay in lockstep with the
/// state's processed tokens as long as the buffer is attached at the state's
/// current position and truncated alongside every state rewind
/// ([`DsV4Taps::truncate`] drops rejected-draft positions exactly like
/// `truncate_state_to_at_most` drops their state). Every accessor takes
/// absolute positions and returns `None` below `base` — never a misaligned
/// row. Flat rows are the concatenated hc streams
/// `[stream_0 | .. | stream_{hc-1}]` (hc*embed wide); the stream-averaged
/// view is their arithmetic mean (embed wide).
// Accessor surface is consumed by the tap tests today and the Stage B/C
// drafters next; production code only drives capture + truncate.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct DsV4Taps {
    config: DsV4TapConfig,
    hc: usize,
    embed: usize,
    /// Absolute position of the first captured row (0 unless the request
    /// resumed from a prefix-cache restore).
    base: usize,
    /// Rows captured so far (advances even for an empty config so the
    /// lockstep invariant is checkable).
    captured: usize,
    /// One flat (hc*embed) row per position `base + i`, when
    /// `config.pre_hc_head`.
    pre_hc_head: Vec<Vec<f32>>,
    /// `aux[g]` holds one flat row per position `base + i` for
    /// `config.aux_layers[g]`.
    aux: Vec<Vec<Vec<f32>>>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DsV4Taps {
    pub(crate) fn config(&self) -> &DsV4TapConfig {
        &self.config
    }

    /// Absolute position of the first captured row; accessors return `None`
    /// below it. 0 unless the request resumed from a prefix-cache restore.
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    /// Absolute END of the captured range (`base + captured rows`); equals
    /// the attached state's processed-token count when the buffer has been
    /// attached (and truncated) consistently. Rows exist for positions
    /// `base() .. positions()`.
    pub(crate) fn positions(&self) -> usize {
        self.base + self.captured
    }

    /// Width of every flat row (hc * embed).
    pub(crate) fn flat_width(&self) -> usize {
        self.hc * self.embed
    }

    /// The flat pre-hc-head residual captured at ABSOLUTE `position`
    /// (`None` below the base).
    pub(crate) fn pre_hc_head(&self, position: usize) -> Option<&[f32]> {
        self.pre_hc_head
            .get(position.checked_sub(self.base)?)
            .map(Vec::as_slice)
    }

    /// The flat post-layer streams captured at ABSOLUTE `position` for
    /// `layer` (`None` below the base).
    pub(crate) fn aux_flat(&self, layer: usize, position: usize) -> Option<&[f32]> {
        let group = self.config.aux_layers.binary_search(&layer).ok()?;
        self.aux[group]
            .get(position.checked_sub(self.base)?)
            .map(Vec::as_slice)
    }

    /// Stream-averaged (embed-wide arithmetic mean over the hc streams) view
    /// of [`Self::aux_flat`].
    pub(crate) fn aux_averaged(&self, layer: usize, position: usize) -> Option<Vec<f32>> {
        let flat = self.aux_flat(layer, position)?;
        let mut avg = vec![0.0f32; self.embed];
        for stream in flat.chunks(self.embed) {
            for (out, value) in avg.iter_mut().zip(stream) {
                *out += value;
            }
        }
        let inv = (self.hc as f32).recip();
        for value in &mut avg {
            *value *= inv;
        }
        Some(avg)
    }

    /// Drop every captured row at or beyond ABSOLUTE `positions` (state-rewind
    /// mirror). Truncating to the base (or below it) empties the buffer but
    /// keeps the base — a rewind below the base must instead [`Self::rebase`]
    /// so re-captured rows realign.
    pub(crate) fn truncate(&mut self, positions: usize) {
        let keep = positions.saturating_sub(self.base).min(self.captured);
        self.captured = keep;
        self.pre_hc_head.truncate(keep);
        for group in &mut self.aux {
            group.truncate(keep);
        }
    }

    /// Empty the buffer and restart capture at ABSOLUTE `base` (a state
    /// rewind that landed below the current base, or a full state rebuild
    /// from position 0).
    pub(crate) fn rebase(&mut self, base: usize) {
        self.base = base;
        self.captured = 0;
        self.pre_hc_head.clear();
        for group in &mut self.aux {
            group.clear();
        }
    }

    pub(crate) fn wants_layer(&self, layer: usize) -> bool {
        self.config.aux_layers.binary_search(&layer).is_ok()
    }

    fn push_layer_row(&mut self, layer: usize, streams: &[Vec<f32>]) {
        if let Ok(group) = self.config.aux_layers.binary_search(&layer) {
            self.aux[group].push(flat_streams(streams));
        }
    }

    fn push_pre_head_row(&mut self, streams: &[Vec<f32>]) {
        if self.config.pre_hc_head {
            self.pre_hc_head.push(flat_streams(streams));
        }
    }

    /// [`Self::push_layer_row`] from an already-flat (hc*embed) row — the GPU
    /// device-verify path captures rows as flat device copies, so it feeds
    /// them back without a streams round-trip. Identical layout and order.
    pub(crate) fn push_layer_row_flat(&mut self, layer: usize, flat: &[f32]) {
        if let Ok(group) = self.config.aux_layers.binary_search(&layer) {
            self.aux[group].push(flat.to_vec());
        }
    }

    /// [`Self::push_pre_head_row`] from an already-flat row (see
    /// [`Self::push_layer_row_flat`]).
    pub(crate) fn push_pre_head_row_flat(&mut self, flat: &[f32]) {
        if self.config.pre_hc_head {
            self.pre_hc_head.push(flat.to_vec());
        }
    }

    /// Advance the captured-row counter after a forward pass captured `n` new
    /// positions into every enabled group.
    pub(crate) fn note_positions(&mut self, n: usize) {
        self.captured += n;
        debug_assert!(!self.config.pre_hc_head || self.pre_hc_head.len() == self.captured);
        debug_assert!(self.aux.iter().all(|group| group.len() == self.captured));
    }
}

/// Concatenate the hc streams into one flat row (the layout `hc_pre_math`
/// flattens to, and the layout the MTP module consumes).
fn flat_streams(streams: &[Vec<f32>]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(streams.iter().map(Vec::len).sum());
    for stream in streams {
        flat.extend_from_slice(stream);
    }
    flat
}

impl<L: DsV4Linear> DsV4Engine<L> {
    /// Load the shared engine from an already-open GGUF; `linear` serves the
    /// heavy ops and `backend` labels run outputs.
    pub(crate) fn new(gguf: Arc<GgufFile>, linear: L, backend: &'static str) -> Result<Self> {
        let config = gguf.qwen_config()?;
        if !config.is_deepseek4() {
            bail!(
                "DeepSeek-V4 engine requires a deepseek4 GGUF, got architecture '{}'",
                config.architecture
            );
        }
        let tokenizer = gguf.tokenizer()?;
        let rms_eps = config.rms_norm_eps.unwrap_or(1.0e-6);
        let hc_eps = config.hyper_connection_epsilon.unwrap_or(1.0e-6);

        let embed = usize::try_from(config.embedding_length)
            .context("deepseek4 embedding_length does not fit usize")?;
        let heads = require_dim(Some(config.attention_head_count), "attention.head_count")?;
        let head_dim = require_dim(config.attention_key_length, "attention.key_length")?;
        if let Some(value_length) = config.attention_value_length
            && value_length as usize != head_dim
        {
            bail!(
                "deepseek4 attention.value_length {value_length} must equal attention.key_length {head_dim} (shared KV latent)"
            );
        }
        let rope_dims = config.rope_dimension_count.unwrap_or(0) as usize;
        if rope_dims > head_dim || !rope_dims.is_multiple_of(2) {
            bail!(
                "deepseek4 rope.dimension_count {rope_dims} must be even and <= head dim {head_dim}"
            );
        }
        let q_lora = require_dim(config.attention_q_lora_rank, "attention.q_lora_rank")?;
        let o_groups = config.attention_output_group_count.unwrap_or(1) as usize;
        let o_rank = require_dim(
            config.attention_output_lora_rank,
            "attention.output_lora_rank",
        )?;
        if o_groups == 0 || !(heads * head_dim).is_multiple_of(o_groups) {
            bail!(
                "deepseek4 attention.output_group_count {o_groups} must divide heads*head_dim {}",
                heads * head_dim
            );
        }
        let window = match config.attention_sliding_window {
            Some(0) => bail!("deepseek4 attention.sliding_window must be non-zero"),
            Some(window) => Some(window as usize),
            None => None,
        };
        let hc = require_dim(config.hyper_connection_count, "hyper_connection.count")?;
        let sinkhorn_iterations = config
            .hyper_connection_sinkhorn_iterations
            .ok_or_else(|| anyhow!("deepseek4 GGUF missing hyper_connection.sinkhorn_iterations"))?
            as usize;
        let experts = require_dim(config.expert_count, "expert_count")?;
        let moe_top_k = require_dim(config.expert_used_count, "expert_used_count")?;
        if moe_top_k > experts {
            bail!("deepseek4 expert_used_count {moe_top_k} exceeds expert_count {experts}");
        }
        let expert_ff = require_dim(
            config.expert_feed_forward_length,
            "expert_feed_forward_length",
        )?;
        if let Some(gating) = config.expert_gating_func
            && gating != 4
        {
            bail!(
                "deepseek4 CPU reference supports expert_gating_func 4 (sqrt-softplus) only, got {gating}"
            );
        }
        let hash_layers = config.hash_layer_count.unwrap_or(0) as usize;
        let ratios: Vec<usize> = (0..config.block_count as usize)
            .map(|layer| {
                config
                    .attention_compress_ratios
                    .as_ref()
                    .and_then(|ratios| ratios.get(layer))
                    .copied()
                    .unwrap_or(0) as usize
            })
            .collect();
        let rope_base = config
            .rope_freq_base
            .unwrap_or_else(|| config.default_rope_freq_base());
        let compress_rope_base = if ratios.iter().any(|ratio| *ratio > 0) {
            config.attention_compress_rope_freq_base.ok_or_else(|| {
                anyhow!("deepseek4 GGUF missing attention.compress_rope_freq_base")
            })?
        } else {
            rope_base
        };
        // The lightning indexer exists exactly on ratio-4 layers (MLX reference
        // behavior; matches the real GGUF's tensor census).
        let (idx_heads, idx_key, idx_top_k) = if ratios.contains(&4) {
            (
                require_dim(
                    config.attention_indexer_head_count,
                    "attention.indexer.head_count",
                )?,
                require_dim(
                    config.attention_indexer_key_length,
                    "attention.indexer.key_length",
                )?,
                require_dim(config.attention_indexer_top_k, "attention.indexer.top_k")?,
            )
        } else {
            (0, 0, 0)
        };
        let vocab = config
            .vocab_size
            .map(usize::try_from)
            .transpose()
            .context("deepseek4 vocab size does not fit usize")?
            .unwrap_or_else(|| tokenizer.token_count());
        if vocab != tokenizer.token_count() {
            bail!(
                "deepseek4 vocab size {vocab} does not match tokenizer size {}",
                tokenizer.token_count()
            );
        }
        let context = usize::try_from(config.context_length)
            .context("deepseek4 context_length does not fit usize")?;

        let geometry = DsV4Geometry {
            embed,
            heads,
            head_dim,
            rope_dims,
            q_lora,
            o_groups,
            o_rank,
            window,
            hc,
            sinkhorn_iterations,
            idx_heads,
            idx_key,
            idx_top_k,
            experts,
            moe_top_k,
            expert_weights_norm: config.expert_weights_norm,
            expert_weights_scale: config.expert_weights_scale.unwrap_or(1.0),
            vocab,
            context,
        };

        let output_norm = load_vector(&gguf, "output_norm.weight", embed)?;
        let token_embd = raw_matrix(&gguf, "token_embd.weight", vocab, embed)?;
        let output_head = if gguf.tensor("output.weight").is_some() {
            raw_matrix(&gguf, "output.weight", vocab, embed)?
        } else {
            token_embd.clone()
        };
        let hyper_head = HcWeights::load(&gguf, "output_hc", hc, hc * embed, 1)?;

        let shared_ff = expert_ff * config.expert_shared_count.unwrap_or(0) as usize;
        let mut layers = Vec::with_capacity(config.block_count as usize);
        for idx in 0..config.block_count as usize {
            layers.push(DsV4Layer::load(
                &gguf,
                &config,
                &geometry,
                idx,
                ratios[idx],
                hash_layers,
                expert_ff,
                shared_ff,
                rope_base,
                compress_rope_base,
            )?);
        }

        Ok(Self {
            linear,
            gguf,
            config,
            tokenizer,
            geometry,
            layers,
            output_norm,
            output_head,
            token_embd,
            hyper_head,
            rms_eps,
            hc_eps,
            prefill_chunk: prefill_chunk_from_env(),
            backend,
        })
    }

    pub(crate) fn config(&self) -> &QwenGgufConfig {
        &self.config
    }

    pub(crate) fn tokenizer(&self) -> &GgufTokenizer {
        &self.tokenizer
    }

    // Only the native-cuda `dsv4_gpu` provider consumes these two; keep the
    // non-cuda build warning-free without cfg'ing the engine API itself.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn linear(&self) -> &L {
        &self.linear
    }

    // ---- Wave-2 Stage 2b device-step seams -------------------------------
    // The GPU provider mirrors the engine's small host weights and per-token
    // state machine on device; these read-only accessors expose exactly what
    // it uploads/orchestrates. All are inert for the CPU oracle.

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn geometry(&self) -> &DsV4Geometry {
        &self.geometry
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn layers(&self) -> &[DsV4Layer] {
        &self.layers
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn output_norm_weights(&self) -> &[f32] {
        &self.output_norm
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn hyper_head_weights(&self) -> &HcWeights {
        &self.hyper_head
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn output_head_matrix(&self) -> &RawMatrix {
        &self.output_head
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn token_embd_matrix(&self) -> &RawMatrix {
        &self.token_embd
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn rms_eps(&self) -> f32 {
        self.rms_eps
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn hc_eps(&self) -> f32 {
        self.hc_eps
    }

    /// Every non-expert heavy matrix the engine will route through the
    /// provider, paired with the grouped rank for block-diagonal ones. A
    /// GPU-resident provider must upload exactly this set: every
    /// [`TensorKey::Dense`]/[`TensorKey::Grouped`] the forward pass can issue
    /// comes from it, and packed experts are deliberately excluded (streamed
    /// per [`TensorKey::Expert`] slice instead).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn resident_matrices(&self) -> Vec<(RawMatrix, Option<usize>)> {
        let mut matrices = vec![(self.output_head.clone(), None)];
        for layer in &self.layers {
            matrices.push((layer.q_a.clone(), None));
            matrices.push((layer.q_b.clone(), None));
            matrices.push((layer.kv.clone(), None));
            matrices.push((layer.out_a.clone(), Some(self.geometry.o_rank)));
            matrices.push((layer.out_b.clone(), None));
            matrices.push((layer.router.clone(), None));
            if let Some(compressor) = &layer.compressor {
                matrices.push((compressor.gate.clone(), None));
                matrices.push((compressor.kv.clone(), None));
            }
            if let Some(indexer) = &layer.indexer {
                matrices.push((indexer.q_b.clone(), None));
                matrices.push((indexer.proj.clone(), None));
                matrices.push((indexer.compressor.gate.clone(), None));
                matrices.push((indexer.compressor.kv.clone(), None));
            }
            if let Some(shared) = &layer.shared {
                matrices.push((shared.gate.clone(), None));
                matrices.push((shared.up.clone(), None));
                matrices.push((shared.down.clone(), None));
            }
        }
        matrices
    }

    /// Logits after the last input token, from a fresh per-run state.
    pub(crate) fn last_logits(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let mut state = self.new_state();
        self.feed_tokens(&mut state, input_ids)
    }

    pub(crate) fn run_tokens(
        &self,
        input_ids: &[u32],
        options: QwenCpuRunOptions,
    ) -> Result<QwenCpuRunOutput> {
        let mut state = self.new_state();
        let logits = self.feed_tokens(&mut state, input_ids)?;
        let next_token = argmax(&logits)?;
        // Generation continues from the prompt state; each streaming step is
        // numerically identical to qwen_cpu's from-scratch re-run of the grown
        // token sequence, so the output contract matches exactly.
        let generated_tokens = if let Some(seed) = options.seed {
            let mut rng = StdRng::seed_from_u64(seed);
            self.generate_from_state(&mut state, &logits, &options, &mut rng)?
        } else {
            let mut rng = rand::thread_rng();
            self.generate_from_state(&mut state, &logits, &options, &mut rng)?
        };
        let next_text = self.tokenizer.decode(&[next_token])?;
        let generated_text = self.tokenizer.decode(&generated_tokens)?;
        let top_logits = top_logits(&logits, &self.tokenizer, options.top_k)?;
        let logit_count = logits.len();
        Ok(QwenCpuRunOutput {
            backend: self.backend,
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

    pub(crate) fn run_prompt(
        &self,
        prompt: &str,
        options: QwenCpuRunOptions,
    ) -> Result<QwenCpuRunOutput> {
        let tokens = self.tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            bail!("prompt encoded to zero tokens");
        }
        self.run_tokens(&tokens, options)
    }

    pub(crate) fn new_state(&self) -> DsV4State {
        self.new_state_with_ring_slack(0)
    }

    /// A fresh state whose raw rings retain `ring_slack` extra entries beyond
    /// the attention window, making later [`Self::truncate_state_to_at_most`]
    /// calls able to rewind up to that many tokens. Logits are bit-identical
    /// for any slack (attention reads only the trailing window).
    pub(crate) fn new_state_with_ring_slack(&self, ring_slack: usize) -> DsV4State {
        DsV4State {
            layers: self
                .layers
                .iter()
                .map(|layer| DsV4LayerState {
                    ring: VecDeque::new(),
                    compressor: layer.compressor.as_ref().map(|_| CompressorState::new()),
                    indexer: layer.indexer.as_ref().map(|_| CompressorState::new()),
                })
                .collect(),
            pos: 0,
            ring_slack,
            device_tag: 0,
        }
    }

    fn feed_tokens(&self, state: &mut DsV4State, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.prefill(state, input_ids)
    }

    /// The engine's configured prefill chunk size (`HI_DSV4_PREFILL_CHUNK`).
    // Consumed by the native-cuda serving backend.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn prefill_chunk_size(&self) -> usize {
        self.prefill_chunk
    }

    /// Prefill `tokens` with the engine's configured chunk size, returning the
    /// logits after the last token.
    pub(crate) fn prefill(&self, state: &mut DsV4State, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_with_chunk(state, tokens, self.prefill_chunk)
    }

    /// [`Self::prefill`] with optional hidden-state capture (a speculative
    /// drafter's context; see [`DsV4Taps`]). `None` is exactly `prefill`.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn prefill_with_taps(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        taps: Option<&mut DsV4Taps>,
    ) -> Result<Vec<f32>> {
        self.prefill_with_chunk_taps(state, tokens, self.prefill_chunk, taps)
    }

    /// Prefill `tokens` processing up to `chunk` tokens per pass. `chunk <= 1`
    /// is the legacy strictly-sequential path (one [`Self::step`] per token);
    /// larger chunks batch the heavy linears and parallelize the host math
    /// while producing results identical to the sequential path (the batched
    /// parity tests gate exactly that; the GPU provider's opt-in
    /// `HI_DSV4_PREFILL_GEMM=1` mode trades that exactness for speed — see
    /// `dsv4_gpu`).
    pub(crate) fn prefill_with_chunk(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<Vec<f32>> {
        self.prefill_with_chunk_taps(state, tokens, chunk, None)
    }

    /// [`Self::prefill_with_chunk`] with optional hidden-state capture. Tap
    /// capture on the `chunk <= 1` path forces host steps (see
    /// [`Self::step_with_taps`]); logits are identical either way.
    pub(crate) fn prefill_with_chunk_taps(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        chunk: usize,
        mut taps: Option<&mut DsV4Taps>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            bail!("DeepSeek-V4 engine requires at least one input token");
        }
        if chunk <= 1 {
            let mut logits = Vec::new();
            for &token in tokens {
                logits = self.step_with_taps(state, token, taps.as_deref_mut())?;
            }
            return Ok(logits);
        }
        let mut logits = None;
        let mut offset = 0;
        while offset < tokens.len() {
            let take = chunk.min(tokens.len() - offset);
            let heads = if offset + take == tokens.len() {
                ChunkHeads::Last
            } else {
                ChunkHeads::None
            };
            logits = self
                .step_chunk_heads(
                    state,
                    &tokens[offset..offset + take],
                    heads,
                    taps.as_deref_mut(),
                )?
                .pop();
            offset += take;
        }
        logits.ok_or_else(|| anyhow!("chunked prefill produced no logits"))
    }

    /// Forward a chunk of already-sampled/drafted tokens and return the vocab
    /// logits at EVERY position — the speculative-decoding verify step
    /// (`docs/deepseek-v4-spec-decode-plan.md` Stage A). Bit-exact with
    /// running [`Self::host_step`] over the same tokens: the chunk machinery
    /// is bit-exact with the sequential path by construction (including the
    /// GPU provider's device MoE + prefill expert pool), the per-position
    /// output head below runs the sequential step's exact math, and
    /// [`DsV4Linear::set_exact_batching`] pins the provider's `mul_mat` to
    /// per-token loops for the duration (suspending `HI_DSV4_PREFILL_GEMM=1`,
    /// which stays prefill-only). Advances `state` over ALL `tokens` — the
    /// caller rolls rejected positions back via [`Self::rewind_state_to`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn verify_tokens(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        self.verify_tokens_with_taps(state, tokens, None)
    }

    /// [`Self::verify_tokens`] with optional hidden-state capture.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn verify_tokens_with_taps(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        mut taps: Option<&mut DsV4Taps>,
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            bail!("DeepSeek-V4 verify requires at least one token");
        }
        let _exact = ExactBatchingGuard::new(&self.linear);
        let mut rows = Vec::with_capacity(tokens.len());
        // Wave-3 device-sequenced verify: serve the call in device chunks of
        // up to DSV4_DEVICE_VERIFY_CAP tokens (bit-exact with the host path
        // below by contract). Support is static per provider/env/tap-config,
        // so the first decline routes the remainder through the host chunk
        // path; chunking is semantically invisible either way (the state
        // machine is sequential across chunk boundaries).
        let mut offset = 0;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(DSV4_DEVICE_VERIFY_CAP);
            match self.linear.try_device_verify(
                self,
                state,
                &tokens[offset..offset + take],
                taps.as_deref_mut(),
                true,
            ) {
                Some(result) => {
                    rows.extend(result?);
                    offset += take;
                }
                None => break,
            }
        }
        let chunk = self.prefill_chunk.max(1);
        for piece in tokens[offset..].chunks(chunk) {
            rows.extend(self.step_chunk_heads(
                state,
                piece,
                ChunkHeads::All,
                taps.as_deref_mut(),
            )?);
        }
        Ok(rows)
    }

    /// Rewind `state` to exactly `target` processed tokens after a verify
    /// overshoot, given `tokens` — the full token history the state was fed
    /// (`tokens.len() >= target`). [`Self::truncate_state_to_at_most`] may
    /// round down to a compressor block boundary (mid-block interiors are
    /// unrecoverable once compressed); the rounded-off suffix is then re-fed
    /// through the exact chunk path, which reproduces the sequential state
    /// bit for bit. A state whose retention cannot reach any rewind point
    /// (`None` — e.g. resumed from a snapshot taken without speculative ring
    /// slack) rebuilds from scratch: rare, slow, never wrong. `taps` (when
    /// attached from position 0) is truncated and re-captured in lockstep.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn rewind_state_to(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        target: usize,
        mut taps: Option<&mut DsV4Taps>,
    ) -> Result<()> {
        if target == 0 || target > state.pos || target > tokens.len() {
            bail!(
                "cannot rewind a state at position {} over {} known tokens to {target}",
                state.pos,
                tokens.len()
            );
        }
        // Re-fed suffixes must reproduce the sequential state exactly, so the
        // provider stays pinned to exact batching here too (see verify_tokens).
        let _exact = ExactBatchingGuard::new(&self.linear);
        if state.pos > target {
            match self.truncate_state_to_at_most(state, target) {
                Some(position) if position == target => {}
                Some(position) => {
                    if let Some(taps) = taps.as_deref_mut() {
                        // A round-down below the taps base restarts capture
                        // there so the re-fed rows realign (rows can never
                        // exist below a buffer's base).
                        if position < taps.base() {
                            taps.rebase(position);
                        } else {
                            taps.truncate(position);
                        }
                    }
                    self.refeed_chunks(state, &tokens[position..target], taps.as_deref_mut())?;
                }
                None => {
                    *state = self.new_state_with_ring_slack(state.ring_slack);
                    if let Some(taps) = taps.as_deref_mut() {
                        // The rebuild recomputes every position from 0, so the
                        // buffer re-captures from 0 whatever its old base was.
                        taps.rebase(0);
                    }
                    self.refeed_chunks(state, &tokens[..target], taps.as_deref_mut())?;
                }
            }
        }
        if let Some(taps) = taps {
            taps.truncate(target);
        }
        debug_assert_eq!(state.pos, target);
        Ok(())
    }

    /// Feed a rewind's rounded-off suffix back through the exact chunk
    /// machinery, headless. The device chunk path serves it when available
    /// (keeping the whole rewind device-resident — no restore on the next
    /// device step); the first decline falls to the host chunk path, exactly
    /// the pre-Wave-3 behavior. The caller holds the exact-batching guard.
    fn refeed_chunks(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        mut taps: Option<&mut DsV4Taps>,
    ) -> Result<()> {
        let mut offset = 0;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(DSV4_DEVICE_VERIFY_CAP);
            match self.linear.try_device_verify(
                self,
                state,
                &tokens[offset..offset + take],
                taps.as_deref_mut(),
                false,
            ) {
                Some(result) => {
                    result?;
                    offset += take;
                }
                None => break,
            }
        }
        let chunk = self.prefill_chunk.max(1);
        for piece in tokens[offset..].chunks(chunk) {
            self.step_chunk_heads(state, piece, ChunkHeads::None, taps.as_deref_mut())?;
        }
        Ok(())
    }

    /// Build a hidden-state capture buffer for this engine's geometry,
    /// normalizing (sort + dedup) and validating the aux layer indices.
    // Production attaches at a restore base via new_taps_at; the base-0 form
    // serves the tap suites.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new_taps(&self, config: DsV4TapConfig) -> Result<DsV4Taps> {
        self.new_taps_at(config, 0)
    }

    /// [`Self::new_taps`] attached at ABSOLUTE position `base` — the state
    /// position the buffer starts capturing from. A request resuming from a
    /// prefix-cache restore passes the restore point: the restored prefix's
    /// activations were never recomputed, so no rows exist below it and the
    /// accessors return `None` there (drafters cold-start at the base).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn new_taps_at(&self, config: DsV4TapConfig, base: usize) -> Result<DsV4Taps> {
        let mut aux_layers = config.aux_layers;
        aux_layers.sort_unstable();
        aux_layers.dedup();
        if let Some(&layer) = aux_layers.last()
            && layer >= self.layers.len()
        {
            bail!(
                "tap layer index {layer} is outside the model's {} layers",
                self.layers.len()
            );
        }
        let groups = aux_layers.len();
        Ok(DsV4Taps {
            config: DsV4TapConfig {
                pre_hc_head: config.pre_hc_head,
                aux_layers,
            },
            hc: self.geometry.hc,
            embed: self.geometry.embed,
            base,
            captured: 0,
            pre_hc_head: Vec::new(),
            aux: vec![Vec::new(); groups],
        })
    }

    fn generate_from_state<R: Rng + ?Sized>(
        &self,
        state: &mut DsV4State,
        prompt_logits: &[f32],
        options: &QwenCpuRunOptions,
        rng: &mut R,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut logits = prompt_logits.to_vec();
        for step in 0..options.max_tokens {
            let next = sample_from_logits_with_rng(
                &logits,
                options.temperature,
                options.top_p,
                None,
                rng,
            )?;
            generated.push(next);
            if Some(next) == self.config.eos_token_id {
                break;
            }
            if step + 1 < options.max_tokens {
                logits = self.step(state, next)?;
            }
        }
        Ok(generated)
    }

    /// Process one token (S=1) at the state's current position and return the
    /// vocab logits. The provider may serve the whole step device-side
    /// ([`DsV4Linear::try_device_step`], Wave-2 Stage 2b); a decline runs the
    /// host step below — today's exact path, byte for byte.
    pub(crate) fn step(&self, state: &mut DsV4State, token: u32) -> Result<Vec<f32>> {
        if let Some(result) = self.linear.try_device_step(self, state, token) {
            return result;
        }
        self.host_step(state, token)
    }

    /// [`Self::step`] with optional hidden-state capture. Capture forces the
    /// host step (the hc streams live in host memory there; the device step
    /// keeps them GPU-resident) — the two are bit-identical, so attaching
    /// taps never changes logits, and `None` is exactly `step`.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn step_with_taps(
        &self,
        state: &mut DsV4State,
        token: u32,
        taps: Option<&mut DsV4Taps>,
    ) -> Result<Vec<f32>> {
        match taps {
            Some(taps) if !taps.config.is_empty() => {
                self.host_step_tapped(state, token, Some(taps))
            }
            Some(taps) => {
                // Empty config: nothing to capture, but keep the position
                // counter in lockstep with the state like the chunk path does.
                let logits = self.step(state, token)?;
                taps.note_positions(1);
                Ok(logits)
            }
            None => self.step(state, token),
        }
    }

    /// The host-orchestrated step (pre-Stage-2b behavior, unchanged): all
    /// hyper-connection/rope/attention/compressor/indexer math on host, heavy
    /// linears through the provider. Kept fully selectable as the device
    /// step's bisection fallback (`HI_DSV4_HOST_STEP=1`).
    pub(crate) fn host_step(&self, state: &mut DsV4State, token: u32) -> Result<Vec<f32>> {
        self.host_step_tapped(state, token, None)
    }

    /// [`Self::host_step`] plus optional per-position hidden-state capture
    /// (post-layer streams and the pre-hc-head residual; see [`DsV4Taps`]).
    fn host_step_tapped(
        &self,
        state: &mut DsV4State,
        token: u32,
        mut taps: Option<&mut DsV4Taps>,
    ) -> Result<Vec<f32>> {
        let pos = state.pos;
        if pos >= self.geometry.context {
            bail!(
                "input length {} exceeds deepseek4 context length {}",
                pos + 1,
                self.geometry.context
            );
        }
        // A host-side mutation diverges this state from any device-resident
        // trajectory sharing its tag: sever the link so a later device step
        // restores from the mirror instead of trusting a (tag, pos)
        // coincidence (reachable via snapshot resume + prefill).
        state.device_tag = 0;
        let hidden = self.embed_row(token)?;
        let ring_slack = state.ring_slack;
        // Broadcast the embedding into the hc residual streams.
        let mut streams = vec![hidden; self.geometry.hc];
        for (idx, (layer, layer_state)) in self.layers.iter().zip(&mut state.layers).enumerate() {
            let residual = streams.clone();
            let (mut y, post, comb) = self.hc_pre(&layer.hc_attn, &streams)?;
            rms_norm_in_place(&mut y, &layer.attn_norm, self.rms_eps)?;
            let attn = self.attention(layer, layer_state, &y, pos, ring_slack)?;
            streams = hc_post(&attn, &residual, &post, &comb);

            let residual = streams.clone();
            let (mut y, post, comb) = self.hc_pre(&layer.hc_ffn, &streams)?;
            rms_norm_in_place(&mut y, &layer.ffn_norm, self.rms_eps)?;
            let ffn = self.moe(layer, &y, token)?;
            streams = hc_post(&ffn, &residual, &post, &comb);
            if let Some(taps) = taps.as_deref_mut() {
                taps.push_layer_row(idx, &streams);
            }
        }
        if let Some(taps) = taps.as_deref_mut() {
            taps.push_pre_head_row(&streams);
            taps.note_positions(1);
        }
        let mut hidden = self.hyper_head(&streams)?;
        rms_norm_in_place(&mut hidden, &self.output_norm, self.rms_eps)?;
        let logits = self.matvec(&self.output_head, &hidden)?;
        state.pos += 1;
        Ok(logits)
    }

    /// Process a chunk of prompt tokens in one batched pass: heavy linears go
    /// through [`DsV4Linear::mul_mat`], the per-token host math (hyper
    /// connections, rope, softmax attention) parallelizes over rayon workers,
    /// and the compressor/indexer state advances with exact sequential
    /// semantics (blocks complete mid-chunk in order; each query sees exactly
    /// the blocks a sequential run would, including one its own token just
    /// completed). Results are identical to feeding the tokens through
    /// [`Self::step`] one at a time — the batched-prefill parity tests gate
    /// exactly that. `heads` selects which positions get the output head:
    /// none (interior prefill chunks), the last (prefill tail), or every
    /// position (speculative verify) — the per-position head is the
    /// sequential step's exact math (hyper-head collapse, final norm, one
    /// lm-head row through `mul_mat`, which the exact-batching mode serves as
    /// the sequential `mul_vec`). `taps` optionally captures per-position
    /// hidden states; `None` skips all capture work.
    fn step_chunk_heads(
        &self,
        state: &mut DsV4State,
        tokens: &[u32],
        heads: ChunkHeads,
        mut taps: Option<&mut DsV4Taps>,
    ) -> Result<Vec<Vec<f32>>> {
        let g = &self.geometry;
        let b = tokens.len();
        let pos0 = state.pos;
        if pos0 + b > g.context {
            bail!(
                "input length {} exceeds deepseek4 context length {}",
                pos0 + b,
                g.context
            );
        }
        // Host-side mutation: sever any device-state link (see host_step).
        state.device_tag = 0;
        let ring_slack = state.ring_slack;
        let mut streams: Vec<Vec<Vec<f32>>> = Vec::with_capacity(b);
        for &token in tokens {
            streams.push(vec![self.embed_row(token)?; g.hc]);
        }
        let params = self.hc_params();
        for (idx, (layer, layer_state)) in self.layers.iter().zip(&mut state.layers).enumerate() {
            let (ys, posts, combs) =
                chunk_hc_pre(&layer.hc_attn, &streams, &layer.attn_norm, params)?;
            let attn = self.attention_chunk(layer, layer_state, &ys, pos0, ring_slack)?;
            chunk_hc_post(&mut streams, &attn, &posts, &combs);

            let (ys, posts, combs) =
                chunk_hc_pre(&layer.hc_ffn, &streams, &layer.ffn_norm, params)?;
            let ffn = self.moe_chunk(layer, &ys, tokens)?;
            chunk_hc_post(&mut streams, &ffn, &posts, &combs);
            if let Some(taps) = taps.as_deref_mut()
                && taps.wants_layer(idx)
            {
                for token_streams in &streams {
                    taps.push_layer_row(idx, token_streams);
                }
            }
        }
        state.pos += b;
        if let Some(taps) = taps.as_deref_mut() {
            for token_streams in &streams {
                taps.push_pre_head_row(token_streams);
            }
            taps.note_positions(b);
        }
        match heads {
            ChunkHeads::None => Ok(Vec::new()),
            ChunkHeads::Last => {
                let mut hidden = self.hyper_head(&streams[b - 1])?;
                rms_norm_in_place(&mut hidden, &self.output_norm, self.rms_eps)?;
                Ok(vec![self.matvec(&self.output_head, &hidden)?])
            }
            ChunkHeads::All => {
                let mut hiddens = Vec::with_capacity(b);
                for token_streams in &streams {
                    let mut hidden = self.hyper_head(token_streams)?;
                    rms_norm_in_place(&mut hidden, &self.output_norm, self.rms_eps)?;
                    hiddens.push(hidden);
                }
                self.linear
                    .mul_mat(TensorKey::Dense(&self.output_head), &hiddens)
            }
        }
    }

    /// Batched form of [`Self::attention`] over a chunk of queries: causal
    /// within the chunk plus the cached raw window and compressed blocks, with
    /// per-query visibility identical to the sequential path.
    fn attention_chunk(
        &self,
        layer: &DsV4Layer,
        layer_state: &mut DsV4LayerState,
        xs: &[Vec<f32>],
        pos0: usize,
        ring_slack: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let g = &self.geometry;
        let b = xs.len();

        let mut qrs = self.linear.mul_mat(TensorKey::Dense(&layer.q_a), xs)?;
        for qr in &mut qrs {
            rms_norm_in_place(qr, &layer.q_a_norm, self.rms_eps)?;
        }
        let mut qs = self.linear.mul_mat(TensorKey::Dense(&layer.q_b), &qrs)?;
        let (head_dim, rope_dims, rope_base, rms_eps) =
            (g.head_dim, g.rope_dims, layer.rope_base, self.rms_eps);
        qs.par_iter_mut().enumerate().for_each(|(t, q)| {
            for head in q.chunks_mut(head_dim) {
                // Same unweighted per-head RMS + rope as the sequential path.
                let mean_square =
                    head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
                let inv = (mean_square + rms_eps).sqrt().recip();
                for value in head.iter_mut() {
                    *value *= inv;
                }
                v4_rope_tail(head, rope_dims, pos0 + t, rope_base, false);
            }
        });

        let mut kvs = self.linear.mul_mat(TensorKey::Dense(&layer.kv), xs)?;
        for (t, kv) in kvs.iter_mut().enumerate() {
            rms_norm_in_place(kv, &layer.kv_norm, self.rms_eps)?;
            v4_rope_tail(kv, g.rope_dims, pos0 + t, layer.rope_base, false);
        }

        // Compressors consume every token; blocks complete mid-chunk exactly
        // where the sequential path would emit them.
        if let (Some(weights), Some(cstate)) = (&layer.compressor, &mut layer_state.compressor) {
            self.compressor_update_chunk(weights, cstate, xs)?;
        }
        if let (Some(indexer), Some(istate)) = (&layer.indexer, &mut layer_state.indexer) {
            self.compressor_update_chunk(&indexer.compressor, istate, xs)?;
        }

        // Per-query indexer narrowing. Query t sees exactly the block prefix
        // that exists after its own token's update — (pos0+t+1)/ratio entries
        // of the append-only key list (a block completed BY token t is visible
        // to it) — and narrows only once that count exceeds top_k. The
        // projections feeding the top-k run through the per-token mul_vec, NOT
        // the batched GEMM: block selection is discrete, and a batched
        // reduction order flipping one near-tied score would swap whole blocks
        // in and out of a query's attention. The scoring math itself still
        // parallelizes over rayon workers.
        let selections: Vec<Option<Vec<usize>>> = match (&layer.indexer, &layer_state.indexer) {
            (Some(indexer), Some(istate)) => {
                let ratio = indexer.compressor.ratio;
                let (idx_heads, idx_key, idx_top_k) = (g.idx_heads, g.idx_key, g.idx_top_k);
                let mut projections: Vec<Option<(Vec<f32>, Vec<f32>)>> = Vec::with_capacity(b);
                for t in 0..b {
                    projections.push(if (pos0 + t + 1) / ratio > idx_top_k {
                        let qi = self.matvec(&indexer.q_b, &qrs[t])?;
                        let head_weights = self.matvec(&indexer.proj, &xs[t])?;
                        Some((qi, head_weights))
                    } else {
                        None
                    });
                }
                let keys = &istate.keys;
                projections
                    .into_par_iter()
                    .enumerate()
                    .map(|(t, projections)| {
                        projections.map(|(qi, head_weights)| {
                            let visible = (pos0 + t + 1) / ratio;
                            indexer_select_math(
                                &qi,
                                head_weights,
                                &keys[..visible],
                                idx_heads,
                                idx_key,
                                idx_top_k,
                            )
                        })
                    })
                    .collect()
            }
            _ => vec![None; b],
        };

        // Raw candidates for query t are the pre-chunk ring entries followed
        // by the chunk's own latents up to and including t; the attention
        // window keeps the trailing min(window, pos0+t+1) of them.
        let DsV4LayerState {
            ring,
            compressor,
            indexer: _,
        } = layer_state;
        ring.make_contiguous();
        let (prev, tail) = ring.as_slices();
        debug_assert!(tail.is_empty());
        let prev_len = prev.len();
        let cstate = compressor.as_ref();
        let ratio = layer.compressor.as_ref().map(|weights| weights.ratio);

        let per_query: Vec<(Vec<&[f32]>, Vec<&[f32]>)> = (0..b)
            .map(|t| {
                let mut keys: Vec<&[f32]> = Vec::new();
                let mut values: Vec<&[f32]> = Vec::new();
                if let (Some(cstate), Some(ratio)) = (cstate, ratio) {
                    match &selections[t] {
                        Some(blocks) => {
                            for &block in blocks {
                                keys.push(cstate.keys[block].as_slice());
                                values.push(cstate.values[block].as_slice());
                            }
                        }
                        None => {
                            for block in 0..(pos0 + t + 1) / ratio {
                                keys.push(cstate.keys[block].as_slice());
                                values.push(cstate.values[block].as_slice());
                            }
                        }
                    }
                }
                let candidates = prev_len + t + 1;
                let visible_raw = g
                    .window
                    .map_or(candidates, |window| window.min(pos0 + t + 1));
                for idx in candidates.saturating_sub(visible_raw)..candidates {
                    let entry = if idx < prev_len {
                        prev[idx].as_slice()
                    } else {
                        kvs[idx - prev_len].as_slice()
                    };
                    keys.push(entry);
                    values.push(entry);
                }
                (keys, values)
            })
            .collect();

        // Attention proper, parallel over heads x queries (pure host math;
        // per (query, head) it is the sequential path verbatim).
        let scale = (g.head_dim as f32).powf(-0.5);
        let heads = g.heads;
        let sinks = layer.sinks.as_deref();
        let mut out = vec![0.0f32; b * heads * head_dim];
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(chunk_idx, out_head)| {
                let t = chunk_idx / heads;
                let head = chunk_idx % heads;
                let q_head = &qs[t][head * head_dim..(head + 1) * head_dim];
                let (keys, values) = &per_query[t];
                let mut weights: Vec<f32> =
                    keys.iter().map(|key| dot(q_head, key) * scale).collect();
                let sink = sinks.map(|sinks| sinks[head]);
                let mut max = sink.unwrap_or(f32::NEG_INFINITY);
                for weight in &weights {
                    max = max.max(*weight);
                }
                let mut denom = sink.map(|sink| (sink - max).exp()).unwrap_or(0.0);
                for weight in &mut weights {
                    *weight = (*weight - max).exp();
                    denom += *weight;
                }
                for (weight, value) in weights.iter().zip(values) {
                    let weight = weight / denom;
                    for (out, value) in out_head.iter_mut().zip(*value) {
                        *out += weight * value;
                    }
                }
                v4_rope_tail(out_head, rope_dims, pos0 + t, rope_base, true);
            });
        drop(per_query);

        // Ring update: pushing the chunk then evicting once leaves exactly the
        // entries per-token push+evict would.
        for kv in kvs {
            ring.push_back(kv);
        }
        if let Some(window) = g.window {
            while ring.len() > window + ring_slack {
                ring.pop_front();
            }
        }

        let outs: Vec<Vec<f32>> = out.chunks(heads * head_dim).map(<[f32]>::to_vec).collect();
        let projected = self.linear.mul_mat(
            TensorKey::Grouped {
                matrix: &layer.out_a,
                rank: g.o_rank,
            },
            &outs,
        )?;
        self.linear
            .mul_mat(TensorKey::Dense(&layer.out_b), &projected)
    }

    /// Batched form of [`Self::moe`]: the whole MoE block goes through the
    /// provider's [`DsV4Linear::moe_block`] (default = the exact host path in
    /// [`host_moe_block`]; the GPU provider runs it device-side).
    fn moe_chunk(
        &self,
        layer: &DsV4Layer,
        xs: &[Vec<f32>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        self.linear.moe_block(&self.moe_ctx(layer), xs, tokens)
    }

    /// Bundle a layer's MoE references + routing scalars for
    /// [`DsV4Linear::moe_block`]. Crate-visible so the device step can build
    /// the identical ctx for its inlined per-layer MoE block.
    pub(crate) fn moe_ctx<'a>(&'a self, layer: &'a DsV4Layer) -> DsV4MoeBlockCtx<'a> {
        let g = &self.geometry;
        DsV4MoeBlockCtx {
            router: &layer.router,
            probs_bias: layer.probs_bias.as_deref(),
            tid2eid: layer.tid2eid.as_ref(),
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

    /// Rewind `state` to the largest reconstructable position `<= target` and
    /// return it. A position is reconstructable when (a) every windowed raw
    /// ring still holds that position's full attention window (states built
    /// with ring slack retain history for exactly this) and (b) every
    /// compressor either stays inside its current incomplete block (the
    /// pending buffer shortens) or lands on a clean block boundary (whole
    /// blocks drop; mid-block interiors are unrecoverable once compressed).
    /// Returns `None` with `state` untouched when no position > 0 works.
    // Consumed by the native-cuda serving backend (and tests).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn truncate_state_to_at_most(
        &self,
        state: &mut DsV4State,
        target: usize,
    ) -> Option<usize> {
        // The scan only ever needs to cover block-boundary rounding (max
        // compress ratio) plus the ring slack; anything deeper is infeasible
        // for windowed rings anyway.
        const MAX_TRUNCATION_SCAN: usize = 1024;
        let target = target.min(state.pos);
        let position = (0..=MAX_TRUNCATION_SCAN.min(target))
            .map(|back| target - back)
            .find(|&candidate| candidate > 0 && self.truncation_feasible(state, candidate))?;
        let drop = state.pos - position;
        // Truncation deliberately KEEPS the device tag: it reconstructs an
        // exact PREFIX of the tag's trajectory (bit-exact contract), so the
        // lineage invariant on `device_tag` still holds at the new position.
        // The provider sees (tag, pos < dev.pos) and either adopts the rewind
        // device-side under a FRESH tag (cheap — the spec loop's per-verify
        // rollback) or falls back to a full restore; content can never go
        // stale because adoption re-tags and every other state holding the
        // old (tag, pos) pair mismatches afterwards. Forward host mutations
        // (host_step / step_chunk_heads) still zero the tag.
        for (layer, layer_state) in self.layers.iter().zip(&mut state.layers) {
            let keep = layer_state.ring.len().saturating_sub(drop);
            layer_state.ring.truncate(keep);
            if let (Some(weights), Some(cstate)) = (&layer.compressor, &mut layer_state.compressor)
            {
                truncate_compressor(cstate, weights.ratio, position);
            }
            if let (Some(indexer), Some(istate)) = (&layer.indexer, &mut layer_state.indexer) {
                truncate_compressor(istate, indexer.compressor.ratio, position);
            }
        }
        state.pos = position;
        Some(position)
    }

    /// Replay one device step's state delta into the host mirror, advancing
    /// `state` exactly as [`Self::host_step`] would have: ring push + window
    /// eviction, pending pushes, and block completion (block appended, pending
    /// cleared) in the host order. The delta values were computed on device by
    /// kernels that are bit-exact ports of the host math, so the resulting
    /// mirror is indistinguishable from a host-stepped state.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn apply_device_step_mirror(
        &self,
        state: &mut DsV4State,
        mirror: &DsV4StepMirror<'_>,
    ) -> Result<()> {
        let g = &self.geometry;
        let layers = self.layers.len();
        if mirror.kv.len() != layers
            || mirror.x.len() != layers
            || mirror.comp_block.len() != layers
            || mirror.idx_block.len() != layers
        {
            bail!("device step mirror layer count does not match the model");
        }
        let ring_slack = state.ring_slack;
        for (idx, (layer, lstate)) in self.layers.iter().zip(&mut state.layers).enumerate() {
            let kv = mirror.kv[idx];
            if kv.len() != g.head_dim {
                bail!("device step mirror kv latent has wrong width {}", kv.len());
            }
            lstate.ring.push_back(kv.to_vec());
            if let Some(window) = g.window
                && lstate.ring.len() > window + ring_slack
            {
                lstate.ring.pop_front();
            }
            for (weights, cstate, block) in [
                (
                    layer.compressor.as_ref(),
                    lstate.compressor.as_mut(),
                    mirror.comp_block[idx],
                ),
                (
                    layer.indexer.as_ref().map(|indexer| &indexer.compressor),
                    lstate.indexer.as_mut(),
                    mirror.idx_block[idx],
                ),
            ] {
                let (Some(weights), Some(cstate)) = (weights, cstate) else {
                    continue;
                };
                let x = mirror.x[idx].ok_or_else(|| {
                    anyhow!("device step mirror is missing layer {idx}'s activation")
                })?;
                cstate.pending.push(x.to_vec());
                if cstate.pending.len() >= weights.ratio {
                    let (key, value) = block.ok_or_else(|| {
                        anyhow!("device step mirror is missing layer {idx}'s compressed block")
                    })?;
                    if key.len() != weights.dim || value.len() != weights.dim {
                        bail!("device step mirror block has wrong width {}", key.len());
                    }
                    cstate.keys.push(key.to_vec());
                    cstate.values.push(value.to_vec());
                    cstate.pending.clear();
                }
            }
        }
        state.pos += 1;
        Ok(())
    }

    fn truncation_feasible(&self, state: &DsV4State, position: usize) -> bool {
        let drop = state.pos - position;
        self.layers
            .iter()
            .zip(&state.layers)
            .all(|(layer, layer_state)| {
                let remaining = layer_state.ring.len().saturating_sub(drop);
                let needed = self
                    .geometry
                    .window
                    .map_or(position, |window| window.min(position));
                if remaining < needed {
                    return false;
                }
                if let (Some(weights), Some(cstate)) = (&layer.compressor, &layer_state.compressor)
                    && !compressor_truncation_feasible(cstate, weights.ratio, position)
                {
                    return false;
                }
                if let (Some(indexer), Some(istate)) = (&layer.indexer, &layer_state.indexer)
                    && !compressor_truncation_feasible(istate, indexer.compressor.ratio, position)
                {
                    return false;
                }
                true
            })
    }

    fn hc_params(&self) -> HcParams {
        HcParams {
            hc: self.geometry.hc,
            embed: self.geometry.embed,
            rms_eps: self.rms_eps,
            hc_eps: self.hc_eps,
            sinkhorn_iterations: self.geometry.sinkhorn_iterations,
        }
    }

    /// HyperConnection.pre: collapse the hc streams to one activation and
    /// produce the post gates and the sinkhorn-normalized comb matrix.
    fn hc_pre(
        &self,
        hc: &HcWeights,
        streams: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        hc_pre_math(hc, streams, self.hc_params())
    }

    /// HyperHead: like hc.pre but only the first hc mixes (pre gates), used to
    /// collapse the streams before the final norm + lm head.
    fn hyper_head(&self, streams: &[Vec<f32>]) -> Result<Vec<f32>> {
        hyper_head_math(
            &self.hyper_head,
            streams,
            self.geometry.embed,
            self.rms_eps,
            self.hc_eps,
        )
    }

    fn attention(
        &self,
        layer: &DsV4Layer,
        layer_state: &mut DsV4LayerState,
        x: &[f32],
        pos: usize,
        ring_slack: usize,
    ) -> Result<Vec<f32>> {
        dsv4_attention_step(
            &self.linear,
            &self.geometry,
            self.rms_eps,
            layer,
            layer_state,
            x,
            pos,
            ring_slack,
        )
    }

    /// Chunked-prefill form of [`dsv4_compressor_update`]: feed the chunk's
    /// tokens in order, batching the block projections through
    /// [`DsV4Linear::mul_mat`] whenever a block completes mid-chunk. Block
    /// completion order (and therefore the compressed cache contents) is
    /// exactly the sequential path's.
    fn compressor_update_chunk(
        &self,
        weights: &CompressorWeights,
        state: &mut CompressorState,
        xs: &[Vec<f32>],
    ) -> Result<()> {
        for x in xs {
            state.pending.push(x.clone());
            if state.pending.len() < weights.ratio {
                continue;
            }
            let gates = self
                .linear
                .mul_mat(TensorKey::Dense(&weights.gate), &state.pending)?;
            let kvs = self
                .linear
                .mul_mat(TensorKey::Dense(&weights.kv), &state.pending)?;
            compressor_emit_block(weights, state, gates, kvs, self.rms_eps)?;
        }
        Ok(())
    }

    /// One token's MoE block, via the provider's [`DsV4Linear::moe_block`]
    /// (B=1). Bit-identical to the pre-seam per-token path: the host
    /// implementation issues the same router matvec, routing math, per-expert
    /// matmuls (one row through `mul_mat` = the sequential `mul_vec`), and the
    /// same selection-order accumulation.
    fn moe(&self, layer: &DsV4Layer, x: &[f32], token: u32) -> Result<Vec<f32>> {
        let xs = [x.to_vec()];
        let mut out = self.linear.moe_block(&self.moe_ctx(layer), &xs, &[token])?;
        out.pop()
            .ok_or_else(|| anyhow!("moe_block returned no output rows"))
    }

    /// Dense matvec, routed through the provider.
    fn matvec(&self, matrix: &RawMatrix, x: &[f32]) -> Result<Vec<f32>> {
        self.linear.mul_vec(TensorKey::Dense(matrix), x)
    }

    /// Dequantize just the token's embedding row. Host-side for the host
    /// paths; the device step gathers the same row from a packed device copy
    /// (bit-identical dequant) and only falls back here for exotic embedding
    /// dtypes.
    pub(crate) fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        dsv4_embed_row(&self.gguf, &self.token_embd, self.geometry.vocab, token)
    }

    /// The GGUF the engine reads its weights from. The MTP drafter clones the
    /// `Arc` so it can serve the target-shared embedding and lm-head rows
    /// without borrowing the engine.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn gguf(&self) -> &Arc<GgufFile> {
        &self.gguf
    }
}

/// [`DsV4Engine::embed_row`]'s body as a free function: dequantize one
/// embedding row straight out of the GGUF mmap (bit-identical for every
/// caller — the engine, and the MTP drafter which holds its own `Arc`).
pub(crate) fn dsv4_embed_row(
    gguf: &GgufFile,
    token_embd: &RawMatrix,
    vocab: usize,
    token: u32,
) -> Result<Vec<f32>> {
    let token = usize::try_from(token).context("token id does not fit usize")?;
    if token >= vocab {
        bail!("token id {token} is outside vocab size {vocab}");
    }
    let view = gguf
        .tensor(&token_embd.name)
        .ok_or_else(|| anyhow!("GGUF tensor {} is missing", token_embd.name))?;
    dequantize_elem_range(&view, token * token_embd.cols, token_embd.cols)
}

impl DsV4Linear for DsV4CpuLinear {
    fn mul_vec(&self, key: TensorKey<'_>, x: &[f32]) -> Result<Vec<f32>> {
        match key {
            TensorKey::Dense(matrix) => self.dense_mul_vec(matrix, x),
            TensorKey::Grouped { matrix, rank } => self.grouped_mul_vec(matrix, rank, x),
            TensorKey::Expert { experts, expert } => self.expert_mul_vec(experts, expert, x),
        }
    }
}

/// The host reference for one layer's whole MoE block — the exact pre-seam
/// engine path, shared by [`DsV4Linear::moe_block`]'s default implementation
/// and any provider's fallback (the GPU kill switch `HI_DSV4_NO_DEVICE_MOE=1`
/// routes here). One routed pass per token with the batch's (token, expert)
/// selections grouped by expert id, so a provider serves each unique expert
/// once per call; B = 1 reproduces the sequential per-token path bit-for-bit
/// (single-row `mul_mat` defaults to `mul_vec`, and accumulation runs in
/// selection order either way).
pub(crate) fn host_moe_block<L: DsV4Linear + ?Sized>(
    linear: &L,
    ctx: &DsV4MoeBlockCtx<'_>,
    xs: &[Vec<f32>],
    tokens: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let b = xs.len();
    if tokens.len() != b {
        bail!(
            "moe_block got {} token ids for {b} activation rows",
            tokens.len()
        );
    }
    // Router logits per token via the sequential matvec, NOT a batched GEMM:
    // top-k expert selection is discrete, so its inputs must be bit-identical
    // to the sequential path — a batched reduction order flipping one
    // near-tied expert would route the token through entirely different
    // weights. The router is tiny; per-token GEMVs cost little.
    let router_logits: Vec<Vec<f32>> = xs
        .iter()
        .map(|x| linear.mul_vec(TensorKey::Dense(ctx.router), x))
        .collect::<Result<_>>()?;
    let mut selected_all = Vec::with_capacity(b);
    let mut weights_all = Vec::with_capacity(b);
    for (logits, &token) in router_logits.iter().zip(tokens) {
        let (selected, weights) = moe_route_math(ctx, logits, token)?;
        selected_all.push(selected);
        weights_all.push(weights);
    }

    let mut by_expert: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (t, selected) in selected_all.iter().enumerate() {
        for &expert in selected {
            by_expert.entry(expert).or_default().push(t);
        }
    }
    // Prefetch the whole batch's routed experts (the keys are the unique set,
    // in the ascending order the expert loop below consumes them).
    let unique: Vec<usize> = by_expert.keys().copied().collect();
    linear.prefetch_experts(
        DsV4ExpertTensors {
            gate: ctx.gate,
            up: ctx.up,
            down: ctx.down,
        },
        &unique,
    )?;
    let mut expert_out: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
    for (&expert, token_ids) in &mut by_expert {
        // Hash routing may select the same expert twice for one token; the
        // token list is ascending, so dedup keeps one row per token.
        token_ids.dedup();
        let expert_xs: Vec<Vec<f32>> = token_ids.iter().map(|&t| xs[t].clone()).collect();
        let gates = linear.mul_mat(
            TensorKey::Expert {
                experts: ctx.gate,
                expert,
            },
            &expert_xs,
        )?;
        let ups = linear.mul_mat(
            TensorKey::Expert {
                experts: ctx.up,
                expert,
            },
            &expert_xs,
        )?;
        let hiddens: Vec<Vec<f32>> = gates
            .into_iter()
            .zip(ups)
            .map(|(gate, up)| swiglu_hidden(gate, up, ctx.swiglu_clamp))
            .collect();
        let downs = linear.mul_mat(
            TensorKey::Expert {
                experts: ctx.down,
                expert,
            },
            &hiddens,
        )?;
        for (&t, down) in token_ids.iter().zip(downs) {
            expert_out.insert((expert, t), down);
        }
    }

    // Accumulate per token in selection order — the sequential path's
    // floating-point order — then add the shared expert, batched.
    let mut acc = vec![vec![0.0f32; ctx.embed]; b];
    for (t, acc) in acc.iter_mut().enumerate() {
        for (&expert, &weight) in selected_all[t].iter().zip(&weights_all[t]) {
            let out = expert_out
                .get(&(expert, t))
                .ok_or_else(|| anyhow!("missing chunked output for expert {expert}"))?;
            for (acc, value) in acc.iter_mut().zip(out) {
                *acc += weight * value;
            }
        }
    }
    if let Some(shared) = &ctx.shared {
        // Shared expert: plain SwiGLU, no clamp.
        let gates = linear.mul_mat(TensorKey::Dense(shared.gate), xs)?;
        let ups = linear.mul_mat(TensorKey::Dense(shared.up), xs)?;
        let hiddens: Vec<Vec<f32>> = gates
            .into_iter()
            .zip(ups)
            .map(|(gate, up)| {
                gate.iter()
                    .zip(&up)
                    .map(|(gate, up)| silu(*gate) * up)
                    .collect()
            })
            .collect();
        let downs = linear.mul_mat(TensorKey::Dense(shared.down), &hiddens)?;
        for (acc, down) in acc.iter_mut().zip(&downs) {
            for (acc, value) in acc.iter_mut().zip(down) {
                *acc += value;
            }
        }
    }
    Ok(acc)
}

/// Route one token from its router logits: sqrt-softplus scores (gating
/// func 4), hash or biased top-k selection (descending score, LOWER index
/// wins ties), then the raw selected scores normalized and scaled into
/// mixture weights. The GPU selection kernel mirrors this math operation for
/// operation (including the glibc expf/logf rounding), so keep any change
/// here in lockstep with `dsv4_moe_select_kernel` in kernels.cu.
pub(crate) fn moe_route_math(
    ctx: &DsV4MoeBlockCtx<'_>,
    logits: &[f32],
    token: u32,
) -> Result<(Vec<usize>, Vec<f32>)> {
    // Gating func 4 (sqrt-softplus); softplus already guards large |x|.
    let scores: Vec<f32> = logits.iter().map(|logit| softplus(*logit).sqrt()).collect();

    let selected: Vec<usize> = match ctx.tid2eid {
        Some(table) => {
            // Clamp the token id into the table like the reference does.
            // Entries were range-validated at load.
            let row = (token as usize).min(table.tokens.saturating_sub(1));
            table.values[row * table.stride..(row + 1) * table.stride]
                .iter()
                .map(|&expert| expert as usize)
                .collect()
        }
        None => {
            let mut adjusted = scores.clone();
            if let Some(bias) = ctx.probs_bias {
                for (score, bias) in adjusted.iter_mut().zip(bias) {
                    *score += *bias;
                }
            }
            let mut ranked: Vec<(usize, f32)> = adjusted.into_iter().enumerate().collect();
            ranked.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            ranked
                .into_iter()
                .take(ctx.top_k.min(ctx.experts))
                .map(|(expert, _)| expert)
                .collect()
        }
    };

    // Weights are the RAW scores at the selected experts (selection bias is
    // selection-only), normalized then scaled.
    let mut weights: Vec<f32> = selected.iter().map(|&expert| scores[expert]).collect();
    if ctx.weights_norm && weights.len() > 1 {
        let denom: f32 = weights.iter().sum();
        if denom > f32::EPSILON {
            for weight in &mut weights {
                *weight /= denom;
            }
        }
    }
    for weight in &mut weights {
        *weight *= ctx.weights_scale;
    }
    Ok((selected, weights))
}

/// One token's V4 attention (S=1), shared verbatim by [`DsV4Engine::step`]'s
/// host path and the MTP draft layer (`dsv4_mtp`, whose block is a
/// compress-ratio-1 layer: no compressor/indexer, raw SWA ring only). All
/// heavy linears go through `linear`; everything else is host f32 math.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsv4_attention_step<L: DsV4Linear + ?Sized>(
    linear: &L,
    g: &DsV4Geometry,
    rms_eps: f32,
    layer: &DsV4Layer,
    layer_state: &mut DsV4LayerState,
    x: &[f32],
    pos: usize,
    ring_slack: usize,
) -> Result<Vec<f32>> {
    let mut qr = linear.mul_vec(TensorKey::Dense(&layer.q_a), x)?;
    rms_norm_in_place(&mut qr, &layer.q_a_norm, rms_eps)?;
    let mut q = linear.mul_vec(TensorKey::Dense(&layer.q_b), &qr)?;
    for head in q.chunks_mut(g.head_dim) {
        // Unweighted per-head RMS scaling (no learned weight), then rope on
        // the trailing rope_dims at the query position.
        let mean_square = head.iter().map(|value| value * value).sum::<f32>() / g.head_dim as f32;
        let inv = (mean_square + rms_eps).sqrt().recip();
        for value in head.iter_mut() {
            *value *= inv;
        }
        v4_rope_tail(head, g.rope_dims, pos, layer.rope_base, false);
    }

    let mut kv = linear.mul_vec(TensorKey::Dense(&layer.kv), x)?;
    rms_norm_in_place(&mut kv, &layer.kv_norm, rms_eps)?;
    v4_rope_tail(&mut kv, g.rope_dims, pos, layer.rope_base, false);
    layer_state.ring.push_back(kv);
    if let Some(window) = g.window
        && layer_state.ring.len() > window + ring_slack
    {
        layer_state.ring.pop_front();
    }

    // Both the attention compressor and the indexer's private compressor
    // consume every token; blocks appear once `ratio` tokens accumulate.
    if let (Some(weights), Some(cstate)) = (&layer.compressor, &mut layer_state.compressor) {
        dsv4_compressor_update(linear, rms_eps, weights, cstate, x)?;
    }
    if let (Some(indexer), Some(istate)) = (&layer.indexer, &mut layer_state.indexer) {
        dsv4_compressor_update(linear, rms_eps, &indexer.compressor, istate, x)?;
    }
    // Decode-form block causality: every complete block ends at or before
    // the current position, so all cached blocks are visible. The indexer
    // narrows to its top_k blocks only once more than top_k exist.
    let selected = match (&layer.indexer, &layer_state.indexer) {
        (Some(indexer), Some(istate)) if istate.keys.len() > g.idx_top_k => {
            Some(dsv4_indexer_select(linear, g, indexer, istate, &qr, x)?)
        }
        _ => None,
    };

    // K/V order: compressed blocks first, then the raw ring (order does not
    // affect the softmax; kept to mirror the reference concatenation).
    let mut keys: Vec<&[f32]> = Vec::new();
    let mut values: Vec<&[f32]> = Vec::new();
    if let Some(cstate) = &layer_state.compressor {
        match &selected {
            Some(blocks) => {
                for &block in blocks {
                    keys.push(&cstate.keys[block]);
                    values.push(&cstate.values[block]);
                }
            }
            None => {
                for (key, value) in cstate.keys.iter().zip(&cstate.values) {
                    keys.push(key);
                    values.push(value);
                }
            }
        }
    }
    // Only the trailing `window` ring entries are attention-visible; any
    // slack-retained older entries exist purely for state truncation.
    let raw_skip = g
        .window
        .map_or(0, |window| layer_state.ring.len().saturating_sub(window));
    for entry in layer_state.ring.iter().skip(raw_skip) {
        keys.push(entry);
        values.push(entry);
    }

    let scale = (g.head_dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; g.heads * g.head_dim];
    for (head, (q_head, out_head)) in q
        .chunks(g.head_dim)
        .zip(out.chunks_mut(g.head_dim))
        .enumerate()
    {
        let mut weights: Vec<f32> = keys.iter().map(|key| dot(q_head, key) * scale).collect();
        let sink = layer.sinks.as_ref().map(|sinks| sinks[head]);
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
        for (weight, value) in weights.iter().zip(&values) {
            let weight = weight / denom;
            for (out, value) in out_head.iter_mut().zip(*value) {
                *out += weight * value;
            }
        }
        // Inverse rope (negated angle) on the output tail at the query pos.
        v4_rope_tail(out_head, g.rope_dims, pos, layer.rope_base, true);
    }

    let projected = linear.mul_vec(
        TensorKey::Grouped {
            matrix: &layer.out_a,
            rank: g.o_rank,
        },
        &out,
    )?;
    linear.mul_vec(TensorKey::Dense(&layer.out_b), &projected)
}

/// Buffer the token; once `ratio` tokens accumulate, emit one compressed
/// K/V block: gated (softmax over block positions, per channel) average of
/// the block's wkv projections, halves RMS-normed with the shared weight.
fn dsv4_compressor_update<L: DsV4Linear + ?Sized>(
    linear: &L,
    rms_eps: f32,
    weights: &CompressorWeights,
    state: &mut CompressorState,
    x: &[f32],
) -> Result<()> {
    state.pending.push(x.to_vec());
    if state.pending.len() < weights.ratio {
        return Ok(());
    }
    let mut gates = Vec::with_capacity(weights.ratio);
    let mut kvs = Vec::with_capacity(weights.ratio);
    for token in &state.pending {
        gates.push(linear.mul_vec(TensorKey::Dense(&weights.gate), token)?);
        kvs.push(linear.mul_vec(TensorKey::Dense(&weights.kv), token)?);
    }
    compressor_emit_block(weights, state, gates, kvs, rms_eps)
}

/// Lightning-indexer top-k block selection (no rope anywhere in it):
/// score[b] = Σ_h w[h] · relu(q_h · ick_b) · idx_key^-0.5 with
/// w = proj(x) · idx_heads^-0.5. Returns ascending block indices (gather
/// order does not affect the attention softmax).
fn dsv4_indexer_select<L: DsV4Linear + ?Sized>(
    linear: &L,
    g: &DsV4Geometry,
    indexer: &IndexerWeights,
    istate: &CompressorState,
    qr: &[f32],
    x: &[f32],
) -> Result<Vec<usize>> {
    let qi = linear.mul_vec(TensorKey::Dense(&indexer.q_b), qr)?;
    let head_weights = linear.mul_vec(TensorKey::Dense(&indexer.proj), x)?;
    Ok(indexer_select_math(
        &qi,
        head_weights,
        &istate.keys,
        g.idx_heads,
        g.idx_key,
        g.idx_top_k,
    ))
}

impl DsV4CpuLinear {
    /// Build a CPU provider over an already-open GGUF. The engine constructs
    /// its own; this entry point serves providers that overlay extra host
    /// weights on top of the GGUF (the MTP host reference in `dsv4_mtp`) and
    /// the GPU MoE-block parity test's pure-CPU oracle.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_gguf(gguf: Arc<GgufFile>) -> Self {
        Self { gguf }
    }

    /// Test-only alias of [`Self::from_gguf`] (kept for the existing suites).
    #[cfg(test)]
    pub(crate) fn new_for_tests(gguf: Arc<GgufFile>) -> Self {
        Self::from_gguf(gguf)
    }

    fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        self.gguf
            .tensor(name)
            .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))
    }

    /// Dense matvec over a raw matrix, dequantizing a chunk of rows at a time
    /// so the transient f32 buffer stays small even for the vocab-sized lm
    /// head. Bit-identical to whole-matrix dequantization (see
    /// [`MATVEC_CHUNK_ROWS`]).
    fn dense_mul_vec(&self, matrix: &RawMatrix, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != matrix.cols {
            bail!(
                "matvec input length {} does not match tensor {} input dim {}",
                x.len(),
                matrix.name,
                matrix.cols
            );
        }
        let view = self.tensor_view(&matrix.name)?;
        let mut out = Vec::with_capacity(matrix.rows);
        let mut row = 0;
        while row < matrix.rows {
            let take = MATVEC_CHUNK_ROWS.min(matrix.rows - row);
            let data = dequantize_elem_range(&view, row * matrix.cols, take * matrix.cols)?;
            for chunk_row in 0..take {
                out.push(dot(
                    &data[chunk_row * matrix.cols..(chunk_row + 1) * matrix.cols],
                    x,
                ));
            }
            row += take;
        }
        Ok(out)
    }

    /// Block-diagonal matvec (wo_a): output rows g*rank..(g+1)*rank read input
    /// slice g of `matrix.cols` elements.
    fn grouped_mul_vec(&self, matrix: &RawMatrix, rank: usize, x: &[f32]) -> Result<Vec<f32>> {
        let group_features = matrix.cols;
        if rank == 0 || !matrix.rows.is_multiple_of(rank) {
            bail!(
                "grouped matvec tensor {} shape [{}, {}] does not fit rank {rank} x group width {group_features}",
                matrix.name,
                matrix.cols,
                matrix.rows
            );
        }
        let groups = matrix.rows / rank;
        if x.len() != groups * group_features {
            bail!(
                "grouped matvec input length {} does not match {groups} groups of {group_features}",
                x.len()
            );
        }
        let view = self.tensor_view(&matrix.name)?;
        let data = dequantize_elem_range(&view, 0, matrix.rows * matrix.cols)?;
        Ok((0..matrix.rows)
            .map(|row| {
                let group = row / rank;
                let x_group = &x[group * group_features..(group + 1) * group_features];
                dot(&data[row * matrix.cols..(row + 1) * matrix.cols], x_group)
            })
            .collect())
    }

    /// Matvec over the selected expert's contiguous slice of a rank-3 packed
    /// expert tensor; only that slice is dequantized.
    fn expert_mul_vec(&self, experts: &RawExperts, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != experts.in_dim {
            bail!(
                "expert matvec input length {} does not match tensor {} input dim {}",
                x.len(),
                experts.name,
                experts.in_dim
            );
        }
        let view = self.tensor_view(&experts.name)?;
        let per_expert = experts.in_dim * experts.out_dim;
        let data = dequantize_elem_range(&view, expert * per_expert, per_expert)?;
        Ok((0..experts.out_dim)
            .map(|row| dot(&data[row * experts.in_dim..(row + 1) * experts.in_dim], x))
            .collect())
    }
}

impl DsV4Layer {
    fn load(
        gguf: &GgufFile,
        config: &QwenGgufConfig,
        geometry: &DsV4Geometry,
        idx: usize,
        ratio: usize,
        hash_layers: usize,
        expert_ff: usize,
        shared_ff: usize,
        rope_base: f32,
        compress_rope_base: f32,
    ) -> Result<Self> {
        let prefix = format!("blk.{idx}");
        let embed = geometry.embed;
        let head_dim = geometry.head_dim;
        let q_dim = geometry.heads * head_dim;

        let compressor = if ratio > 0 {
            Some(CompressorWeights::load(
                gguf,
                &format!("{prefix}.attn_compressor"),
                embed,
                head_dim,
                ratio,
            )?)
        } else {
            None
        };
        // Indexer exists exactly on ratio-4 layers (see the geometry note).
        // Its queries come from the q latent, so q_b's input width is q_lora.
        let indexer = if ratio == 4 {
            Some(IndexerWeights {
                q_b: raw_matrix(
                    gguf,
                    &format!("{prefix}.indexer.attn_q_b.weight"),
                    geometry.idx_heads * geometry.idx_key,
                    geometry.q_lora,
                )?,
                proj: raw_matrix(
                    gguf,
                    &format!("{prefix}.indexer.proj.weight"),
                    geometry.idx_heads,
                    embed,
                )?,
                compressor: CompressorWeights::load(
                    gguf,
                    &format!("{prefix}.indexer_compressor"),
                    embed,
                    geometry.idx_key,
                    ratio,
                )?,
            })
        } else {
            None
        };

        let tid2eid = if idx < hash_layers {
            let name = format!("{prefix}.ffn_gate_tid2eid.weight");
            let view = gguf
                .tensor(&name)
                .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
            let dims: Vec<usize> = view
                .info
                .dimensions
                .iter()
                .map(|dim| usize::try_from(*dim).context("tensor dimension does not fit usize"))
                .collect::<Result<Vec<_>>>()?;
            let [stride, tokens] = dims.as_slice() else {
                bail!("tensor {name} must be rank 2, got {dims:?}");
            };
            if *stride != geometry.moe_top_k {
                bail!(
                    "tensor {name} has {stride} experts per token; expected expert_used_count {}",
                    geometry.moe_top_k
                );
            }
            if view.info.dtype != GgufTensorType::I32 {
                bail!(
                    "deepseek4 tensor {name} must be I32, got {}",
                    view.info.dtype.label()
                );
            }
            // Materialize + range-validate the whole table once (it is tiny)
            // so per-token routing — host or device — can index it blindly.
            let count = stride
                .checked_mul(*tokens)
                .context("tid2eid element count overflows usize")?;
            let bytes = view
                .bytes
                .get(..count * 4)
                .ok_or_else(|| anyhow!("deepseek4 tensor {name} is truncated"))?;
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|raw| i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
                .collect();
            for (at, &expert) in values.iter().enumerate() {
                let token = at / stride;
                if expert < 0 {
                    bail!("deepseek4 tid2eid entry {expert} for token {token} is negative");
                }
                if expert as usize >= geometry.experts {
                    bail!(
                        "deepseek4 tid2eid entry {expert} for token {token} exceeds expert_count {}",
                        geometry.experts
                    );
                }
            }
            Some(Tid2Eid {
                name,
                stride: *stride,
                tokens: *tokens,
                values,
            })
        } else {
            None
        };

        let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
        let shared = if shared_ff > 0 && gguf.tensor_info(&shared_gate_name).is_some() {
            Some(SharedExpertWeights {
                gate: raw_matrix(gguf, &shared_gate_name, shared_ff, embed)?,
                up: raw_matrix(
                    gguf,
                    &format!("{prefix}.ffn_up_shexp.weight"),
                    shared_ff,
                    embed,
                )?,
                down: raw_matrix(
                    gguf,
                    &format!("{prefix}.ffn_down_shexp.weight"),
                    embed,
                    shared_ff,
                )?,
            })
        } else {
            None
        };

        Ok(Self {
            attn_norm: load_vector(gguf, &format!("{prefix}.attn_norm.weight"), embed)?,
            ffn_norm: load_vector(gguf, &format!("{prefix}.ffn_norm.weight"), embed)?,
            hc_attn: HcWeights::load(
                gguf,
                &format!("{prefix}.hc_attn"),
                geometry.hc * geometry.hc + 2 * geometry.hc,
                geometry.hc * embed,
                3,
            )?,
            hc_ffn: HcWeights::load(
                gguf,
                &format!("{prefix}.hc_ffn"),
                geometry.hc * geometry.hc + 2 * geometry.hc,
                geometry.hc * embed,
                3,
            )?,
            q_a: raw_matrix(
                gguf,
                &format!("{prefix}.attn_q_a.weight"),
                geometry.q_lora,
                embed,
            )?,
            q_a_norm: load_vector(
                gguf,
                &format!("{prefix}.attn_q_a_norm.weight"),
                geometry.q_lora,
            )?,
            q_b: raw_matrix(
                gguf,
                &format!("{prefix}.attn_q_b.weight"),
                q_dim,
                geometry.q_lora,
            )?,
            kv: raw_matrix(gguf, &format!("{prefix}.attn_kv.weight"), head_dim, embed)?,
            kv_norm: load_vector(gguf, &format!("{prefix}.attn_kv_a_norm.weight"), head_dim)?,
            sinks: optional_vector(gguf, &format!("{prefix}.attn_sinks.weight"), geometry.heads)?,
            out_a: raw_matrix(
                gguf,
                &format!("{prefix}.attn_output_a.weight"),
                geometry.o_groups * geometry.o_rank,
                q_dim / geometry.o_groups,
            )?,
            out_b: raw_matrix(
                gguf,
                &format!("{prefix}.attn_output_b.weight"),
                embed,
                geometry.o_groups * geometry.o_rank,
            )?,
            rope_base: if ratio == 0 {
                rope_base
            } else {
                compress_rope_base
            },
            compressor,
            indexer,
            router: raw_matrix(
                gguf,
                &format!("{prefix}.ffn_gate_inp.weight"),
                geometry.experts,
                embed,
            )?,
            probs_bias: optional_vector(
                gguf,
                &format!("{prefix}.exp_probs_b.bias"),
                geometry.experts,
            )?,
            tid2eid,
            gate_exps: raw_experts(
                gguf,
                &format!("{prefix}.ffn_gate_exps.weight"),
                embed,
                expert_ff,
                geometry.experts,
            )?,
            up_exps: raw_experts(
                gguf,
                &format!("{prefix}.ffn_up_exps.weight"),
                embed,
                expert_ff,
                geometry.experts,
            )?,
            down_exps: raw_experts(
                gguf,
                &format!("{prefix}.ffn_down_exps.weight"),
                expert_ff,
                embed,
                geometry.experts,
            )?,
            shared,
            swiglu_clamp: config
                .swiglu_clamp_exp
                .as_ref()
                .and_then(|clamps| clamps.get(idx))
                .copied()
                .unwrap_or(0.0),
        })
    }
}

impl HcWeights {
    fn load(
        gguf: &GgufFile,
        base: &str,
        rows: usize,
        cols: usize,
        scale_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            func: DsV4HcFunc::load(gguf, &format!("{base}_fn.weight"), rows, cols)?,
            base: load_vector(gguf, &format!("{base}_base.weight"), rows)?,
            scale: load_vector(gguf, &format!("{base}_scale.weight"), scale_len)?,
        })
    }
}

impl CompressorWeights {
    fn load(gguf: &GgufFile, base: &str, embed: usize, dim: usize, ratio: usize) -> Result<Self> {
        // The projection width varies with the compressor form (real GGUF
        // census, verified against the 43-layer checkpoint): ratio-4 layers and
        // every indexer compressor emit a split K|V pair (width 2*dim), while
        // ratio-128 layers emit a single shared latent (width dim) that serves
        // as both K and V. Read the width off the gate tensor and accept
        // exactly those two forms.
        let gate_name = format!("{base}_gate.weight");
        let info = gguf
            .tensor_info(&gate_name)
            .ok_or_else(|| anyhow!("GGUF tensor {gate_name} is missing"))?;
        let width = match info.dimensions.as_slice() {
            [_, ne1] => usize::try_from(*ne1)
                .with_context(|| format!("tensor {gate_name} width does not fit usize"))?,
            dims => bail!("tensor {gate_name} must be rank 2, got {dims:?}"),
        };
        if width != dim && width != 2 * dim {
            bail!(
                "tensor {gate_name} projects width {width}; expected {dim} (shared K=V) or {} (split K|V)",
                2 * dim
            );
        }
        Ok(Self {
            gate: raw_matrix(gguf, &gate_name, width, embed)?,
            kv: raw_matrix(gguf, &format!("{base}_kv.weight"), width, embed)?,
            ape: load_ape(gguf, &format!("{base}_ape.weight"), width, ratio)?,
            norm: load_vector(gguf, &format!("{base}_norm.weight"), dim)?,
            ratio,
            dim,
            width,
        })
    }
}

impl CompressorState {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// f32 payload bytes of the buffered (`pending`) plus compressed
    /// (`keys`/`values`) latents. The dominant term in a snapshot's host
    /// footprint; container overhead is ignored.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    fn payload_bytes(&self) -> usize {
        const F32: usize = std::mem::size_of::<f32>();
        [&self.pending, &self.keys, &self.values]
            .into_iter()
            .flat_map(|group| group.iter())
            .map(|latent| latent.len() * F32)
            .sum()
    }
}

/// Scalars threaded through the hyper-connection helpers so the chunked path
/// can run them on rayon workers without borrowing the engine (the provider
/// is deliberately not `Sync`). Fields are crate-visible so the MTP draft
/// layer (`dsv4_mtp`) can build one from its copied trunk geometry.
#[derive(Clone, Copy)]
pub(crate) struct HcParams {
    pub(crate) hc: usize,
    pub(crate) embed: usize,
    pub(crate) rms_eps: f32,
    pub(crate) hc_eps: f32,
    pub(crate) sinkhorn_iterations: usize,
}

/// The output hyper-head math (see [`DsV4Engine::hyper_head`]): RMS-scaled
/// sigmoid pre gates only, collapsing the hc streams to one activation.
/// Shared by the engine and the MTP module's own `hc_head_*` collapse.
pub(crate) fn hyper_head_math(
    hc: &HcWeights,
    streams: &[Vec<f32>],
    embed: usize,
    rms_eps: f32,
    hc_eps: f32,
) -> Result<Vec<f32>> {
    let mut flat = Vec::with_capacity(streams.iter().map(Vec::len).sum());
    for stream in streams {
        flat.extend_from_slice(stream);
    }
    let mean_square = flat.iter().map(|value| value * value).sum::<f32>() / flat.len() as f32;
    let inv = (mean_square + rms_eps).sqrt().recip();
    let mixes = hc.func.mul_vec(&flat)?;
    let mut collapsed = vec![0.0f32; embed];
    for (i, stream) in streams.iter().enumerate() {
        let weight = sigmoid(mixes[i] * inv * hc.scale[0] + hc.base[i]) + hc_eps;
        for (out, value) in collapsed.iter_mut().zip(stream) {
            *out += weight * value;
        }
    }
    Ok(collapsed)
}

/// HyperConnection.pre math (see [`DsV4Engine::hc_pre`]); shared verbatim by
/// the sequential and chunked paths (and the MTP draft layer).
pub(crate) fn hc_pre_math(
    hc: &HcWeights,
    streams: &[Vec<f32>],
    params: HcParams,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = params.hc;
    let embed = params.embed;
    let mut flat = Vec::with_capacity(n * embed);
    for stream in streams {
        flat.extend_from_slice(stream);
    }
    let mean_square = flat.iter().map(|value| value * value).sum::<f32>() / flat.len() as f32;
    let inv = (mean_square + params.rms_eps).sqrt().recip();
    let mut mixes = hc.func.mul_vec(&flat)?;
    for mix in &mut mixes {
        *mix *= inv;
    }

    let mut pre = vec![0.0f32; n];
    let mut post = vec![0.0f32; n];
    for i in 0..n {
        pre[i] = sigmoid(mixes[i] * hc.scale[0] + hc.base[i]) + params.hc_eps;
        post[i] = sigmoid(mixes[n + i] * hc.scale[1] + hc.base[n + i]) * 2.0;
    }
    // comb[i][j]: row-softmax over j, +eps, then sinkhorn balancing — one
    // column normalization followed by (iters-1) row/column pairs.
    let mut comb = vec![0.0f32; n * n];
    for i in 0..n {
        let row = &mut comb[i * n..(i + 1) * n];
        for (j, value) in row.iter_mut().enumerate() {
            *value = mixes[2 * n + i * n + j] * hc.scale[2] + hc.base[2 * n + i * n + j];
        }
        softmax_in_place(row);
        for value in row {
            *value += params.hc_eps;
        }
    }
    normalize_comb_columns(&mut comb, n, params.hc_eps);
    for _ in 1..params.sinkhorn_iterations {
        normalize_comb_rows(&mut comb, n, params.hc_eps);
        normalize_comb_columns(&mut comb, n, params.hc_eps);
    }

    let mut collapsed = vec![0.0f32; embed];
    for (weight, stream) in pre.iter().zip(streams) {
        for (out, value) in collapsed.iter_mut().zip(stream) {
            *out += weight * value;
        }
    }
    Ok((collapsed, post, comb))
}

/// Per-token hc.pre + pre-norm for a chunk, parallelized over tokens (each
/// token's math is independent and identical to the sequential call).
fn chunk_hc_pre(
    hc: &HcWeights,
    streams: &[Vec<Vec<f32>>],
    norm: &[f32],
    params: HcParams,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = streams
        .par_iter()
        .map(|token_streams| {
            let (mut y, post, comb) = hc_pre_math(hc, token_streams, params)?;
            rms_norm_in_place(&mut y, norm, params.rms_eps)?;
            Ok((y, post, comb))
        })
        .collect::<Result<_>>()?;
    let mut ys = Vec::with_capacity(results.len());
    let mut posts = Vec::with_capacity(results.len());
    let mut combs = Vec::with_capacity(results.len());
    for (y, post, comb) in results {
        ys.push(y);
        posts.push(post);
        combs.push(comb);
    }
    Ok((ys, posts, combs))
}

/// Per-token hc.post for a chunk: token t's stream set becomes
/// `hc_post(f_out[t], old streams[t], posts[t], combs[t])`.
fn chunk_hc_post(
    streams: &mut [Vec<Vec<f32>>],
    f_out: &[Vec<f32>],
    posts: &[Vec<f32>],
    combs: &[Vec<f32>],
) {
    streams
        .par_iter_mut()
        .zip(f_out.par_iter())
        .zip(posts.par_iter().zip(combs.par_iter()))
        .for_each(|((streams, f_out), (post, comb))| {
            *streams = hc_post(f_out, streams, post, comb);
        });
}

/// Shared block-emit math for the compressors: add the APE bias to the
/// block's gate projections, softmax-average the kv projections per channel
/// (softmax over the block's positions), RMS-norm the halves, append the
/// block, and clear the pending buffer. Identical operation order for the
/// sequential and chunked paths.
fn compressor_emit_block(
    weights: &CompressorWeights,
    state: &mut CompressorState,
    mut gates: Vec<Vec<f32>>,
    kvs: Vec<Vec<f32>>,
    rms_eps: f32,
) -> Result<()> {
    let out_dim = weights.width;
    for (row, gate) in gates.iter_mut().enumerate() {
        for (channel, value) in gate.iter_mut().enumerate() {
            *value += weights.ape[row * out_dim + channel];
        }
    }
    let mut compressed = vec![0.0f32; out_dim];
    for (channel, out) in compressed.iter_mut().enumerate() {
        let mut max = f32::NEG_INFINITY;
        for gate in &gates {
            max = max.max(gate[channel]);
        }
        let mut denom = 0.0f32;
        let mut weighted = 0.0f32;
        for (gate, kv) in gates.iter().zip(&kvs) {
            let weight = (gate[channel] - max).exp();
            denom += weight;
            weighted += weight * kv[channel];
        }
        *out = weighted / denom;
    }
    let mut key = compressed[..weights.dim].to_vec();
    rms_norm_in_place(&mut key, &weights.norm, rms_eps)?;
    // Split form (width == 2*dim) carries independent K|V halves; the
    // shared form's single latent serves as both (K = V), like the raw
    // ring entries.
    let value = if weights.width == weights.dim {
        key.clone()
    } else {
        let mut value = compressed[weights.dim..].to_vec();
        rms_norm_in_place(&mut value, &weights.norm, rms_eps)?;
        value
    };
    state.keys.push(key);
    state.values.push(value);
    state.pending.clear();
    Ok(())
}

/// Lightning-indexer top-k selection over an explicit key prefix (shared
/// verbatim by the sequential and chunked paths): score[b] = Σ_h w[h] ·
/// relu(q_h · key_b) · idx_key^-0.5 with w = head_weights · idx_heads^-0.5.
/// Returns ascending block indices.
fn indexer_select_math(
    qi: &[f32],
    mut head_weights: Vec<f32>,
    keys: &[Vec<f32>],
    idx_heads: usize,
    idx_key: usize,
    idx_top_k: usize,
) -> Vec<usize> {
    let head_scale = (idx_heads as f32).powf(-0.5);
    for weight in &mut head_weights {
        *weight *= head_scale;
    }
    let key_scale = (idx_key as f32).powf(-0.5);
    let mut scores = vec![0.0f32; keys.len()];
    for (q_head, weight) in qi.chunks(idx_key).zip(&head_weights) {
        for (score, key) in scores.iter_mut().zip(keys) {
            *score += weight * dot(q_head, key).max(0.0) * key_scale;
        }
    }
    let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut selected: Vec<usize> = ranked
        .into_iter()
        .take(idx_top_k)
        .map(|(block, _)| block)
        .collect();
    selected.sort_unstable();
    selected
}

/// SwiGLU inner activation with the per-layer clamp semantics: gate ceiled at
/// +clamp, up clamped to ±clamp (clamp <= 0 disables both), then
/// silu(gate) * up.
fn swiglu_hidden(mut gate: Vec<f32>, mut up: Vec<f32>, clamp: f32) -> Vec<f32> {
    if clamp > 0.0 {
        for value in &mut gate {
            *value = value.min(clamp);
        }
        for value in &mut up {
            *value = value.clamp(-clamp, clamp);
        }
    }
    gate.iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect()
}

/// Can this compressor state rewind to `position`? Yes when the position
/// keeps the current incomplete block (the pending buffer shortens) or lands
/// exactly on a block boundary (later blocks drop whole).
fn compressor_truncation_feasible(state: &CompressorState, ratio: usize, position: usize) -> bool {
    position / ratio == state.keys.len() || position.is_multiple_of(ratio)
}

/// Apply a feasible truncation (see [`compressor_truncation_feasible`]).
fn truncate_compressor(state: &mut CompressorState, ratio: usize, position: usize) {
    let blocks = position / ratio;
    if blocks < state.keys.len() {
        state.keys.truncate(blocks);
        state.values.truncate(blocks);
        state.pending.clear();
    } else {
        state.pending.truncate(position - blocks * ratio);
    }
}

/// HyperConnection.post: stream i gets post[i]*f plus the comb-mixed residual.
pub(crate) fn hc_post(
    f_out: &[f32],
    residual: &[Vec<f32>],
    post: &[f32],
    comb: &[f32],
) -> Vec<Vec<f32>> {
    let n = residual.len();
    let mut streams = Vec::with_capacity(n);
    for i in 0..n {
        let mut stream: Vec<f32> = f_out.iter().map(|value| post[i] * value).collect();
        for (j, source) in residual.iter().enumerate() {
            let weight = comb[i * n + j];
            for (out, value) in stream.iter_mut().zip(source) {
                *out += weight * value;
            }
        }
        streams.push(stream);
    }
    streams
}

/// Divide each comb column j by (Σ_i comb[i][j] + eps).
fn normalize_comb_columns(comb: &mut [f32], n: usize, eps: f32) {
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += comb[i * n + j];
        }
        let denom = sum + eps;
        for i in 0..n {
            comb[i * n + j] /= denom;
        }
    }
}

/// Divide each comb row i by (Σ_j comb[i][j] + eps).
fn normalize_comb_rows(comb: &mut [f32], n: usize, eps: f32) {
    for row in comb.chunks_mut(n) {
        let denom = row.iter().sum::<f32>() + eps;
        for value in row {
            *value /= denom;
        }
    }
}

/// The (sin, cos) pair per interleaved rope pair that [`v4_rope_tail`] would
/// compute at `pos` — the identical expressions (same `powf`/`sin_cos` libm
/// calls), so a device kernel fed this table rotates bit-identically to the
/// host. Consumed only by the GPU device step (one tiny upload per step).
#[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
pub(crate) fn v4_rope_sincos(
    rope_dims: usize,
    pos: usize,
    base: f32,
    inverse: bool,
) -> Vec<(f32, f32)> {
    let mut out = Vec::with_capacity(rope_dims / 2);
    for pair in 0..rope_dims / 2 {
        let freq = 1.0 / base.powf((2 * pair) as f32 / rope_dims as f32);
        let mut angle = pos as f32 * freq;
        if inverse {
            angle = -angle;
        }
        out.push(angle.sin_cos());
    }
    out
}

/// V4 rope on the trailing `rope_dims` of `values`: INTERLEAVED pairs
/// (x[2i], x[2i+1]) rotated by angle pos / base^(2i/rope_dims); `inverse`
/// negates the angle (used on the attention output tail). No YARN.
fn v4_rope_tail(values: &mut [f32], rope_dims: usize, pos: usize, base: f32, inverse: bool) {
    if rope_dims == 0 {
        return;
    }
    let start = values.len() - rope_dims;
    let tail = &mut values[start..];
    for pair in 0..rope_dims / 2 {
        let freq = 1.0 / base.powf((2 * pair) as f32 / rope_dims as f32);
        let mut angle = pos as f32 * freq;
        if inverse {
            angle = -angle;
        }
        let (sin, cos) = angle.sin_cos();
        let x0 = tail[2 * pair];
        let x1 = tail[2 * pair + 1];
        tail[2 * pair] = x0 * cos - x1 * sin;
        tail[2 * pair + 1] = x0 * sin + x1 * cos;
    }
}

fn require_dim(value: Option<u32>, key: &str) -> Result<usize> {
    let value = value.ok_or_else(|| anyhow!("deepseek4 GGUF missing {key}"))?;
    if value == 0 {
        bail!("deepseek4 GGUF {key} must be non-zero");
    }
    usize::try_from(value).with_context(|| format!("deepseek4 {key} does not fit usize"))
}

/// Validate presence and orientation of a raw matrix without dequantizing it.
/// deepseek4 GGUFs store weights as [ne0 = input, ne1 = output].
fn raw_matrix(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<RawMatrix> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
    let dims: Vec<u64> = info.dimensions.clone();
    if dims != [cols as u64, rows as u64] {
        bail!("tensor {name} has shape {dims:?}; expected [{cols}, {rows}]");
    }
    Ok(RawMatrix {
        name: name.to_string(),
        rows,
        cols,
    })
}

fn raw_experts(
    gguf: &GgufFile,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    experts: usize,
) -> Result<RawExperts> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
    let dims: Vec<u64> = info.dimensions.clone();
    if dims != [in_dim as u64, out_dim as u64, experts as u64] {
        bail!("tensor {name} has shape {dims:?}; expected [{in_dim}, {out_dim}, {experts}]");
    }
    Ok(RawExperts {
        name: name.to_string(),
        in_dim,
        out_dim,
    })
}

/// Load the compressor's additive positional bias `[out_dim, ratio]` as a flat
/// `[ratio][out_dim]` f32 buffer.
fn load_ape(gguf: &GgufFile, name: &str, out_dim: usize, ratio: usize) -> Result<Vec<f32>> {
    let view = gguf
        .tensor(name)
        .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
    let dims: Vec<u64> = view.info.dimensions.clone();
    if dims != [out_dim as u64, ratio as u64] {
        bail!("tensor {name} has shape {dims:?}; expected [{out_dim}, {ratio}]");
    }
    dequantize_elem_range(&view, 0, out_dim * ratio)
}

/// Dequantize an element subrange of a tensor. The offset/count must land on
/// block boundaries for block-quantized dtypes (always true here: every sliced
/// row width is a multiple of the 32-element block size).
pub(crate) fn dequantize_elem_range(
    view: &TensorView<'_>,
    elem_offset: usize,
    elem_count: usize,
) -> Result<Vec<f32>> {
    let dtype = view.info.dtype;
    let start = dtype
        .byte_len(elem_offset as u64)
        .with_context(|| format!("slicing tensor {} at element {elem_offset}", view.info.name))?;
    let len = dtype
        .byte_len(elem_count as u64)
        .with_context(|| format!("slicing {elem_count} elements of tensor {}", view.info.name))?;
    let start = usize::try_from(start).context("tensor byte offset does not fit usize")?;
    let len = usize::try_from(len).context("tensor byte length does not fit usize")?;
    let bytes = view
        .bytes
        .get(start..start + len)
        .ok_or_else(|| anyhow!("tensor {} slice is out of range", view.info.name))?;
    dequantize_tensor_as_f32(bytes, dtype, elem_count)
}

/// Tiny synthetic deepseek4 fixture shared by the CPU tests here and the GPU
/// parity test in `dsv4_gpu`.
#[cfg(test)]
pub(crate) mod fixture {
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Tiny synthetic deepseek4 GGUF: 3 layers (layer 0 ratio 0 + hash-routed,
    /// layer 1 ratio 4 with split-K|V compressor + indexer, layer 2 ratio 2
    /// with the shared-K=V compressor form), hc 2, sinkhorn 3, 2 heads of
    /// head_dim 8 (nope 4 + rope 4), q_lora 4, grouped output 2x4, 4 experts
    /// top-2 (ff 4) + 1 shared, sliding window 4, indexer top_k 1, vocab 3.
    pub(crate) fn write_deepseek4_gguf(path: &Path) {
        write_deepseek4_gguf_core(path, false, 64);
    }

    /// Speculative-decoding fixture variant: vocab 4 (`a b c d`) with eos on
    /// the extra token `d` and NO unknown-token fallback. The standard
    /// fixture's eos (`c`) and unknown (`a`) ids are delta-skipped by the
    /// streaming decoder, so the oracle suites could not observe emitted
    /// token ids through the event stream; here tokens 0..=2 all stream
    /// visibly and greedy only stops if it genuinely reaches `d`.
    // Consumed by the native-cuda backend's oracle suites only.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn write_deepseek4_spec_gguf(path: &Path) {
        write_deepseek4_gguf_core(path, true, 64);
    }

    /// [`write_deepseek4_gguf`] with an enlarged `deepseek4.context_length`,
    /// so long-context regimes (positions past the real model's 2048-token
    /// indexer-engagement boundary; hundreds of compressed blocks) run on
    /// fixture-scale weights. Same tensors, same tiny dims.
    // Consumed by the native-cuda device-verify long-context gate only.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn write_deepseek4_gguf_with_context(path: &Path, context: u32) {
        write_deepseek4_gguf_core(path, false, context);
    }

    fn write_deepseek4_gguf_core(path: &Path, spec_decode: bool, context: u32) {
        let vocab: usize = if spec_decode { 4 } else { 3 };
        let mut tensors = vec![
            tensor_f32(
                "token_embd.weight",
                vec![4, vocab as u64],
                &vals(1, 4 * vocab),
            ),
            tensor_f32("output.weight", vec![4, vocab as u64], &vals(2, 4 * vocab)),
            tensor_f32("output_norm.weight", vec![4], &[1.0; 4]),
            tensor_f32("output_hc_fn.weight", vec![8, 2], &vals(3, 16)),
            tensor_f32("output_hc_base.weight", vec![2], &vals(4, 2)),
            tensor_f32("output_hc_scale.weight", vec![1], &[0.7]),
        ];
        for layer in 0u32..3 {
            let p = format!("blk.{layer}");
            let seed = 10 + layer * 40;
            tensors.extend([
                tensor_f32(&format!("{p}.attn_norm.weight"), vec![4], &[1.0; 4]),
                tensor_f32(&format!("{p}.ffn_norm.weight"), vec![4], &[1.0; 4]),
                tensor_f32(
                    &format!("{p}.hc_attn_fn.weight"),
                    vec![8, 8],
                    &vals(seed, 64),
                ),
                tensor_f32(
                    &format!("{p}.hc_attn_base.weight"),
                    vec![8],
                    &vals(seed + 1, 8),
                ),
                tensor_f32(
                    &format!("{p}.hc_attn_scale.weight"),
                    vec![3],
                    &[0.6, 0.4, 0.8],
                ),
                tensor_f32(
                    &format!("{p}.hc_ffn_fn.weight"),
                    vec![8, 8],
                    &vals(seed + 2, 64),
                ),
                tensor_f32(
                    &format!("{p}.hc_ffn_base.weight"),
                    vec![8],
                    &vals(seed + 3, 8),
                ),
                tensor_f32(
                    &format!("{p}.hc_ffn_scale.weight"),
                    vec![3],
                    &[0.5, 0.7, 0.3],
                ),
                tensor_f32(
                    &format!("{p}.attn_q_a.weight"),
                    vec![4, 4],
                    &vals(seed + 4, 16),
                ),
                tensor_f32(&format!("{p}.attn_q_a_norm.weight"), vec![4], &[1.0; 4]),
                tensor_f32(
                    &format!("{p}.attn_q_b.weight"),
                    vec![4, 16],
                    &vals(seed + 5, 64),
                ),
                tensor_f32(
                    &format!("{p}.attn_kv.weight"),
                    vec![4, 8],
                    &vals(seed + 6, 32),
                ),
                tensor_f32(&format!("{p}.attn_kv_a_norm.weight"), vec![8], &[1.0; 8]),
                tensor_f32(
                    &format!("{p}.attn_sinks.weight"),
                    vec![2],
                    &vals(seed + 7, 2),
                ),
                tensor_f32(
                    &format!("{p}.attn_output_a.weight"),
                    vec![8, 8],
                    &vals(seed + 8, 64),
                ),
                tensor_f32(
                    &format!("{p}.attn_output_b.weight"),
                    vec![8, 4],
                    &vals(seed + 9, 32),
                ),
                tensor_f32(
                    &format!("{p}.ffn_gate_inp.weight"),
                    vec![4, 4],
                    &vals(seed + 10, 16),
                ),
                tensor_f32(
                    &format!("{p}.ffn_gate_exps.weight"),
                    vec![4, 4, 4],
                    &vals(seed + 11, 64),
                ),
                tensor_f32(
                    &format!("{p}.ffn_up_exps.weight"),
                    vec![4, 4, 4],
                    &vals(seed + 12, 64),
                ),
                tensor_f32(
                    &format!("{p}.ffn_down_exps.weight"),
                    vec![4, 4, 4],
                    &vals(seed + 13, 64),
                ),
                tensor_f32(
                    &format!("{p}.ffn_gate_shexp.weight"),
                    vec![4, 4],
                    &vals(seed + 14, 16),
                ),
                tensor_f32(
                    &format!("{p}.ffn_up_shexp.weight"),
                    vec![4, 4],
                    &vals(seed + 15, 16),
                ),
                tensor_f32(
                    &format!("{p}.ffn_down_shexp.weight"),
                    vec![4, 4],
                    &vals(seed + 16, 16),
                ),
            ]);
        }
        // Layer 0 is the hash-routed layer (top-2 experts per token id).
        tensors.push(tensor_i32(
            "blk.0.ffn_gate_tid2eid.weight",
            vec![2, 3],
            &[0, 1, 1, 2, 2, 3],
        ));
        // Layer 1 carries the learned-routing selection bias, the ratio-4
        // split-K|V compressor (gate/kv width 2*head_dim = 16), and the
        // indexer.
        tensors.extend([
            tensor_f32("blk.1.exp_probs_b.bias", vec![4], &vals(90, 4)),
            tensor_f32(
                "blk.1.attn_compressor_gate.weight",
                vec![4, 16],
                &vals(91, 64),
            ),
            tensor_f32(
                "blk.1.attn_compressor_kv.weight",
                vec![4, 16],
                &vals(92, 64),
            ),
            tensor_f32(
                "blk.1.attn_compressor_ape.weight",
                vec![16, 4],
                &vals(93, 64),
            ),
            tensor_f32("blk.1.attn_compressor_norm.weight", vec![8], &[1.0; 8]),
            tensor_f32("blk.1.indexer.attn_q_b.weight", vec![4, 8], &vals(94, 32)),
            tensor_f32("blk.1.indexer.proj.weight", vec![4, 2], &vals(95, 8)),
            tensor_f32(
                "blk.1.indexer_compressor_gate.weight",
                vec![4, 8],
                &vals(96, 32),
            ),
            tensor_f32(
                "blk.1.indexer_compressor_kv.weight",
                vec![4, 8],
                &vals(97, 32),
            ),
            tensor_f32(
                "blk.1.indexer_compressor_ape.weight",
                vec![8, 4],
                &vals(98, 32),
            ),
            tensor_f32("blk.1.indexer_compressor_norm.weight", vec![4], &[1.0; 4]),
        ]);
        // Layer 2 carries the shared-K=V compressor form (ratio 2, gate/kv
        // width == head_dim = 8, so the single compressed latent serves as
        // both K and V), mirroring the real GGUF's ratio-128 layers.
        tensors.extend([
            tensor_f32("blk.2.exp_probs_b.bias", vec![4], &vals(120, 4)),
            tensor_f32(
                "blk.2.attn_compressor_gate.weight",
                vec![4, 8],
                &vals(121, 32),
            ),
            tensor_f32(
                "blk.2.attn_compressor_kv.weight",
                vec![4, 8],
                &vals(122, 32),
            ),
            tensor_f32(
                "blk.2.attn_compressor_ape.weight",
                vec![8, 2],
                &vals(123, 16),
            ),
            tensor_f32("blk.2.attn_compressor_norm.weight", vec![8], &[1.0; 8]),
        ]);

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

        // Metadata is written through a counting writer so the header count
        // always matches (the fixture has ~38 keys).
        let mut meta = MetadataWriter::default();
        meta.string("general.architecture", "deepseek4");
        meta.string("general.name", "cpu-reference-deepseek4");
        meta.u32("general.alignment", 32);
        meta.u32("general.file_type", 1);
        meta.u32("deepseek4.context_length", context);
        meta.u32("deepseek4.embedding_length", 4);
        meta.u32("deepseek4.block_count", 3);
        meta.u32("deepseek4.attention.head_count", 2);
        meta.u32("deepseek4.attention.head_count_kv", 1);
        meta.u32("deepseek4.attention.key_length", 8);
        meta.u32("deepseek4.attention.value_length", 8);
        meta.u32("deepseek4.attention.q_lora_rank", 4);
        meta.f32("deepseek4.attention.layer_norm_rms_epsilon", 1.0e-6);
        meta.u32("deepseek4.rope.dimension_count", 4);
        meta.f32("deepseek4.rope.freq_base", 10_000.0);
        meta.f32("deepseek4.attention.compress_rope_freq_base", 160_000.0);
        meta.u32("deepseek4.attention.sliding_window", 4);
        meta.u32("deepseek4.attention.indexer.head_count", 2);
        meta.u32("deepseek4.attention.indexer.key_length", 4);
        meta.u32("deepseek4.attention.indexer.top_k", 1);
        meta.u32("deepseek4.attention.output_group_count", 2);
        meta.u32("deepseek4.attention.output_lora_rank", 4);
        // Trailing extra entry mirrors the real GGUF's stripped-MTP slot.
        meta.u32_array("deepseek4.attention.compress_ratios", &[0, 4, 2, 0]);
        meta.u32("deepseek4.hyper_connection.count", 2);
        meta.u32("deepseek4.hyper_connection.sinkhorn_iterations", 3);
        meta.f32("deepseek4.hyper_connection.epsilon", 1.0e-6);
        meta.u32("deepseek4.hash_layer_count", 1);
        meta.u32("deepseek4.expert_count", 4);
        meta.u32("deepseek4.expert_used_count", 2);
        meta.u32("deepseek4.expert_feed_forward_length", 4);
        meta.u32("deepseek4.expert_shared_count", 1);
        meta.boolean("deepseek4.expert_weights_norm", true);
        meta.u32("deepseek4.expert_gating_func", 4);
        meta.f32("deepseek4.expert_weights_scale", 1.5);
        meta.f32_array("deepseek4.swiglu_clamp_exp", &[10.0, 10.0, 10.0]);
        meta.f32_array("deepseek4.swiglu_clamp_shexp", &[10.0, 10.0, 10.0]);
        if spec_decode {
            meta.u32("tokenizer.ggml.eos_token_id", 3);
            meta.string_array("tokenizer.ggml.tokens", &["a", "b", "c", "d"]);
        } else {
            meta.u32("tokenizer.ggml.eos_token_id", 2);
            // Unknown fallback lets the serving smoke test encode arbitrary
            // prompt text (chat-template markers included) against the
            // 3-token vocab.
            meta.u32("tokenizer.ggml.unknown_token_id", 0);
            meta.string_array("tokenizer.ggml.tokens", &["a", "b", "c"]);
        }
        // Structural fragment of the real DeepSeek-V4-Flash chat template: just
        // enough for hi-local-core's V4 discriminator (`<｜User｜>` + `</think>`
        // + `｜DSML｜`) so the serving smoke test renders the V4 prompt shape.
        meta.string(
            "tokenizer.chat_template",
            "{%- set dsml_token = '｜DSML｜' -%}{%- set thinking_end_token = '</think>' -%}\
             {{- bos_token -}}{%- for message in messages -%}{{- '<｜User｜>' -}}\
             {{- message['content'] -}}{%- endfor -%}\
             {%- if add_generation_prompt -%}{{- '<｜Assistant｜>' -}}{{- thinking_end_token -}}{%- endif -%}",
        );

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, meta.count);
        bytes.extend_from_slice(&meta.bytes);

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

    /// Deterministic pseudo-random weights in roughly [-0.35, 0.35].
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

    struct TestTensor {
        name: String,
        dims: Vec<u64>,
        dtype: u32,
        offset: u64,
        bytes: Vec<u8>,
    }

    fn tensor_f32(name: &str, dims: Vec<u64>, values: &[f32]) -> TestTensor {
        assert_eq!(
            dims.iter().product::<u64>(),
            values.len() as u64,
            "fixture tensor {name} value count does not match dims"
        );
        TestTensor {
            name: name.to_string(),
            dims,
            dtype: 0,
            offset: 0,
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn tensor_i32(name: &str, dims: Vec<u64>, values: &[i32]) -> TestTensor {
        assert_eq!(
            dims.iter().product::<u64>(),
            values.len() as u64,
            "fixture tensor {name} value count does not match dims"
        );
        TestTensor {
            name: name.to_string(),
            dims,
            dtype: 26,
            offset: 0,
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    #[derive(Default)]
    struct MetadataWriter {
        bytes: Vec<u8>,
        count: u64,
    }

    impl MetadataWriter {
        fn string(&mut self, key: &str, value: &str) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 8);
            write_string(&mut self.bytes, value);
            self.count += 1;
        }

        fn u32(&mut self, key: &str, value: u32) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 4);
            write_u32(&mut self.bytes, value);
            self.count += 1;
        }

        fn f32(&mut self, key: &str, value: f32) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 6);
            self.bytes.extend_from_slice(&value.to_le_bytes());
            self.count += 1;
        }

        fn boolean(&mut self, key: &str, value: bool) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 7);
            self.bytes.push(value as u8);
            self.count += 1;
        }

        fn u32_array(&mut self, key: &str, values: &[u32]) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 9);
            write_u32(&mut self.bytes, 4);
            write_u64(&mut self.bytes, values.len() as u64);
            for value in values {
                write_u32(&mut self.bytes, *value);
            }
            self.count += 1;
        }

        fn f32_array(&mut self, key: &str, values: &[f32]) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 9);
            write_u32(&mut self.bytes, 6);
            write_u64(&mut self.bytes, values.len() as u64);
            for value in values {
                self.bytes.extend_from_slice(&value.to_le_bytes());
            }
            self.count += 1;
        }

        fn string_array(&mut self, key: &str, values: &[&str]) {
            write_string(&mut self.bytes, key);
            write_u32(&mut self.bytes, 9);
            write_u32(&mut self.bytes, 8);
            write_u64(&mut self.bytes, values.len() as u64);
            for value in values {
                write_string(&mut self.bytes, value);
            }
            self.count += 1;
        }
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

    pub(crate) fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-cuda-dsv4-cpu-{name}-{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{tempfile_path, write_deepseek4_gguf};
    use super::*;

    #[test]
    fn dsv4_cpu_reference_runs_greedy_and_deterministic() {
        let path = tempfile_path("tiny");
        write_deepseek4_gguf(&path);

        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        assert!(model.config().is_deepseek4());
        assert_eq!(model.config().expert_count, Some(4));
        assert_eq!(model.config().hash_layer_count, Some(1));
        assert_eq!(model.config().hyper_connection_count, Some(2));

        let options = QwenCpuRunOptions {
            max_tokens: 3,
            top_k: 2,
            include_logits: true,
            ..QwenCpuRunOptions::default()
        };
        let first = model.run_tokens(&[0, 1, 2], options.clone()).unwrap();
        let second = model.run_tokens(&[0, 1, 2], options).unwrap();

        assert_eq!(first.backend, "cpu-reference");
        assert_eq!(first.input_tokens, vec![0, 1, 2]);
        assert_eq!(first.logit_count, 3);
        assert_eq!(first.top_logits.len(), 2);
        let logits = first.logits.as_ref().unwrap();
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|logit| logit.is_finite()));
        // Greedy: the first generated token is the reported argmax; eos may end
        // the continuation early.
        assert!(!first.generated_tokens.is_empty());
        assert!(first.generated_tokens.len() <= 3);
        assert_eq!(first.generated_tokens[0], first.next_token);
        // Deterministic across runs (bit-exact logits included).
        assert_eq!(first.next_token, second.next_token);
        assert_eq!(first.generated_tokens, second.generated_tokens);
        assert_eq!(first.logits, second.logits);
        assert_eq!(first.top_logits[0].token_id, second.top_logits[0].token_id);
    }

    #[test]
    fn dsv4_long_sequence_exercises_ring_compressor_and_indexer() {
        let path = tempfile_path("long");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();

        // 14 tokens: the raw ring (window 4) evicts from token 5 on; the
        // ratio-4 layer holds 3 compressed blocks by token 12; the indexer
        // (top_k 1) starts gathering a block subset once 2 blocks exist.
        let sequence: Vec<u32> = (0..14).map(|idx| idx % 3).collect();
        let first = model.last_logits(&sequence).unwrap();
        let second = model.last_logits(&sequence).unwrap();

        assert_eq!(first.len(), 3);
        assert!(first.iter().all(|logit| logit.is_finite()));
        assert_eq!(first, second);
    }

    #[test]
    fn dsv4_streaming_state_builds_blocks_and_ring() {
        let path = tempfile_path("state");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();

        let mut state = model.new_state();
        for idx in 0..10u32 {
            model.step(&mut state, idx % 3).unwrap();
        }

        // Layer 0: ratio 0 — raw ring only, capped at the window.
        let layer0 = &state.layers[0];
        assert_eq!(layer0.ring.len(), 4);
        assert!(layer0.compressor.is_none());
        assert!(layer0.indexer.is_none());
        // Layer 1: ratio 4 — 10 tokens give 2 complete blocks + 2 pending, and
        // the indexer's private compressor stays in lockstep.
        let layer1 = &state.layers[1];
        assert_eq!(layer1.ring.len(), 4);
        let compressor = layer1.compressor.as_ref().unwrap();
        assert_eq!(compressor.keys.len(), 2);
        assert_eq!(compressor.values.len(), 2);
        assert_eq!(compressor.pending.len(), 2);
        let indexer = layer1.indexer.as_ref().unwrap();
        assert_eq!(indexer.keys.len(), 2);
        assert_eq!(indexer.keys[0].len(), 4);
        assert_eq!(compressor.keys[0].len(), 8);
    }

    #[test]
    fn dsv4_shared_kv_compressor_builds_joint_blocks() {
        let path = tempfile_path("shared");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();

        // Layer 2 is the ratio-2 shared-latent form (gate/kv width ==
        // head_dim): 7 tokens complete 3 blocks with 1 pending, and each
        // block's K and V must be the SAME normalized vector — unlike layer
        // 1's split form, whose halves differ.
        let mut state = model.new_state();
        for idx in 0..7u32 {
            model.step(&mut state, idx % 3).unwrap();
        }

        let layer2 = &state.layers[2];
        assert!(layer2.indexer.is_none(), "indexer is ratio-4 only");
        let shared = layer2.compressor.as_ref().unwrap();
        assert_eq!(shared.keys.len(), 3);
        assert_eq!(shared.pending.len(), 1);
        assert_eq!(shared.keys[0].len(), 8);
        assert_eq!(shared.keys, shared.values);

        let split = state.layers[1].compressor.as_ref().unwrap();
        assert_eq!(split.keys.len(), 1);
        assert_ne!(split.keys, split.values);
    }

    /// Parity gate for the batched prefill on the CPU provider: chunked
    /// prefill (including chunk sizes that split compressor blocks across
    /// chunk boundaries) must reproduce the sequential path bit-for-bit —
    /// the default `mul_mat` loops `mul_vec` and every host op is shared, so
    /// anything short of equality is a chunking bug.
    #[test]
    fn dsv4_chunked_prefill_matches_sequential() {
        let path = tempfile_path("chunked");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;

        // 14 tokens engage ring eviction (window 4), both compressor forms
        // (ratio 4 split and ratio 2 shared), and the indexer's top-k
        // narrowing (top_k 1, blocks visible from token 8 on).
        let tokens: Vec<u32> = (0..14).map(|idx| idx % 3).collect();
        let mut sequential_state = engine.new_state();
        let sequential = engine
            .prefill_with_chunk(&mut sequential_state, &tokens, 1)
            .unwrap();

        for chunk in [3, 4, 14, 64] {
            let mut state = engine.new_state();
            let chunked = engine
                .prefill_with_chunk(&mut state, &tokens, chunk)
                .unwrap();
            assert_eq!(state.pos, tokens.len());
            for (idx, (seq, chk)) in sequential.iter().zip(&chunked).enumerate() {
                assert!(
                    (seq - chk).abs() <= 1.0e-4,
                    "chunk {chunk} logit[{idx}]: sequential {seq} vs chunked {chk}"
                );
            }
            // Greedy continuation from the chunked state must match the
            // sequential state's token-for-token.
            let mut seq_state = sequential_state.clone();
            let mut seq_logits = sequential.clone();
            let mut chk_logits = chunked;
            for step in 0..4 {
                let seq_next = argmax(&seq_logits).unwrap();
                let chk_next = argmax(&chk_logits).unwrap();
                assert_eq!(seq_next, chk_next, "chunk {chunk} greedy step {step}");
                seq_logits = engine.step(&mut seq_state, seq_next).unwrap();
                chk_logits = engine.step(&mut state, chk_next).unwrap();
            }
        }
    }

    /// Ring slack only extends retention: attention reads the trailing window
    /// either way, so logits are bit-identical while the retained ring grows.
    #[test]
    fn dsv4_ring_slack_preserves_logits_and_extends_retention() {
        let path = tempfile_path("slack");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;
        let tokens: Vec<u32> = (0..12).map(|idx| idx % 3).collect();

        let mut plain = engine.new_state();
        let plain_logits = engine.prefill_with_chunk(&mut plain, &tokens, 1).unwrap();
        let mut slacked = engine.new_state_with_ring_slack(6);
        let slack_logits = engine.prefill_with_chunk(&mut slacked, &tokens, 4).unwrap();

        assert_eq!(plain_logits, slack_logits);
        // Window 4: the plain state keeps 4 entries, the slacked one 4+6.
        assert_eq!(plain.layers[0].ring.len(), 4);
        assert_eq!(slacked.layers[0].ring.len(), 10);
    }

    /// Measures the host footprint of block-boundary snapshots (the serving
    /// backend's prefix-cache LRU budget unit). Grows monotonically with
    /// position as compressed blocks accumulate; prints the fixture numbers.
    #[test]
    fn dsv4_snapshot_bytes_grows_with_position() {
        let path = tempfile_path("snapbytes");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;

        let mut sizes = Vec::new();
        for &n in &[4usize, 8, 16, 32, 48] {
            let tokens: Vec<u32> = (0..n as u32).map(|idx| idx % 3).collect();
            let mut state = engine.new_state();
            engine.prefill_with_chunk(&mut state, &tokens, 4).unwrap();
            assert_eq!(state.pos(), n);
            sizes.push((n, state.snapshot_bytes()));
        }
        eprintln!("dsv4 fixture snapshot_bytes by position: {sizes:?}");
        assert!(sizes[0].1 > 0, "even the shortest snapshot has a footprint");
        for pair in sizes.windows(2) {
            assert!(
                pair[1].1 >= pair[0].1,
                "snapshot bytes must not shrink with length: {sizes:?}"
            );
        }
    }

    /// Truncating a slack-retaining state back to an earlier position must
    /// reproduce a fresh prefill of that prefix exactly (bit-identical next
    /// logits), and infeasible mid-block positions must round down to the
    /// nearest reconstructable one.
    #[test]
    fn dsv4_truncate_state_matches_fresh_prefill() {
        let path = tempfile_path("truncate");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;
        let tokens: Vec<u32> = (0..14).map(|idx| (idx * 2) % 3).collect();

        let mut full = engine.new_state_with_ring_slack(8);
        engine.prefill_with_chunk(&mut full, &tokens, 4).unwrap();

        // Position 13 is mid-block for the ratio-2 layer (13/2=6 < 7 complete
        // blocks) and not a boundary, so truncation rounds down to 12 (a
        // multiple of both ratios that also sits inside layer 1's pending
        // block: 12/4 == 14/4 == 3).
        let mut truncated = full.clone();
        assert_eq!(
            engine.truncate_state_to_at_most(&mut truncated, 13),
            Some(12)
        );
        assert_eq!(truncated.pos(), 12);

        let mut fresh = engine.new_state_with_ring_slack(8);
        engine
            .prefill_with_chunk(&mut fresh, &tokens[..12], 1)
            .unwrap();
        // The truncated state must continue exactly like the fresh prefix.
        for &next in &[tokens[12], 1, 2] {
            let from_truncated = engine.step(&mut truncated, next).unwrap();
            let from_fresh = engine.step(&mut fresh, next).unwrap();
            assert_eq!(from_truncated, from_fresh);
        }

        // A state without slack cannot rewind past its evicted ring entries:
        // window 4 means position 12 needs entries 8..11, but after 14 tokens
        // a slack-0 ring only holds 10..13.
        let mut no_slack = engine.new_state();
        engine
            .prefill_with_chunk(&mut no_slack, &tokens, 4)
            .unwrap();
        assert_eq!(engine.truncate_state_to_at_most(&mut no_slack, 13), None);
        // Truncating to the current position is always feasible.
        assert_eq!(
            engine.truncate_state_to_at_most(&mut no_slack, 14),
            Some(14)
        );
    }

    /// Stage-A verify contract on the CPU provider
    /// (`docs/deepseek-v4-spec-decode-plan.md`): `verify_tokens` returns
    /// logits at EVERY position, bit-identical to stepping the same tokens
    /// through `host_step` one at a time, and advances the state identically
    /// (a continued decode agrees bit for bit).
    #[test]
    fn dsv4_verify_tokens_bit_exact_with_sequential_steps() {
        let path = tempfile_path("verify");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;

        let prompt: Vec<u32> = (0..6).map(|idx| idx % 3).collect();
        // The continuation crosses ratio-4 and ratio-2 block boundaries,
        // evicts from the window-4 ring, and engages the indexer's top-k.
        let continuation: Vec<u32> = (0..8).map(|idx| (idx * 2) % 3).collect();

        let mut seq_state = engine.new_state();
        engine
            .prefill_with_chunk(&mut seq_state, &prompt, 1)
            .unwrap();
        let mut verify_state = seq_state.clone();

        let seq_rows: Vec<Vec<f32>> = continuation
            .iter()
            .map(|&token| engine.host_step(&mut seq_state, token).unwrap())
            .collect();
        let verify_rows = engine
            .verify_tokens(&mut verify_state, &continuation)
            .unwrap();
        assert_eq!(verify_rows.len(), continuation.len());
        assert_eq!(
            verify_rows, seq_rows,
            "verify logits must be bit-exact with sequential host steps"
        );

        assert_eq!(verify_state.pos(), seq_state.pos());
        for &token in &[0u32, 1, 2] {
            assert_eq!(
                engine.host_step(&mut verify_state, token).unwrap(),
                engine.host_step(&mut seq_state, token).unwrap(),
                "continued decode after verify must stay bit-exact"
            );
        }
    }

    /// Stage-A rollback: a verify overshoots the accepted prefix, then
    /// [`DsV4Engine::rewind_state_to`] restores the accepted end exactly —
    /// including compressor block-boundary round-down re-feeds and the
    /// no-retention full-rebuild fallback — so a continued decode is
    /// bit-identical to a state that only ever stepped the accepted tokens.
    #[test]
    fn dsv4_verify_rewind_round_trip_matches_accepted_only_state() {
        let path = tempfile_path("rewind");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;

        let prompt: Vec<u32> = (0..6).map(|idx| idx % 3).collect();
        let continuation: Vec<u32> = vec![1, 2, 0, 1, 2];
        let mut history = prompt.clone();
        history.extend(&continuation);
        // The backend's speculative sizing: max compress ratio 4 + K + 1.
        let slack = 4 + continuation.len() + 1;

        for keep in 1..=continuation.len() {
            let target = prompt.len() + keep;
            let mut state = engine.new_state_with_ring_slack(slack);
            engine.prefill_with_chunk(&mut state, &prompt, 4).unwrap();
            let rows = engine.verify_tokens(&mut state, &continuation).unwrap();
            assert_eq!(rows.len(), continuation.len());
            engine
                .rewind_state_to(&mut state, &history[..target], target, None)
                .unwrap();
            assert_eq!(state.pos(), target);

            // Reference: a state that only ever processed the kept prefix.
            let mut reference = engine.new_state_with_ring_slack(slack);
            engine
                .prefill_with_chunk(&mut reference, &prompt, 4)
                .unwrap();
            for &token in &continuation[..keep] {
                engine.host_step(&mut reference, token).unwrap();
            }
            for &token in &[2u32, 0, 1] {
                assert_eq!(
                    engine.host_step(&mut state, token).unwrap(),
                    engine.host_step(&mut reference, token).unwrap(),
                    "keep {keep}: rewound state diverged from accepted-only state"
                );
            }
        }

        // No retention at all (slack 0, ring evicted past the rewind point):
        // rewind falls back to the full rebuild and still matches.
        let mut state = engine.new_state();
        engine.prefill_with_chunk(&mut state, &prompt, 4).unwrap();
        engine.verify_tokens(&mut state, &continuation).unwrap();
        let target = prompt.len() + 1;
        engine
            .rewind_state_to(&mut state, &history[..target], target, None)
            .unwrap();
        assert_eq!(state.pos(), target);
        let mut reference = engine.new_state();
        engine
            .prefill_with_chunk(&mut reference, &history[..target], 4)
            .unwrap();
        for &token in &[2u32, 0, 1] {
            assert_eq!(
                engine.host_step(&mut state, token).unwrap(),
                engine.host_step(&mut reference, token).unwrap(),
                "full-rebuild rewind diverged"
            );
        }
    }

    /// Stage-A hidden taps: chunked prefill, single steps, and verify chunks
    /// capture bit-identical rows to a fully sequential tapped run; the
    /// captured pre-hc-head residual reproduces its position's logits through
    /// the output head (direct computation); averaged views are the stream
    /// means; and disabled/empty configs capture nothing while never
    /// perturbing logits.
    #[test]
    fn dsv4_taps_capture_matches_sequential_and_direct_computation() {
        let path = tempfile_path("taps");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;
        let (hc, embed) = (engine.geometry.hc, engine.geometry.embed);

        let config = DsV4TapConfig {
            pre_hc_head: true,
            aux_layers: vec![2, 0], // unsorted on purpose; new_taps normalizes
        };
        let mut taps = engine.new_taps(config.clone()).unwrap();
        assert_eq!(taps.flat_width(), hc * embed);

        // Mixed capture paths, the drafter-context pattern: chunked prefill,
        // then single tapped steps, then a verify chunk.
        let tokens: Vec<u32> = (0..12).map(|idx| idx % 3).collect();
        let mut state = engine.new_state_with_ring_slack(9);
        let mut step_logits = engine
            .prefill_with_chunk_taps(&mut state, &tokens[..8], 4, Some(&mut taps))
            .unwrap();
        for &token in &tokens[8..10] {
            step_logits = engine
                .step_with_taps(&mut state, token, Some(&mut taps))
                .unwrap();
        }
        let verify_rows = engine
            .verify_tokens_with_taps(&mut state, &tokens[10..12], Some(&mut taps))
            .unwrap();
        assert_eq!(taps.positions(), tokens.len());

        // Oracle: the same sequence stepped one token at a time with taps.
        let mut seq_taps = engine.new_taps(config).unwrap();
        let mut seq_state = engine.new_state_with_ring_slack(9);
        let mut seq_logits = Vec::new();
        for &token in &tokens {
            seq_logits.push(
                engine
                    .step_with_taps(&mut seq_state, token, Some(&mut seq_taps))
                    .unwrap(),
            );
        }
        for position in 0..tokens.len() {
            assert_eq!(
                taps.pre_hc_head(position).unwrap(),
                seq_taps.pre_hc_head(position).unwrap(),
                "pre-hc-head row {position}"
            );
            for layer in [0usize, 2] {
                assert_eq!(
                    taps.aux_flat(layer, position).unwrap(),
                    seq_taps.aux_flat(layer, position).unwrap(),
                    "aux layer {layer} row {position}"
                );
            }
        }
        // Capture never perturbs logits, on any path.
        assert_eq!(step_logits, seq_logits[9]);
        assert_eq!(verify_rows[0], seq_logits[10]);
        assert_eq!(verify_rows[1], seq_logits[11]);

        // Direct computation: the captured pre-hc-head residual is the value
        // immediately before the output head, so pushing it through
        // hyper-head collapse + final norm + lm head reproduces the logits.
        let flat = taps.pre_hc_head(11).unwrap();
        let streams: Vec<Vec<f32>> = flat.chunks(embed).map(<[f32]>::to_vec).collect();
        assert_eq!(streams.len(), hc);
        let mut hidden = engine.hyper_head(&streams).unwrap();
        rms_norm_in_place(&mut hidden, &engine.output_norm, engine.rms_eps).unwrap();
        let expected = engine.matvec(&engine.output_head, &hidden).unwrap();
        assert_eq!(verify_rows[1], expected);

        // Averaged view = arithmetic mean over the hc streams.
        let avg = taps.aux_averaged(2, 5).unwrap();
        let flat = taps.aux_flat(2, 5).unwrap();
        assert_eq!(avg.len(), embed);
        for (channel, value) in avg.iter().enumerate() {
            let want = (flat[channel] + flat[embed + channel]) / 2.0;
            assert!((value - want).abs() < 1.0e-6, "avg channel {channel}");
        }

        // Unrequested layers yield nothing; truncation drops rows in lockstep
        // with a state rewind.
        assert!(taps.aux_flat(1, 0).is_none());
        taps.truncate(7);
        assert_eq!(taps.positions(), 7);
        assert!(taps.pre_hc_head(7).is_none());
        assert!(taps.aux_flat(0, 6).is_some());

        // Empty config: nothing captured, logits identical to the untapped
        // run; out-of-range aux layers are rejected at construction.
        let mut empty = engine.new_taps(DsV4TapConfig::default()).unwrap();
        let mut plain_state = engine.new_state();
        let plain = engine
            .prefill_with_chunk(&mut plain_state, &tokens, 4)
            .unwrap();
        let mut tapped_state = engine.new_state();
        let tapped = engine
            .prefill_with_chunk_taps(&mut tapped_state, &tokens, 4, Some(&mut empty))
            .unwrap();
        assert_eq!(plain, tapped);
        assert_eq!(empty.positions(), tokens.len());
        assert!(empty.pre_hc_head(0).is_none());
        assert!(
            engine
                .new_taps(DsV4TapConfig {
                    pre_hc_head: false,
                    aux_layers: vec![3],
                })
                .is_err()
        );
    }

    /// Base-offset taps (prefix-cache-restore view): a buffer attached at a
    /// non-zero base captures rows for absolute positions `base..` that are
    /// bit-identical to a full buffer's rows there, returns `None` below the
    /// base (never a misaligned row), truncates/rebases in absolute terms,
    /// and stays position-aligned through verify + rewind.
    #[test]
    fn dsv4_taps_base_offset_accessors_truncate_and_rewind_alignment() {
        let path = tempfile_path("taps-base");
        write_deepseek4_gguf(&path);
        let model = DeepSeekV4CpuReference::load(&path).unwrap();
        let engine = &model.engine;
        let config = DsV4TapConfig {
            pre_hc_head: true,
            aux_layers: vec![1],
        };
        let tokens: Vec<u32> = (0..12).map(|idx| idx % 3).collect();
        let base = 5usize;

        // Full-history reference buffer.
        let mut full = engine.new_taps(config.clone()).unwrap();
        let mut full_state = engine.new_state();
        engine
            .prefill_with_chunk_taps(&mut full_state, &tokens, 4, Some(&mut full))
            .unwrap();

        // Restore-shaped buffer: positions 0..base forwarded untapped (the
        // restored prefix), capture attached at `base`.
        let mut taps = engine.new_taps_at(config.clone(), base).unwrap();
        let mut state = engine.new_state_with_ring_slack(9);
        engine
            .prefill_with_chunk(&mut state, &tokens[..base], 4)
            .unwrap();
        engine
            .prefill_with_chunk_taps(&mut state, &tokens[base..], 4, Some(&mut taps))
            .unwrap();

        assert_eq!(taps.base(), base);
        assert_eq!(taps.positions(), tokens.len());
        // Absolute accessors: None strictly below the base, bit-identical to
        // the full buffer at and above it.
        for position in 0..base {
            assert!(taps.pre_hc_head(position).is_none(), "row {position}");
            assert!(taps.aux_flat(1, position).is_none(), "aux row {position}");
            assert!(taps.aux_averaged(1, position).is_none());
        }
        for position in base..tokens.len() {
            assert_eq!(
                taps.pre_hc_head(position).unwrap(),
                full.pre_hc_head(position).unwrap(),
                "pre-hc-head row {position}"
            );
            assert_eq!(
                taps.aux_flat(1, position).unwrap(),
                full.aux_flat(1, position).unwrap(),
                "aux row {position}"
            );
        }
        assert!(taps.pre_hc_head(tokens.len()).is_none());

        // Absolute truncate: to mid-range, then to (and below) the base.
        taps.truncate(8);
        assert_eq!(taps.positions(), 8);
        assert!(taps.pre_hc_head(8).is_none());
        assert_eq!(
            taps.pre_hc_head(7).unwrap(),
            full.pre_hc_head(7).unwrap(),
            "truncate must keep absolute alignment"
        );
        taps.truncate(2);
        assert_eq!(taps.base(), base, "truncate never moves the base");
        assert_eq!(taps.positions(), base, "emptied buffer ends at its base");
        assert!(taps.pre_hc_head(base).is_none());

        // Rebase restarts capture at a new absolute position.
        taps.rebase(2);
        assert_eq!((taps.base(), taps.positions()), (2, 2));

        // Verify + rewind keep a based buffer aligned: rows for rejected
        // positions drop, the survivors still match the full reference.
        let mut taps = engine.new_taps_at(config, base).unwrap();
        let mut state = engine.new_state_with_ring_slack(9);
        engine
            .prefill_with_chunk(&mut state, &tokens[..base], 4)
            .unwrap();
        engine
            .prefill_with_chunk_taps(&mut state, &tokens[base..], 4, Some(&mut taps))
            .unwrap();
        let continuation = [1u32, 2, 0, 1];
        engine
            .verify_tokens_with_taps(&mut state, &continuation, Some(&mut taps))
            .unwrap();
        assert_eq!(taps.positions(), tokens.len() + continuation.len());
        let mut history = tokens.clone();
        history.extend(&continuation);
        let target = tokens.len() + 2;
        engine
            .rewind_state_to(&mut state, &history, target, Some(&mut taps))
            .unwrap();
        assert_eq!(state.pos(), target);
        assert_eq!(taps.positions(), target, "taps track the rewound state");
        assert_eq!(taps.base(), base);
        // The surviving verify rows equal a full-history recompute.
        let mut full2 = engine
            .new_taps(DsV4TapConfig {
                pre_hc_head: true,
                aux_layers: vec![1],
            })
            .unwrap();
        let mut ref_state = engine.new_state();
        engine
            .prefill_with_chunk_taps(&mut ref_state, &history[..target], 4, Some(&mut full2))
            .unwrap();
        for position in base..target {
            assert_eq!(
                taps.pre_hc_head(position).unwrap(),
                full2.pre_hc_head(position).unwrap(),
                "post-rewind row {position}"
            );
        }
    }

    #[test]
    fn dsv4_rope_inverse_restores_tail() {
        let original = vec![0.3, -0.7, 1.2, 0.05, 0.9, -0.4, 0.2, 0.8];
        let mut values = original.clone();

        v4_rope_tail(&mut values, 4, 9, 160_000.0, false);
        assert_eq!(values[..4], original[..4]);
        assert!(values[4..] != original[4..]);
        v4_rope_tail(&mut values, 4, 9, 160_000.0, true);

        for (actual, expected) in values.iter().zip(&original) {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "rope roundtrip mismatch: {actual} vs {expected}"
            );
        }
    }
}
