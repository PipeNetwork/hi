//! DFlash block-diffusion drafter for DeepSeek-V4-Flash — Stage C of
//! `docs/deepseek-v4-spec-decode-plan.md`.
//!
//! Owns: loading the RedHat speculator checkpoint (via
//! [`crate::safetensors`]), the 5-layer SWA drafter forward (fc aux-hidden
//! combiner, context-KV precompute from target taps, mask-token block
//! drafting, d2t vocab mapping), and the [`crate::dsv4_backend::Drafter`]
//! implementation selected by `HI_DSV4_SPEC=dflash`.
//!
//! Ported from vLLM's `qwen3_dflash.py` + `spec_decode/dflash.py`:
//!
//! - **Conditioning**: per verified context position, the target's post-layer
//!   hc-stream hiddens at the five aux layers are concatenated in layer order
//!   (5 x hc_mult*4096 = 81920 wide) and pushed through `fc` (raw concat in —
//!   vLLM `combine_hidden_states`); `hidden_norm` applies only afterwards on
//!   the context-KV path (`precompute_and_store_context_kv`).
//! - **Context KV**: the normed fc output is projected by each draft layer's
//!   own `k_proj`/`v_proj` (per-head `k_norm` + rope at the absolute position
//!   on K) and appended to that layer's KV cache. The cache holds CONTEXT
//!   entries derived from target hiddens, never draft self-attention outputs,
//!   and grows incrementally as the verify loop accepts tokens (taps are
//!   truncated after rejections, so captured positions never contain rejected
//!   tokens).
//! - **Query pass**: one forward per proposal regardless of context length —
//!   1 + K query tokens `[anchor, mask, mask, ...]` (the pending token is the
//!   anchor, embedded by id; the mask embedding is `embed_tokens[1]`, there
//!   is no dedicated mask tensor) at absolute positions `p .. p + K`, running
//!   the 5 llama-style layers against context KV plus the in-block query KV
//!   under the checkpoint's causal SWA-2048 mask. Draft logits are the
//!   `lm_head` rows at the K mask positions, argmaxed per row.
//! - **Vocab map**: `d2t` is an i64 OFFSET table — emitted target id =
//!   `draft_id + d2t[draft_id]` (census-verified); `embed_tokens` spans the
//!   full target vocab so anchor/context ids need no mapping.
//!
//! Execution split: every fat GEMM (fc, q/k/v/o, gate/up/down, lm_head, and
//! the attention score/value products) runs through a [`DfLinear`] provider —
//! resident-bf16 cuBLAS on the GPU, plain f32 loops for the CPU reference —
//! while the tiny per-row math (RMS norms, rope, SWA masking + softmax over
//! at most `window + block` keys, SwiGLU, residuals, argmax, d2t) stays in
//! host f32 shared verbatim between both providers, so CPU==GPU parity is a
//! pure matmul-precision comparison.
#![cfg_attr(all(not(feature = "native-cuda"), not(test)), allow(dead_code))]

use std::path::Path;
#[cfg(any(test, feature = "native-cuda"))]
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};

use crate::safetensors::{SafetensorsDtype, SafetensorsFile, bf16_to_f32, f16_to_f32};

#[cfg(feature = "native-cuda")]
use crate::dsv4_backend::{DraftContext, Drafter};
#[cfg(feature = "native-cuda")]
use crate::dsv4_cpu::{DsV4TapConfig, DsV4Taps};
#[cfg(feature = "native-cuda")]
use crate::dsv4_gpu::DeepSeekV4GpuEngine;
#[cfg(feature = "native-cuda")]
use crate::runtime::{Cublas, DeviceBuffer, GemmDType, Stream};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Target-model layers (OUR tap indexing: post-0-based-layer) whose
/// hc-stream hiddens condition the drafter.
///
/// Indexing: the checkpoint's `aux_hidden_state_layer_ids` are
/// `[3,13,23,32,42]` in vLLM's convention, where the target records aux id
/// `n` as "the hidden after `n` layers have executed" (`if idx + 1 in
/// aux_hidden_state_layers` after 0-based layer `idx`) — i.e. the outputs of
/// 0-based layers `[2,12,22,31,41]`, exactly vLLM `update_dflash`'s
/// `target_layer_ids = [i - 1 for i in aux_layer_ids]`. Our engine taps are
/// post-0-based-layer indexed, so they take the converted values. The
/// real-model e2e A/B confirms the conversion: `[2,12,22,31,41]` accepts
/// 16-18% of drafts vs 10-16% for the unshifted `[3,13,23,32,42]`. The
/// `HI_DSV4_DFLASH_AUX` env override (comma-separated raw tap indices, same
/// count as trained) re-runs that experiment without a rebuild.
const DFLASH_DEFAULT_AUX_LAYERS: [usize; 5] = [2, 12, 22, 31, 41];

/// Default checkpoint location under `$HOME` when `HI_DSV4_DFLASH_PATH` is
/// not set.
#[cfg(feature = "native-cuda")]
const DFLASH_DEFAULT_SUFFIX: &str = ".hi/models/deepseek-v4-flash/dflash-redhat/model.safetensors";

/// Upper bound on one uploaded lhs chunk in the cuBLAS provider (bounds the
/// transient device allocation when fc processes a whole prompt's context in
/// one call).
#[cfg(feature = "native-cuda")]
const DFLASH_GEMM_CHUNK_BYTES: usize = 48 << 20;

/// Drafter geometry + attention semantics. Baked defaults match the
/// census-verified RedHat checkpoint; `apply_config_json` overrides them from
/// the sibling `config.json` when present, and the loader validates every
/// tensor shape against the final values.
#[derive(Clone, Debug)]
struct DFlashConfig {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    /// `embed_tokens` rows (full target vocab).
    embed_vocab: usize,
    /// `lm_head` rows (reduced draft vocab).
    draft_vocab: usize,
    /// Target layers to tap, strictly ascending (concat order = this order).
    aux_layers: Vec<usize>,
    /// Flat width of one tap row (target hc_mult * target hidden).
    aux_width: usize,
    rope_theta: f32,
    rms_eps: f32,
    /// Sliding-window size; `None` = full attention.
    window: Option<usize>,
    /// Causal mask flag (vLLM speculators: `!sliding_window_non_causal`).
    causal: bool,
    /// Draft block size; proposals draft `block_size - 1` tokens.
    block_size: usize,
    mask_token: u32,
    #[allow(dead_code)]
    max_anchors: usize,
}

impl DFlashConfig {
    /// Census-verified values for the RedHat V4-Flash speculator.
    fn real_default() -> Self {
        Self {
            hidden: 4096,
            layers: 5,
            heads: 64,
            kv_heads: 1,
            head_dim: 256,
            intermediate: 2048,
            embed_vocab: 129_280,
            draft_vocab: 32_000,
            aux_layers: DFLASH_DEFAULT_AUX_LAYERS.to_vec(),
            aux_width: 4 * 4096,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            window: Some(2048),
            causal: true,
            block_size: 8,
            mask_token: 1,
            max_anchors: 3072,
        }
    }

    fn fc_in(&self) -> usize {
        self.aux_width * self.aux_layers.len()
    }

    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn heads_per_kv(&self) -> usize {
        self.heads / self.kv_heads
    }

    fn validate(&self) -> Result<()> {
        if self.hidden == 0
            || self.layers == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.intermediate == 0
            || self.draft_vocab == 0
        {
            bail!("DFlash config has a zero dimension: {self:?}");
        }
        if self.heads % self.kv_heads != 0 {
            bail!(
                "DFlash heads {} not divisible by kv heads {}",
                self.heads,
                self.kv_heads
            );
        }
        if self.head_dim % 2 != 0 {
            bail!("DFlash head_dim {} must be even for rope", self.head_dim);
        }
        if self.aux_layers.is_empty() || self.aux_width == 0 {
            bail!("DFlash aux conditioning is empty: {self:?}");
        }
        if !self.aux_layers.windows(2).all(|w| w[0] < w[1]) {
            bail!(
                "DFlash aux layers {:?} must be strictly ascending (fc consumes the concat in layer order)",
                self.aux_layers
            );
        }
        if self.window == Some(0) {
            bail!("DFlash sliding window must be >= 1");
        }
        if self.block_size < 2 {
            bail!(
                "DFlash block_size {} leaves no draft slots",
                self.block_size
            );
        }
        if (self.mask_token as usize) >= self.embed_vocab {
            bail!(
                "DFlash mask_token_id {} outside embed vocab {}",
                self.mask_token,
                self.embed_vocab
            );
        }
        if !(self.rope_theta > 0.0) || !(self.rms_eps > 0.0) {
            bail!(
                "DFlash rope_theta/rms_norm_eps must be positive, got {} / {}",
                self.rope_theta,
                self.rms_eps
            );
        }
        Ok(())
    }

    /// Override the baked defaults from the checkpoint's `config.json`
    /// (speculators layout: drafter fields at the top level, llama-style
    /// geometry under `transformer_layer_config`). Mirrors vLLM
    /// `update_dflash` + `_resolve_layer_attention`: the causal flag is
    /// `!sliding_window_non_causal` (default true -> non-causal), all-SWA vs
    /// all-full comes from `layer_types`, and `aux_width` is
    /// `hc_mult * hidden_size`.
    fn apply_config_json(&mut self, path: &Path) -> Result<()> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let root: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;

        if let Some(v) = json_usize(&root, "block_size") {
            self.block_size = v;
        }
        if let Some(v) = json_usize(&root, "mask_token_id") {
            self.mask_token = u32::try_from(v).context("mask_token_id does not fit u32")?;
        }
        if let Some(v) = json_usize(&root, "draft_vocab_size") {
            self.draft_vocab = v;
        }
        if let Some(v) = json_usize(&root, "max_anchors") {
            self.max_anchors = v;
        }
        if let Some(layers) = json_usize_array(&root, "aux_hidden_state_layer_ids") {
            // vLLM update_dflash: target_layer_ids = [i - 1 for i in ids]
            // (aux id n is the hidden AFTER n layers; our taps are post-layer
            // indexed). See DFLASH_DEFAULT_AUX_LAYERS.
            self.aux_layers = layers
                .into_iter()
                .map(|id| {
                    id.checked_sub(1)
                        .ok_or_else(|| anyhow!("aux_hidden_state_layer_ids contains 0 (no hidden exists before any layer ran)"))
                })
                .collect::<Result<Vec<_>>>()?;
        }
        // vLLM speculators algos.py: causal = not get("sliding_window_non_causal", True).
        self.causal = !root
            .get("sliding_window_non_causal")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let Some(tlc) = root.get("transformer_layer_config") else {
            return self.validate();
        };
        if let Some(v) = json_usize(tlc, "hidden_size") {
            self.hidden = v;
        }
        if let Some(v) = json_usize(tlc, "num_hidden_layers") {
            self.layers = v;
        }
        if let Some(v) = json_usize(tlc, "num_attention_heads") {
            self.heads = v;
        }
        if let Some(v) = json_usize(tlc, "num_key_value_heads") {
            self.kv_heads = v;
        }
        if let Some(v) = json_usize(tlc, "head_dim") {
            self.head_dim = v;
        }
        if let Some(v) = json_usize(tlc, "intermediate_size") {
            self.intermediate = v;
        }
        if let Some(v) = json_usize(tlc, "vocab_size") {
            self.embed_vocab = v;
        }
        if let Some(v) = tlc.get("rms_norm_eps").and_then(serde_json::Value::as_f64) {
            self.rms_eps = v as f32;
        }
        if let Some(v) = tlc
            .get("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
        {
            self.rope_theta = v as f32;
        }
        if let Some(hc) = json_usize(tlc, "hc_mult") {
            self.aux_width = hc * self.hidden;
        }
        if let Some(types) = tlc.get("layer_types").and_then(serde_json::Value::as_array) {
            let sliding = types
                .iter()
                .filter(|t| t.as_str() == Some("sliding_attention"))
                .count();
            if sliding != 0 && sliding != types.len() {
                bail!(
                    "DFlash checkpoints with mixed sliding/full layer_types are not supported ({sliding}/{} sliding)",
                    types.len()
                );
            }
            self.window = if sliding == types.len() {
                Some(json_usize(tlc, "sliding_window").ok_or_else(|| {
                    anyhow!("layer_types are all SWA but sliding_window is missing")
                })?)
            } else {
                None
            };
        }
        self.validate()
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
// Checkpoint tensor layout
// ---------------------------------------------------------------------------

const EMBED_TENSOR: &str = "embed_tokens.weight";
const D2T_TENSOR: &str = "d2t";

/// The provider-owned GEMM weights, addressed by a stable slot index shared
/// between the host and cuBLAS providers. Projections sharing an input are
/// FUSED at load by stacking their checkpoint tensors' output rows (out =
/// `x * W^T`, so row-concatenation concatenates outputs bit-for-bat): `Qkv`
/// is q|k|v per layer, `GateUp` is gate|up, and `CtxKv` stacks every layer's
/// k|v for the context-KV precompute — one GEMM for all layers, mirroring
/// vLLM's `_fused_kv_weight`. Fusion only cuts host<->device round trips;
/// the per-row math is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DfMat {
    Fc,
    LmHead,
    /// `[layers * 2 * kv_dim, hidden]`: rows `l*2*kv_dim ..` are layer `l`'s
    /// k_proj then v_proj.
    CtxKv,
    /// `[q_dim + 2 * kv_dim, hidden]`: q_proj | k_proj | v_proj.
    Qkv(usize),
    O(usize),
    /// `[2 * intermediate, hidden]`: gate_proj | up_proj.
    GateUp(usize),
    Down(usize),
}

/// Per-layer slot count in the provider weight table.
const LAYER_MATS: usize = 4;

fn mat_slot(mat: DfMat) -> usize {
    match mat {
        DfMat::Fc => 0,
        DfMat::LmHead => 1,
        DfMat::CtxKv => 2,
        DfMat::Qkv(l) => 3 + l * LAYER_MATS,
        DfMat::O(l) => 4 + l * LAYER_MATS,
        DfMat::GateUp(l) => 5 + l * LAYER_MATS,
        DfMat::Down(l) => 6 + l * LAYER_MATS,
    }
}

/// All provider mats in slot order.
fn mat_list(layers: usize) -> Vec<DfMat> {
    let mut list = vec![DfMat::Fc, DfMat::LmHead, DfMat::CtxKv];
    for l in 0..layers {
        list.extend([DfMat::Qkv(l), DfMat::O(l), DfMat::GateUp(l), DfMat::Down(l)]);
    }
    list
}

/// The checkpoint tensors whose rows are stacked (in order) to build `mat`.
fn mat_sources(cfg: &DFlashConfig, mat: DfMat) -> Vec<String> {
    let attn = |l: usize, which: &str| format!("layers.{l}.self_attn.{which}_proj.weight");
    let mlp = |l: usize, which: &str| format!("layers.{l}.mlp.{which}_proj.weight");
    match mat {
        DfMat::Fc => vec!["fc.weight".to_string()],
        DfMat::LmHead => vec!["lm_head.weight".to_string()],
        DfMat::CtxKv => (0..cfg.layers)
            .flat_map(|l| [attn(l, "k"), attn(l, "v")])
            .collect(),
        DfMat::Qkv(l) => vec![attn(l, "q"), attn(l, "k"), attn(l, "v")],
        DfMat::O(l) => vec![attn(l, "o")],
        DfMat::GateUp(l) => vec![mlp(l, "gate"), mlp(l, "up")],
        DfMat::Down(l) => vec![mlp(l, "down")],
    }
}

/// `(rows, cols)` of a provider mat as stored (row-major `[out, in]`; every
/// projection multiplies as `x [m, in] * W [out, in]^T`).
fn mat_dims(cfg: &DFlashConfig, mat: DfMat) -> (usize, usize) {
    match mat {
        DfMat::Fc => (cfg.hidden, cfg.fc_in()),
        DfMat::LmHead => (cfg.draft_vocab, cfg.hidden),
        DfMat::CtxKv => (cfg.layers * 2 * cfg.kv_dim(), cfg.hidden),
        DfMat::Qkv(_) => (cfg.q_dim() + 2 * cfg.kv_dim(), cfg.hidden),
        DfMat::O(_) => (cfg.hidden, cfg.q_dim()),
        DfMat::GateUp(_) => (2 * cfg.intermediate, cfg.hidden),
        DfMat::Down(_) => (cfg.hidden, cfg.intermediate),
    }
}

/// Load a provider mat as host f32 by concatenating its source tensors.
fn load_mat_f32(file: &SafetensorsFile, cfg: &DFlashConfig, mat: DfMat) -> Result<Vec<f32>> {
    let (n, k) = mat_dims(cfg, mat);
    let mut out = Vec::with_capacity(n * k);
    for name in mat_sources(cfg, mat) {
        out.extend(file.tensor_f32(&name)?);
    }
    if out.len() != n * k {
        bail!(
            "fused mat {mat:?} has {} values, expected {n}x{k}",
            out.len()
        );
    }
    Ok(out)
}

/// Every float tensor the checkpoint must contain, with its shape (the
/// granular projections that fused provider mats are built from, plus norms
/// and the embedding table). `d2t` (I64) is validated separately; extra
/// tensors like `t2d` are ignored.
fn float_tensor_shapes(cfg: &DFlashConfig) -> Vec<(String, Vec<usize>)> {
    let mut shapes = vec![
        (EMBED_TENSOR.to_string(), vec![cfg.embed_vocab, cfg.hidden]),
        ("hidden_norm.weight".to_string(), vec![cfg.hidden]),
        ("norm.weight".to_string(), vec![cfg.hidden]),
        ("fc.weight".to_string(), vec![cfg.hidden, cfg.fc_in()]),
        (
            "lm_head.weight".to_string(),
            vec![cfg.draft_vocab, cfg.hidden],
        ),
    ];
    for l in 0..cfg.layers {
        let attn = |which: &str| format!("layers.{l}.self_attn.{which}");
        shapes.push((attn("q_proj.weight"), vec![cfg.q_dim(), cfg.hidden]));
        shapes.push((attn("k_proj.weight"), vec![cfg.kv_dim(), cfg.hidden]));
        shapes.push((attn("v_proj.weight"), vec![cfg.kv_dim(), cfg.hidden]));
        shapes.push((attn("o_proj.weight"), vec![cfg.hidden, cfg.q_dim()]));
        shapes.push((attn("q_norm.weight"), vec![cfg.head_dim]));
        shapes.push((attn("k_norm.weight"), vec![cfg.head_dim]));
        shapes.push((
            format!("layers.{l}.mlp.gate_proj.weight"),
            vec![cfg.intermediate, cfg.hidden],
        ));
        shapes.push((
            format!("layers.{l}.mlp.up_proj.weight"),
            vec![cfg.intermediate, cfg.hidden],
        ));
        shapes.push((
            format!("layers.{l}.mlp.down_proj.weight"),
            vec![cfg.hidden, cfg.intermediate],
        ));
        shapes.push((
            format!("layers.{l}.input_layernorm.weight"),
            vec![cfg.hidden],
        ));
        shapes.push((
            format!("layers.{l}.post_attention_layernorm.weight"),
            vec![cfg.hidden],
        ));
    }
    shapes
}

/// Open + validate a checkpoint against `cfg`: every expected tensor present
/// with the exact shape and a float dtype `tensor_f32` can decode, plus the
/// `d2t` offset table (I64, in-range targets).
fn open_validated(path: &Path, cfg: &DFlashConfig) -> Result<SafetensorsFile> {
    let file = SafetensorsFile::open(path)?;
    for (name, shape) in float_tensor_shapes(cfg) {
        let info = file
            .info(&name)
            .ok_or_else(|| anyhow!("DFlash checkpoint is missing tensor {name:?}"))?;
        if info.shape != shape {
            bail!(
                "DFlash tensor {name:?} has shape {:?}, expected {shape:?}",
                info.shape
            );
        }
        if !matches!(
            info.dtype,
            SafetensorsDtype::BF16 | SafetensorsDtype::F16 | SafetensorsDtype::F32
        ) {
            bail!(
                "DFlash tensor {name:?} has dtype {}, expected BF16/F16/F32",
                info.dtype.name()
            );
        }
    }
    let d2t = file
        .info(D2T_TENSOR)
        .ok_or_else(|| anyhow!("DFlash checkpoint is missing the d2t vocab map"))?;
    if d2t.dtype != SafetensorsDtype::I64 || d2t.shape != vec![cfg.draft_vocab] {
        bail!(
            "DFlash d2t has dtype {} shape {:?}, expected I64 [{}]",
            d2t.dtype.name(),
            d2t.shape,
            cfg.draft_vocab
        );
    }
    Ok(file)
}

// ---------------------------------------------------------------------------
// GEMM providers
// ---------------------------------------------------------------------------

/// The drafter's matmul backend. `matmul_w` multiplies against a resident
/// checkpoint weight; the `gemm_*` variants take both operands from the
/// caller (attention scores and mixtures). All inputs/outputs are host f32
/// row-major.
trait DfLinear {
    /// `out [m, n] = x [m, k] * W [n, k]^T` for the resident weight `mat`.
    fn matmul_w(&mut self, mat: DfMat, x: &[f32], m: usize) -> Result<Vec<f32>>;
    /// `out [m, n] = a [m, k] * b [n, k]^T`.
    fn gemm_nt(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>>;
    /// `out [m, n] = a [m, k] * b [k, n]`.
    fn gemm_nn(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>>;
    /// Device bytes held by resident weights (0 for the host provider).
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    fn resident_bytes(&self) -> usize {
        0
    }
}

/// Exact-f32 host provider: the CPU reference the unit tests pin down and the
/// fixture-integration drafter runs on.
struct HostLinear {
    cfg: DFlashConfig,
    mats: Vec<Vec<f32>>,
}

impl HostLinear {
    fn from_file(file: &SafetensorsFile, cfg: &DFlashConfig) -> Result<Self> {
        let mut mats = Vec::new();
        for mat in mat_list(cfg.layers) {
            mats.push(load_mat_f32(file, cfg, mat)?);
        }
        Ok(Self {
            cfg: cfg.clone(),
            mats,
        })
    }
}

fn host_gemm_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        let a_row = &a[r * k..][..k];
        let out_row = &mut out[r * n..][..n];
        for (j, slot) in out_row.iter_mut().enumerate() {
            let b_row = &b[j * k..][..k];
            *slot = a_row.iter().zip(b_row).map(|(x, y)| x * y).sum();
        }
    }
    out
}

fn host_gemm_nn(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        let a_row = &a[r * k..][..k];
        let out_row = &mut out[r * n..][..n];
        for (i, &av) in a_row.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let b_row = &b[i * n..][..n];
            for (slot, &bv) in out_row.iter_mut().zip(b_row) {
                *slot += av * bv;
            }
        }
    }
    out
}

impl DfLinear for HostLinear {
    fn matmul_w(&mut self, mat: DfMat, x: &[f32], m: usize) -> Result<Vec<f32>> {
        let (n, k) = mat_dims(&self.cfg, mat);
        if x.len() != m * k {
            bail!(
                "matmul_w({mat:?}) input is {} values, expected {m}x{k}",
                x.len()
            );
        }
        Ok(host_gemm_nt(x, &self.mats[mat_slot(mat)], m, n, k))
    }

    fn gemm_nt(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        if a.len() != m * k || b.len() != n * k {
            bail!(
                "gemm_nt operand sizes {}x{} do not match {m}x{n}x{k}",
                a.len(),
                b.len()
            );
        }
        Ok(host_gemm_nt(a, b, m, n, k))
    }

    fn gemm_nn(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            bail!(
                "gemm_nn operand sizes {}x{} do not match {m}x{n}x{k}",
                a.len(),
                b.len()
            );
        }
        Ok(host_gemm_nn(a, b, m, n, k))
    }
}

/// cuBLAS provider: every checkpoint GEMM weight resident on-device as BF16
/// (f32 accumulate via `CUBLAS_COMPUTE_32F`), activations converted
/// f32<->bf16 per call — the drafter checkpoint AND vLLM's reference
/// execution are both bf16, and bf16 keeps f32's range (raw hc-stream aux
/// activations can carry outliers past f16's 65504 max). The blocking
/// stream + synchronous `cudaMemcpy` combination keeps host<->device
/// ordering via legacy default-stream semantics, exactly like the rest of
/// the crate's GPU paths.
#[cfg(feature = "native-cuda")]
struct CublasLinear {
    cfg: DFlashConfig,
    _stream: Stream,
    cublas: Cublas,
    mats: Vec<GpuMat>,
    resident_bytes: usize,
}

#[cfg(feature = "native-cuda")]
struct GpuMat {
    buf: DeviceBuffer,
    n: usize,
    k: usize,
}

#[cfg(feature = "native-cuda")]
impl CublasLinear {
    fn from_file(file: &SafetensorsFile, cfg: &DFlashConfig) -> Result<Self> {
        let stream = Stream::create()?;
        let cublas = Cublas::create()?;
        cublas.set_stream(&stream)?;
        let mut mats = Vec::new();
        let mut resident_bytes = 0usize;
        for mat in mat_list(cfg.layers) {
            let (n, k) = mat_dims(cfg, mat);
            let host: Vec<u16> = load_mat_f32(file, cfg, mat)
                .with_context(|| format!("loading {mat:?}"))?
                .into_iter()
                .map(f32_to_bf16_bits)
                .collect();
            let buf = DeviceBuffer::alloc(host.len() * 2)
                .with_context(|| format!("allocating {mat:?} on device"))?;
            buf.copy_from_host(&host)?;
            resident_bytes += host.len() * 2;
            mats.push(GpuMat { buf, n, k });
        }
        Ok(Self {
            cfg: cfg.clone(),
            _stream: stream,
            cublas,
            mats,
            resident_bytes,
        })
    }

    fn upload_bf16(values: &[f32]) -> Result<DeviceBuffer> {
        let bits = f32s_to_bf16(values);
        let buf = DeviceBuffer::alloc(bits.len() * 2)?;
        buf.copy_from_host(&bits)?;
        Ok(buf)
    }
}

/// f32 -> BF16 bits with round-to-nearest-even (bf16 is f32's top half).
#[cfg(feature = "native-cuda")]
fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Keep NaN quiet without rounding into infinity.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Bulk f32 -> bf16, parallelized for the big fc/context uploads.
#[cfg(feature = "native-cuda")]
fn f32s_to_bf16(values: &[f32]) -> Vec<u16> {
    use rayon::prelude::*;
    if values.len() >= (1 << 16) {
        values.par_iter().map(|&v| f32_to_bf16_bits(v)).collect()
    } else {
        values.iter().map(|&v| f32_to_bf16_bits(v)).collect()
    }
}

#[cfg(feature = "native-cuda")]
impl DfLinear for CublasLinear {
    fn matmul_w(&mut self, mat: DfMat, x: &[f32], m: usize) -> Result<Vec<f32>> {
        let gm = &self.mats[mat_slot(mat)];
        let (n, k) = (gm.n, gm.k);
        if x.len() != m * k {
            bail!(
                "matmul_w({mat:?}) input is {} values, expected {m}x{k}",
                x.len()
            );
        }
        let mut out = vec![0.0f32; m * n];
        let rows_per = (DFLASH_GEMM_CHUNK_BYTES / (2 * k)).max(1);
        let mut row = 0;
        while row < m {
            let mc = rows_per.min(m - row);
            let lhs = Self::upload_bf16(&x[row * k..(row + mc) * k])?;
            let dev_out = DeviceBuffer::alloc(mc * n * 4)?;
            self.cublas.matmul_mixed_rhs_transposed_row_major(
                &lhs,
                &gm.buf,
                &dev_out,
                mc,
                n,
                k,
                GemmDType::BF16,
                GemmDType::BF16,
            )?;
            let host: Vec<f32> = dev_out.copy_to_host(mc * n)?;
            out[row * n..(row + mc) * n].copy_from_slice(&host);
            row += mc;
        }
        Ok(out)
    }

    fn gemm_nt(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        if a.len() != m * k || b.len() != n * k {
            bail!(
                "gemm_nt operand sizes {}x{} do not match {m}x{n}x{k}",
                a.len(),
                b.len()
            );
        }
        let lhs = Self::upload_bf16(a)?;
        let rhs = Self::upload_bf16(b)?;
        let dev_out = DeviceBuffer::alloc(m * n * 4)?;
        self.cublas.matmul_mixed_rhs_transposed_row_major(
            &lhs,
            &rhs,
            &dev_out,
            m,
            n,
            k,
            GemmDType::BF16,
            GemmDType::BF16,
        )?;
        dev_out.copy_to_host(m * n)
    }

    fn gemm_nn(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            bail!(
                "gemm_nn operand sizes {}x{} do not match {m}x{n}x{k}",
                a.len(),
                b.len()
            );
        }
        let lhs = Self::upload_bf16(a)?;
        let rhs = Self::upload_bf16(b)?;
        let dev_out = DeviceBuffer::alloc(m * n * 4)?;
        self.cublas
            .matmul_mixed_row_major(&lhs, &rhs, &dev_out, m, n, k, GemmDType::BF16)?;
        dev_out.copy_to_host(m * n)
    }

    fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

// ---------------------------------------------------------------------------
// Host math shared by both providers
// ---------------------------------------------------------------------------

/// RMS-normalize every `weight.len()`-wide chunk of `x` in place (rows when
/// the weight is a layer norm, heads when it is a per-head q/k norm on a
/// contiguous multi-head row).
fn rms_norm_chunks(x: &mut [f32], weight: &[f32], eps: f32) {
    let width = weight.len();
    debug_assert_eq!(x.len() % width, 0);
    for chunk in x.chunks_mut(width) {
        let mean_sq = chunk.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        for (value, w) in chunk.iter_mut().zip(weight) {
            *value = *value * inv * w;
        }
    }
}

/// Apply neox-style rope in place to `heads` contiguous `head_dim`-wide heads
/// at absolute position `pos`: pair `(i, i + head_dim/2)` rotates by
/// `pos * inv_freq[i]`.
fn rope_heads(row: &mut [f32], heads: usize, head_dim: usize, pos: usize, inv_freq: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for h in 0..heads {
        let head = &mut row[h * head_dim..][..head_dim];
        for (i, &freq) in inv_freq.iter().enumerate() {
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// Key visibility per flash-attn window semantics: causal SWA is window
/// `(w-1, 0)` (`k <= q` and `q - k < w`); non-causal symmetrizes to
/// `(w-1, w-1)` (`|q - k| < w`), matching vLLM `_maybe_symmetrize_window`.
fn visible(causal: bool, window: Option<usize>, q_pos: usize, k_pos: usize) -> bool {
    if causal && k_pos > q_pos {
        return false;
    }
    match window {
        Some(w) => q_pos.abs_diff(k_pos) < w,
        None => true,
    }
}

/// Masked in-place softmax over one score row; invisible keys get weight 0.
fn masked_softmax(
    scores: &mut [f32],
    q_pos: usize,
    key_pos: impl Fn(usize) -> usize,
    causal: bool,
    window: Option<usize>,
) {
    let mut max = f32::NEG_INFINITY;
    for (j, &score) in scores.iter().enumerate() {
        if visible(causal, window, q_pos, key_pos(j)) && score > max {
            max = score;
        }
    }
    if max == f32::NEG_INFINITY {
        scores.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for (j, score) in scores.iter_mut().enumerate() {
        if visible(causal, window, q_pos, key_pos(j)) {
            *score = (*score - max).exp();
            sum += *score;
        } else {
            *score = 0.0;
        }
    }
    let inv = sum.recip();
    for score in scores.iter_mut() {
        *score *= inv;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn argmax_row(row: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, &value) in row.iter().enumerate() {
        if value > best_value {
            best = idx;
            best_value = value;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Aux-hidden source (decouples the drafter from DsV4Taps for unit tests)
// ---------------------------------------------------------------------------

/// The drafter's view of captured target hiddens: rows exist for ABSOLUTE
/// positions `base() .. positions()`, each exposing one flat row per tapped
/// layer. `base()` is non-zero when the request resumed from a prefix-cache
/// restore — the restored prefix's activations were never recomputed, so the
/// drafter's context KV floors there.
trait AuxSource {
    /// Absolute position of the first available row.
    fn base(&self) -> usize;
    /// Absolute END of the available range (the verified context length).
    fn positions(&self) -> usize;
    fn aux_row(&self, layer: usize, position: usize) -> Option<&[f32]>;
}

#[cfg(feature = "native-cuda")]
struct TapsAux<'a>(&'a DsV4Taps);

#[cfg(feature = "native-cuda")]
impl AuxSource for TapsAux<'_> {
    fn base(&self) -> usize {
        self.0.base()
    }

    fn positions(&self) -> usize {
        self.0.positions()
    }

    fn aux_row(&self, layer: usize, position: usize) -> Option<&[f32]> {
        self.0.aux_flat(layer, position)
    }
}

// ---------------------------------------------------------------------------
// The drafter
// ---------------------------------------------------------------------------

/// DFlash drafter over a GEMM provider. Owns the model's host-side small
/// weights, the memory-mapped checkpoint (embedding-row gathers), and the
/// incrementally maintained per-layer context KV cache (host f32,
/// authoritative; the GPU only ever sees per-call uploads).
struct DFlashDrafter<P: DfLinear> {
    cfg: DFlashConfig,
    file: SafetensorsFile,
    provider: P,
    hidden_norm: Vec<f32>,
    final_norm: Vec<f32>,
    input_ln: Vec<Vec<f32>>,
    post_ln: Vec<Vec<f32>>,
    q_norm: Vec<Vec<f32>>,
    k_norm: Vec<Vec<f32>>,
    mask_embed: Vec<f32>,
    d2t: Vec<i64>,
    inv_freq: Vec<f32>,
    /// Per-layer context K/V rows (`kv_dim` floats per position), row `i`
    /// holding ABSOLUTE position `ctx_base + i`. K rows are k-normed + roped
    /// at their absolute position; V rows are raw projections.
    ctx_k: Vec<Vec<f32>>,
    ctx_v: Vec<Vec<f32>>,
    /// Token ids whose taps produced the context rows (`inserted[i]` is the
    /// token at absolute position `ctx_base + i`; tap position `p` is
    /// captured while forwarding token `p`, so a matching token prefix
    /// implies matching taps under greedy decode). Mismatch or a new request
    /// truncates back to the common prefix.
    inserted: Vec<u32>,
    /// Absolute position of `inserted[0]` / the first context row — 0 for
    /// full-history requests, the taps base after a prefix-cache restore
    /// floored the buildable context (positions below it stay uncovered:
    /// a quality-only approximation, and invisible anyway once the anchor
    /// moves a full SWA window past the base).
    ctx_base: usize,
    proposals: u64,
    rows_inserted: u64,
    truncations: u64,
    propose_nanos: u128,
    /// `accepted_hist[n]` counts verify steps whose accepted prefix was `n`
    /// drafts long (index capped at the histogram tail). Slot `i >= 1`
    /// implies draft position `i` matched the target argmax, so the running
    /// position-1 rate is directly comparable to the checkpoint's published
    /// `position_1_acc` validation metric.
    accepted_hist: [u64; 9],
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    error_logged: bool,
    #[cfg(test)]
    last_mask_logits: Vec<Vec<f32>>,
}

impl<P: DfLinear> DFlashDrafter<P> {
    fn assemble(file: SafetensorsFile, cfg: DFlashConfig, provider: P) -> Result<Self> {
        let hidden_norm = file.tensor_f32("hidden_norm.weight")?;
        let final_norm = file.tensor_f32("norm.weight")?;
        let mut input_ln = Vec::with_capacity(cfg.layers);
        let mut post_ln = Vec::with_capacity(cfg.layers);
        let mut q_norm = Vec::with_capacity(cfg.layers);
        let mut k_norm = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            input_ln.push(file.tensor_f32(&format!("layers.{l}.input_layernorm.weight"))?);
            post_ln.push(file.tensor_f32(&format!("layers.{l}.post_attention_layernorm.weight"))?);
            q_norm.push(file.tensor_f32(&format!("layers.{l}.self_attn.q_norm.weight"))?);
            k_norm.push(file.tensor_f32(&format!("layers.{l}.self_attn.k_norm.weight"))?);
        }
        let d2t = file.tensor_i64(D2T_TENSOR)?;
        for (draft, &offset) in d2t.iter().enumerate() {
            let target = draft as i64 + offset;
            if target < 0 || target >= cfg.embed_vocab as i64 {
                bail!(
                    "DFlash d2t[{draft}] = {offset} maps outside the target vocab {}",
                    cfg.embed_vocab
                );
            }
        }
        let half = cfg.head_dim / 2;
        let inv_freq = (0..half)
            .map(|i| cfg.rope_theta.powf(-2.0 * i as f32 / cfg.head_dim as f32))
            .collect();
        let mask_embed = embed_row(&file, &cfg, cfg.mask_token)?;
        Ok(Self {
            ctx_k: vec![Vec::new(); cfg.layers],
            ctx_v: vec![Vec::new(); cfg.layers],
            cfg,
            file,
            provider,
            hidden_norm,
            final_norm,
            input_ln,
            post_ln,
            q_norm,
            k_norm,
            mask_embed,
            d2t,
            inv_freq,
            inserted: Vec::new(),
            ctx_base: 0,
            proposals: 0,
            rows_inserted: 0,
            truncations: 0,
            propose_nanos: 0,
            accepted_hist: [0; 9],
            error_logged: false,
            #[cfg(test)]
            last_mask_logits: Vec::new(),
        })
    }

    fn embed_row(&self, id: u32) -> Result<Vec<f32>> {
        embed_row(&self.file, &self.cfg, id)
    }

    /// Drop every context row at or beyond `keep` (rejected drafts, request
    /// switches, prompt divergence).
    fn truncate_context(&mut self, keep: usize) {
        let kv_dim = self.cfg.kv_dim();
        for k in &mut self.ctx_k {
            k.truncate(keep * kv_dim);
        }
        for v in &mut self.ctx_v {
            v.truncate(keep * kv_dim);
        }
        self.inserted.truncate(keep);
        self.truncations += 1;
    }

    /// Project `m` flat aux-concat rows (positions `start_pos ..`) through fc
    /// + hidden_norm and append each layer's K (k_norm + rope) / V rows to the
    /// context cache — vLLM `precompute_and_store_context_kv`, including its
    /// one-GEMM-for-all-layers fused KV projection.
    fn append_context(&mut self, aux_rows: &[f32], m: usize, start_pos: usize) -> Result<()> {
        let cfg = &self.cfg;
        debug_assert_eq!(aux_rows.len(), m * cfg.fc_in());
        let mut hidden = self.provider.matmul_w(DfMat::Fc, aux_rows, m)?;
        rms_norm_chunks(&mut hidden, &self.hidden_norm, cfg.rms_eps);
        let kv_dim = cfg.kv_dim();
        // Row r of the fused output holds [k_0 | v_0 | k_1 | v_1 | ...].
        let all_kv = self.provider.matmul_w(DfMat::CtxKv, &hidden, m)?;
        let row_width = cfg.layers * 2 * kv_dim;
        for l in 0..cfg.layers {
            for r in 0..m {
                let row = &all_kv[r * row_width + l * 2 * kv_dim..][..2 * kv_dim];
                let mut k = row[..kv_dim].to_vec();
                rms_norm_chunks(&mut k, &self.k_norm[l], cfg.rms_eps);
                rope_heads(
                    &mut k,
                    cfg.kv_heads,
                    cfg.head_dim,
                    start_pos + r,
                    &self.inv_freq,
                );
                self.ctx_k[l].extend_from_slice(&k);
                self.ctx_v[l].extend_from_slice(&row[kv_dim..]);
            }
        }
        self.rows_inserted += m as u64;
        Ok(())
    }

    /// One draft block forward: queries `[anchor, mask x k]` at absolute
    /// positions `pos0 .. pos0 + k` against the visible context window plus
    /// the in-block query KV. Returns `k` TARGET-vocab draft ids.
    fn run_block(&mut self, anchor: u32, pos0: usize, k: usize) -> Result<Vec<u32>> {
        let cfg = self.cfg.clone();
        let rows = 1 + k;
        let hidden = cfg.hidden;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let heads_per_kv = cfg.heads_per_kv();
        let scale = (head_dim as f32).sqrt().recip();

        let mut x = Vec::with_capacity(rows * hidden);
        x.extend_from_slice(&self.embed_row(anchor)?);
        for _ in 0..k {
            x.extend_from_slice(&self.mask_embed);
        }

        // Context keys below `first` are outside every query's window (the
        // anchor at pos0 has the widest reach); positions stay absolute. The
        // cache holds rows for absolute positions `ctx_base ..
        // ctx_base + inserted.len()` only, so visibility ALSO floors at the
        // cache base (a restore-floored cache simply exposes fewer context
        // keys until the window moves past the base).
        let ctx_end = self.ctx_base + self.inserted.len();
        let first = match cfg.window {
            Some(w) => (pos0 + 1).saturating_sub(w),
            None => 0,
        }
        .max(self.ctx_base);
        let n_ctx_vis = ctx_end - first;
        let n_keys = n_ctx_vis + rows;
        let key_pos = |j: usize| {
            if j < n_ctx_vis {
                first + j
            } else {
                pos0 + (j - n_ctx_vis)
            }
        };

        for l in 0..cfg.layers {
            let mut x_norm = x.clone();
            rms_norm_chunks(&mut x_norm, &self.input_ln[l], cfg.rms_eps);

            // Fused q|k|v projection, then split rows and apply the per-head
            // norms + rope exactly as the granular path would.
            let qkv = self.provider.matmul_w(DfMat::Qkv(l), &x_norm, rows)?;
            let qkv_width = q_dim + 2 * kv_dim;
            let mut q = Vec::with_capacity(rows * q_dim);
            let mut k_q = Vec::with_capacity(rows * kv_dim);
            let mut v_q = Vec::with_capacity(rows * kv_dim);
            for r in 0..rows {
                let row = &qkv[r * qkv_width..][..qkv_width];
                q.extend_from_slice(&row[..q_dim]);
                k_q.extend_from_slice(&row[q_dim..q_dim + kv_dim]);
                v_q.extend_from_slice(&row[q_dim + kv_dim..]);
            }
            rms_norm_chunks(&mut q, &self.q_norm[l], cfg.rms_eps);
            rms_norm_chunks(&mut k_q, &self.k_norm[l], cfg.rms_eps);
            for r in 0..rows {
                rope_heads(
                    &mut q[r * q_dim..][..q_dim],
                    cfg.heads,
                    head_dim,
                    pos0 + r,
                    &self.inv_freq,
                );
                rope_heads(
                    &mut k_q[r * kv_dim..][..kv_dim],
                    cfg.kv_heads,
                    head_dim,
                    pos0 + r,
                    &self.inv_freq,
                );
            }

            let mut attn = vec![0.0f32; rows * q_dim];
            for h in 0..cfg.kv_heads {
                // Gather this kv head's keys/values: visible context rows,
                // then the in-block query rows.
                let mut keys = Vec::with_capacity(n_keys * head_dim);
                let mut values = Vec::with_capacity(n_keys * head_dim);
                for j in 0..n_ctx_vis {
                    // Row index = absolute position - ctx_base.
                    let base = (first - self.ctx_base + j) * kv_dim + h * head_dim;
                    keys.extend_from_slice(&self.ctx_k[l][base..base + head_dim]);
                    values.extend_from_slice(&self.ctx_v[l][base..base + head_dim]);
                }
                for r in 0..rows {
                    let base = r * kv_dim + h * head_dim;
                    keys.extend_from_slice(&k_q[base..base + head_dim]);
                    values.extend_from_slice(&v_q[base..base + head_dim]);
                }
                // Grouped queries (contiguous GQA mapping), pre-scaled.
                let group_rows = rows * heads_per_kv;
                let mut q_group = Vec::with_capacity(group_rows * head_dim);
                for r in 0..rows {
                    for hh in 0..heads_per_kv {
                        let qh = h * heads_per_kv + hh;
                        let base = r * q_dim + qh * head_dim;
                        q_group.extend(q[base..base + head_dim].iter().map(|v| v * scale));
                    }
                }
                let mut scores = self
                    .provider
                    .gemm_nt(&q_group, &keys, group_rows, n_keys, head_dim)?;
                let softmax_row = |(idx, score_row): (usize, &mut [f32])| {
                    let r = idx / heads_per_kv;
                    masked_softmax(score_row, pos0 + r, key_pos, cfg.causal, cfg.window);
                };
                if group_rows * n_keys >= 1 << 15 {
                    use rayon::prelude::*;
                    scores
                        .par_chunks_mut(n_keys)
                        .enumerate()
                        .for_each(softmax_row);
                } else {
                    scores.chunks_mut(n_keys).enumerate().for_each(softmax_row);
                }
                let mixed = self
                    .provider
                    .gemm_nn(&scores, &values, group_rows, head_dim, n_keys)?;
                for r in 0..rows {
                    for hh in 0..heads_per_kv {
                        let qh = h * heads_per_kv + hh;
                        attn[r * q_dim + qh * head_dim..][..head_dim].copy_from_slice(
                            &mixed[(r * heads_per_kv + hh) * head_dim..][..head_dim],
                        );
                    }
                }
            }
            let o = self.provider.matmul_w(DfMat::O(l), &attn, rows)?;
            for (value, delta) in x.iter_mut().zip(&o) {
                *value += delta;
            }

            let mut y = x.clone();
            rms_norm_chunks(&mut y, &self.post_ln[l], cfg.rms_eps);
            let gate_up = self.provider.matmul_w(DfMat::GateUp(l), &y, rows)?;
            let inter = cfg.intermediate;
            let mut act = Vec::with_capacity(rows * inter);
            for r in 0..rows {
                let row = &gate_up[r * 2 * inter..][..2 * inter];
                act.extend(
                    row[..inter]
                        .iter()
                        .zip(&row[inter..])
                        .map(|(&g, &u)| silu(g) * u),
                );
            }
            let down = self.provider.matmul_w(DfMat::Down(l), &act, rows)?;
            for (value, delta) in x.iter_mut().zip(&down) {
                *value += delta;
            }
        }

        // Only the K mask rows are sampled (the anchor row conditions them
        // through attention but is never a draft).
        let mut mask_hidden = x[hidden..].to_vec();
        rms_norm_chunks(&mut mask_hidden, &self.final_norm, cfg.rms_eps);
        let logits = self.provider.matmul_w(DfMat::LmHead, &mask_hidden, k)?;
        #[cfg(test)]
        {
            self.last_mask_logits = logits
                .chunks(cfg.draft_vocab)
                .map(<[f32]>::to_vec)
                .collect();
        }
        let mut drafts = Vec::with_capacity(k);
        for row in logits.chunks(cfg.draft_vocab) {
            let draft = argmax_row(row);
            drafts.push((draft as i64 + self.d2t[draft]) as u32);
        }
        Ok(drafts)
    }

    /// The full proposal step: reconcile the context KV cache with the taps
    /// (truncate on divergence, append newly verified positions), then run
    /// one mask-query block after the pending token.
    fn propose_inner(
        &mut self,
        tokens: &[u32],
        aux: &dyn AuxSource,
        k_cap: usize,
    ) -> Result<Vec<u32>> {
        let start = Instant::now();
        let n_ctx = aux.positions();
        if tokens.len() != n_ctx + 1 {
            bail!(
                "DFlash drafter got {} context tokens but {n_ctx} tap positions (want tokens = taps + pending)",
                tokens.len()
            );
        }
        let k = k_cap.min(self.cfg.block_size - 1);
        if k == 0 {
            return Ok(Vec::new());
        }
        let anchor = tokens[n_ctx];
        if anchor as usize >= self.cfg.embed_vocab {
            bail!(
                "DFlash anchor token {anchor} outside the drafter embed vocab {}",
                self.cfg.embed_vocab
            );
        }

        // Longest verified prefix both sides agree on, compared at the
        // cache's own absolute offset. Within one request this is everything
        // (taps only ever extend by accepted tokens); across requests it
        // re-uses a shared conversation prefix — rows below the new request's
        // taps base stay valid (their taps existed when they were built) —
        // and truncates the rest. Rows that would have to be REBUILT below
        // the base cannot be (no taps there), so the cache rebases: context
        // building floors at the taps base.
        let base = aux.base();
        let matched = if self.ctx_base <= n_ctx {
            self.inserted
                .iter()
                .zip(&tokens[self.ctx_base..n_ctx])
                .take_while(|(a, b)| a == b)
                .count()
        } else {
            0
        };
        if self.ctx_base > n_ctx || self.ctx_base + matched < base {
            // Nothing reusable at or above the base: restart the cache there
            // (an already-empty cache just moves — no truncation to count).
            if !self.inserted.is_empty() {
                self.truncate_context(0);
            }
            self.ctx_base = base;
        } else if matched < self.inserted.len() {
            self.truncate_context(matched);
        }
        let keep = self.ctx_base + self.inserted.len();
        debug_assert!(keep >= base);
        if keep < n_ctx {
            let m = n_ctx - keep;
            let fc_in = self.cfg.fc_in();
            let mut rows = Vec::with_capacity(m * fc_in);
            for pos in keep..n_ctx {
                for &layer in &self.cfg.aux_layers {
                    let row = aux.aux_row(layer, pos).ok_or_else(|| {
                        anyhow!("missing aux tap for layer {layer} position {pos}")
                    })?;
                    if row.len() != self.cfg.aux_width {
                        bail!(
                            "aux tap rows are {} wide but the drafter fc expects {} per layer",
                            row.len(),
                            self.cfg.aux_width
                        );
                    }
                    rows.extend_from_slice(row);
                }
            }
            self.append_context(&rows, m, keep)?;
            self.inserted.extend_from_slice(&tokens[keep..n_ctx]);
        }

        let drafts = self.run_block(anchor, n_ctx, k)?;
        self.proposals += 1;
        self.propose_nanos += start.elapsed().as_nanos();
        Ok(drafts)
    }

    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    fn log_error_once(&mut self, message: &str) {
        if !self.error_logged {
            self.error_logged = true;
            eprintln!("DFlash drafter disabled itself for this run: {message}");
        }
    }
}

impl<P: DfLinear> Drop for DFlashDrafter<P> {
    fn drop(&mut self) {
        if self.proposals > 0 {
            let steps: u64 = self.accepted_hist.iter().sum();
            let p1 = self.accepted_hist[1..].iter().sum::<u64>() as f64 / steps.max(1) as f64;
            eprintln!(
                "DFlash drafter: {} proposals, {:.2} ms/propose avg, {} context rows inserted, \
                 {} truncations, accepted-prefix histogram {:?} (position-1 acceptance {:.1}%)",
                self.proposals,
                self.propose_nanos as f64 / self.proposals as f64 / 1e6,
                self.rows_inserted,
                self.truncations,
                self.accepted_hist,
                100.0 * p1,
            );
        }
    }
}

/// Read one embedding row straight from the memory-mapped checkpoint (the
/// full 129280x4096 table never becomes device- or heap-resident; per propose
/// only the anchor row is gathered — the mask row is cached at load).
fn embed_row(file: &SafetensorsFile, cfg: &DFlashConfig, id: u32) -> Result<Vec<f32>> {
    let info = file
        .info(EMBED_TENSOR)
        .ok_or_else(|| anyhow!("missing {EMBED_TENSOR}"))?;
    let id = id as usize;
    if id >= cfg.embed_vocab {
        bail!("embedding row {id} outside vocab {}", cfg.embed_vocab);
    }
    let width = cfg.hidden;
    let bytes = file.bytes(EMBED_TENSOR)?;
    let elem = info.dtype.byte_width();
    let row = &bytes[id * width * elem..][..width * elem];
    let via_u16 = |f: fn(u16) -> f32| {
        row.chunks_exact(2)
            .map(|c| f(u16::from_le_bytes([c[0], c[1]])))
            .collect::<Vec<f32>>()
    };
    Ok(match info.dtype {
        SafetensorsDtype::BF16 => via_u16(bf16_to_f32),
        SafetensorsDtype::F16 => via_u16(f16_to_f32),
        SafetensorsDtype::F32 => row
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => bail!("embedding dtype {} unsupported", other.name()),
    })
}

impl DFlashDrafter<HostLinear> {
    /// CPU-reference construction (unit tests, fixture integration).
    fn load_host(path: &Path, cfg: DFlashConfig) -> Result<Self> {
        cfg.validate()?;
        let file = open_validated(path, &cfg)?;
        let provider = HostLinear::from_file(&file, &cfg)?;
        Self::assemble(file, cfg, provider)
    }
}

#[cfg(feature = "native-cuda")]
impl DFlashDrafter<CublasLinear> {
    /// Production construction: resident-bf16 cuBLAS provider.
    fn load_gpu(path: &Path, cfg: DFlashConfig) -> Result<Self> {
        cfg.validate()?;
        let file = open_validated(path, &cfg)?;
        let provider = CublasLinear::from_file(&file, &cfg)?;
        Self::assemble(file, cfg, provider)
    }
}

// ---------------------------------------------------------------------------
// Drafter-trait wiring (native-cuda)
// ---------------------------------------------------------------------------

#[cfg(feature = "native-cuda")]
impl<P: DfLinear> Drafter for DFlashDrafter<P> {
    fn tap_config(&self) -> DsV4TapConfig {
        DsV4TapConfig {
            pre_hc_head: false,
            aux_layers: self.cfg.aux_layers.clone(),
        }
    }

    fn propose(&mut self, ctx: &DraftContext<'_>) -> Vec<u32> {
        let Some(taps) = ctx.taps else {
            self.log_error_once("the verify loop supplied no hidden taps");
            return Vec::new();
        };
        match self.propose_inner(ctx.tokens, &TapsAux(taps), ctx.k) {
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
            let slot = accepted.min(self.accepted_hist.len() - 1);
            self.accepted_hist[slot] += 1;
        }
    }
}

/// Resolve the checkpoint path (`HI_DSV4_DFLASH_PATH`, default the RedHat
/// layout under `$HOME`).
#[cfg(feature = "native-cuda")]
fn dflash_checkpoint_path() -> PathBuf {
    if let Some(path) = std::env::var_os("HI_DSV4_DFLASH_PATH") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME").unwrap_or_default();
    PathBuf::from(home).join(DFLASH_DEFAULT_SUFFIX)
}

/// Build the production GPU drafter from a checkpoint file: baked config,
/// then `config.json` overrides from the sibling file, then the
/// `HI_DSV4_DFLASH_AUX` tap-layer override (comma-separated; must keep the
/// trained layer count or fc validation fails).
#[cfg(feature = "native-cuda")]
fn build_gpu_drafter(path: &Path) -> Result<DFlashDrafter<CublasLinear>> {
    let mut cfg = DFlashConfig::real_default();
    if let Some(dir) = path.parent() {
        let config_json = dir.join("config.json");
        if config_json.exists() {
            cfg.apply_config_json(&config_json)
                .with_context(|| format!("applying {}", config_json.display()))?;
        }
    }
    if let Ok(raw) = std::env::var("HI_DSV4_DFLASH_AUX") {
        let layers = raw
            .split(',')
            .map(|part| part.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("parsing HI_DSV4_DFLASH_AUX {raw:?}"))?;
        eprintln!("HI_DSV4_DFLASH_AUX override: tapping target layers {layers:?}");
        cfg.aux_layers = layers;
    }
    cfg.validate()?;
    DFlashDrafter::load_gpu(path, cfg)
}

/// `HI_DSV4_SPEC=dflash` entry point, constructed on the engine worker thread
/// (device resources are allowed here). Returning `None` leaves speculative
/// decoding off.
#[cfg(feature = "native-cuda")]
pub(crate) fn dflash_drafter_from_env(engine: &DeepSeekV4GpuEngine) -> Option<Box<dyn Drafter>> {
    let path = dflash_checkpoint_path();
    if !path.exists() {
        eprintln!(
            "HI_DSV4_SPEC=dflash: checkpoint {} not found (set HI_DSV4_DFLASH_PATH); \
             speculative decoding stays off",
            path.display()
        );
        return None;
    }
    match build_gpu_drafter(&path) {
        Ok(drafter) => {
            let cfg = &drafter.cfg;
            let target_vocab = engine.tokenizer().token_count();
            if target_vocab > cfg.embed_vocab {
                eprintln!(
                    "HI_DSV4_SPEC=dflash: target vocab {target_vocab} exceeds the drafter's embed \
                     table {}; proposals will skip out-of-range anchors",
                    cfg.embed_vocab
                );
            }
            eprintln!(
                "HI_DSV4_SPEC=dflash: loaded {} — {} layers ({} heads x {} dim, {} kv), \
                 {:.2} GiB bf16 device-resident, aux layers {:?} ({} wide), window {:?}, \
                 causal {}, block {} (K = {}), draft vocab {}",
                path.display(),
                cfg.layers,
                cfg.heads,
                cfg.head_dim,
                cfg.kv_heads,
                drafter.provider.resident_bytes() as f64 / f64::from(1u32 << 30),
                cfg.aux_layers,
                cfg.aux_width,
                cfg.window,
                cfg.causal,
                cfg.block_size,
                cfg.block_size - 1,
                cfg.draft_vocab,
            );
            Some(Box::new(drafter))
        }
        Err(err) => {
            eprintln!(
                "HI_DSV4_SPEC=dflash: failed to load {}: {err:#}; speculative decoding stays off",
                path.display()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Synthetic-checkpoint plumbing (mirrors the safetensors.rs test writer)
    // -----------------------------------------------------------------------

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-cuda-dflash-{name}-{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    fn write_st(path: &Path, tensors: &[(String, &'static str, Vec<usize>, Vec<u8>)]) {
        let mut entries = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len();
            data.extend_from_slice(bytes);
            entries.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
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

    fn bf16_bits_exact(value: f32) -> u16 {
        let bits = (value.to_bits() >> 16) as u16;
        assert_eq!(
            bf16_to_f32(bits),
            value,
            "test value {value} is not bf16-representable"
        );
        bits
    }

    /// Write a synthetic DFlash checkpoint whose float tensors come from
    /// `weights_gen(name, element_count)`; `bf16` selects the storage dtype (values
    /// must then be exactly representable).
    fn write_checkpoint_as(
        path: &Path,
        cfg: &DFlashConfig,
        weights_gen: &dyn Fn(&str, usize) -> Vec<f32>,
        d2t: &[i64],
        bf16: bool,
    ) {
        assert_eq!(d2t.len(), cfg.draft_vocab);
        let mut tensors = Vec::new();
        for (name, shape) in float_tensor_shapes(cfg) {
            let len: usize = shape.iter().product();
            let values = weights_gen(&name, len);
            assert_eq!(values.len(), len, "generator size for {name}");
            let (dtype, bytes): (&'static str, Vec<u8>) = if bf16 {
                (
                    "BF16",
                    values
                        .iter()
                        .flat_map(|&v| bf16_bits_exact(v).to_le_bytes())
                        .collect(),
                )
            } else {
                ("F32", values.iter().flat_map(|v| v.to_le_bytes()).collect())
            };
            tensors.push((name, dtype, shape, bytes));
        }
        tensors.push((
            D2T_TENSOR.to_string(),
            "I64",
            vec![cfg.draft_vocab],
            d2t.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        write_st(path, &tensors);
    }

    fn write_checkpoint(
        path: &Path,
        cfg: &DFlashConfig,
        weights_gen: &dyn Fn(&str, usize) -> Vec<f32>,
        d2t: &[i64],
    ) {
        write_checkpoint_as(path, cfg, weights_gen, d2t, false);
    }

    fn fnv(s: &str) -> u64 {
        s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Deterministic pseudo-random values in `(-scale/2, scale/2)`.
    fn lcg_values(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (((state >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5) * scale
            })
            .collect()
    }

    /// Name-keyed deterministic weights with per-test overrides.
    fn gen_with(
        scale: f32,
        overrides: impl Fn(&str, usize) -> Option<Vec<f32>>,
    ) -> impl Fn(&str, usize) -> Vec<f32> {
        move |name, len| overrides(name, len).unwrap_or_else(|| lcg_values(fnv(name), len, scale))
    }

    fn eye(n: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            out[i * n + i] = scale;
        }
        out
    }

    fn zeros(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn ones(len: usize) -> Vec<f32> {
        vec![1.0; len]
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "{what}[{i}]: got {a}, expected {e} (tol {tol})"
            );
        }
    }

    /// Test double for the engine taps: one flat row per (position, layer),
    /// rows starting at absolute position `base` like a restore-attached
    /// [`DsV4Taps`].
    struct OwnedAux {
        layers: Vec<usize>,
        base: usize,
        /// `rows[position - base][aux_group]`.
        rows: Vec<Vec<Vec<f32>>>,
    }

    impl AuxSource for OwnedAux {
        fn base(&self) -> usize {
            self.base
        }

        fn positions(&self) -> usize {
            self.base + self.rows.len()
        }

        fn aux_row(&self, layer: usize, position: usize) -> Option<&[f32]> {
            let group = self.layers.iter().position(|&l| l == layer)?;
            self.rows
                .get(position.checked_sub(self.base)?)
                .map(|row| row[group].as_slice())
        }
    }

    /// Rows are keyed off the token at each position, so "same tokens =>
    /// same taps" holds across reconstructions like it does for the real
    /// (deterministic, greedy) engine.
    fn owned_aux(layers: &[usize], width: usize, tokens: &[u32]) -> OwnedAux {
        owned_aux_at(layers, width, tokens, 0)
    }

    /// [`owned_aux`] with the first `base` positions withheld (a prefix-cache
    /// restore's view: `tokens` is still the FULL token list, rows exist only
    /// from `base` on and stay identical to the full buffer's rows there).
    fn owned_aux_at(layers: &[usize], width: usize, tokens: &[u32], base: usize) -> OwnedAux {
        let rows = tokens[base..]
            .iter()
            .map(|&token| {
                layers
                    .iter()
                    .map(|&layer| lcg_values(fnv(&format!("aux-{layer}-{token}")), width, 0.8))
                    .collect()
            })
            .collect();
        OwnedAux {
            layers: layers.to_vec(),
            base,
            rows,
        }
    }

    /// Small all-purpose geometry: 2 layers, GQA-free MQA, SWA window 4.
    fn tiny_cfg() -> DFlashConfig {
        DFlashConfig {
            hidden: 8,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 4,
            embed_vocab: 16,
            draft_vocab: 5,
            aux_layers: vec![0, 1],
            aux_width: 8,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            window: Some(4),
            causal: true,
            block_size: 8,
            mask_token: 1,
            max_anchors: 64,
        }
    }

    fn load_tiny(name: &str, cfg: &DFlashConfig) -> DFlashDrafter<HostLinear> {
        let path = tempfile_path(name);
        write_checkpoint(
            &path,
            cfg,
            &gen_with(0.4, |_, _| None),
            &vec![0i64; cfg.draft_vocab],
        );
        let drafter = DFlashDrafter::load_host(&path, cfg.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        drafter
    }

    // -----------------------------------------------------------------------
    // Math helpers: hand-computed assertions
    // -----------------------------------------------------------------------

    #[test]
    fn dflash_rms_norm_chunks_hand_computed() {
        // Two chunks of width 2 against weight [2, 1]:
        //   [3,4]: rms = sqrt(12.5), out = [6/3.53553, 4/3.53553]
        //   [6,8]: rms = sqrt(50),   out = [12/7.07107, 8/7.07107]
        let mut x = vec![3.0, 4.0, 6.0, 8.0];
        rms_norm_chunks(&mut x, &[2.0, 1.0], 1e-9);
        assert_close(
            &x,
            &[1.697_056_3, 1.131_370_8, 1.697_056_3, 1.131_370_8],
            1e-5,
            "rms",
        );
    }

    #[test]
    fn dflash_rope_rotates_neox_pairs_hand_computed() {
        // head_dim 4, theta 10000: inv_freq = [1, 0.01]. Neox pairs (i, i+2).
        let inv_freq = [1.0f32, 0.01];
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        rope_heads(&mut row, 1, 4, 0, &inv_freq);
        assert_close(&row, &[1.0, 2.0, 3.0, 4.0], 1e-6, "rope pos 0");

        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        rope_heads(&mut row, 1, 4, 2, &inv_freq);
        let (s0, c0) = 2.0f32.sin_cos();
        let (s1, c1) = 0.02f32.sin_cos();
        let expected = [
            1.0 * c0 - 3.0 * s0,
            2.0 * c1 - 4.0 * s1,
            1.0 * s0 + 3.0 * c0,
            2.0 * s1 + 4.0 * c1,
        ];
        assert_close(&row, &expected, 1e-5, "rope pos 2");

        // Two heads rotate independently with the same angles.
        let mut two = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rope_heads(&mut two, 2, 4, 2, &inv_freq);
        assert_close(&two[..4], &expected, 1e-5, "rope head 0");
        let expected_h2 = [
            5.0 * c0 - 7.0 * s0,
            6.0 * c1 - 8.0 * s1,
            5.0 * s0 + 7.0 * c0,
            6.0 * s1 + 8.0 * c1,
        ];
        assert_close(&two[4..], &expected_h2, 1e-5, "rope head 1");
    }

    #[test]
    fn dflash_visibility_matches_flash_attn_window_semantics() {
        // Causal SWA w: window (w-1, 0) -> k <= q and q - k < w.
        assert!(visible(true, Some(3), 5, 5));
        assert!(visible(true, Some(3), 5, 4));
        assert!(visible(true, Some(3), 5, 3));
        assert!(!visible(true, Some(3), 5, 2));
        assert!(!visible(true, Some(3), 5, 6));
        // Non-causal SWA symmetrizes to (w-1, w-1) -> |q - k| < w.
        assert!(visible(false, Some(3), 5, 3));
        assert!(visible(false, Some(3), 5, 7));
        assert!(!visible(false, Some(3), 5, 2));
        assert!(!visible(false, Some(3), 5, 8));
        // Full attention.
        assert!(visible(true, None, 5, 0));
        assert!(!visible(true, None, 5, 6));
        assert!(visible(false, None, 5, 100));
    }

    #[test]
    fn dflash_masked_softmax_zeroes_invisible_keys() {
        // Keys at positions [0, 1, 2], query at 1, causal full attention:
        // keys 0 and 1 visible with equal scores -> 0.5 each.
        let mut scores = vec![0.0, 0.0, 0.0];
        masked_softmax(&mut scores, 1, |j| j, true, None);
        assert_close(&scores, &[0.5, 0.5, 0.0], 1e-6, "causal");

        // Window 1: only the self key survives.
        let mut scores = vec![10.0, 3.0, 10.0];
        masked_softmax(&mut scores, 1, |j| j, true, Some(1));
        assert_close(&scores, &[0.0, 1.0, 0.0], 1e-6, "window 1");

        // Non-causal full: plain softmax over all three.
        let mut scores = vec![0.0, (2.0f32).ln(), 0.0];
        masked_softmax(&mut scores, 1, |j| j, false, None);
        assert_close(&scores, &[0.25, 0.5, 0.25], 1e-5, "non-causal");
    }

    #[test]
    fn dflash_silu_and_argmax_helpers() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
        assert!((silu(-1.0) + 0.268_941_4).abs() < 1e-6);
        assert_eq!(argmax_row(&[0.5, 2.0, 2.0, -1.0]), 1, "first max wins ties");
    }

    // -----------------------------------------------------------------------
    // Loader validation
    // -----------------------------------------------------------------------

    #[test]
    fn dflash_loader_validates_shapes_and_d2t() {
        let cfg = tiny_cfg();
        let weights_gen = gen_with(0.4, |_, _| None);
        let d2t: Vec<i64> = vec![0, 1, 2, 3, 0];

        let path = tempfile_path("loader-good");
        write_checkpoint(&path, &cfg, &weights_gen, &d2t);
        assert!(DFlashDrafter::load_host(&path, cfg.clone()).is_ok());
        std::fs::remove_file(&path).ok();

        // Wrong fc shape: validated against aux_layers.len() * aux_width.
        let mut wide = cfg.clone();
        wide.aux_layers = vec![0, 1, 2];
        let path = tempfile_path("loader-fc");
        write_checkpoint(&path, &cfg, &weights_gen, &d2t);
        let err = DFlashDrafter::load_host(&path, wide)
            .err()
            .expect("fc width mismatch must fail")
            .to_string();
        assert!(err.contains("fc.weight"), "got: {err}");
        std::fs::remove_file(&path).ok();

        // d2t offsets must stay inside the target vocab.
        let path = tempfile_path("loader-d2t");
        write_checkpoint(&path, &cfg, &weights_gen, &[0, 0, 0, 0, 100]);
        let err = format!(
            "{:#}",
            DFlashDrafter::load_host(&path, cfg.clone())
                .err()
                .expect("out-of-range d2t must fail")
        );
        assert!(err.contains("d2t"), "got: {err}");
        std::fs::remove_file(&path).ok();

        // A missing tensor is named in the error.
        let mut fewer = cfg.clone();
        fewer.layers = 1;
        let path = tempfile_path("loader-missing");
        write_checkpoint(&path, &fewer, &weights_gen, &d2t);
        let err = DFlashDrafter::load_host(&path, cfg.clone())
            .err()
            .expect("missing tensor must fail")
            .to_string();
        assert!(err.contains("layers.1."), "got: {err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dflash_config_json_overrides_defaults() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "hi-cuda-dflash-config-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // The real checkpoint's fields (trimmed).
        std::fs::write(
            &path,
            serde_json::json!({
                "aux_hidden_state_layer_ids": [3, 13, 23, 32, 42],
                "block_size": 8,
                "draft_vocab_size": 32000,
                "mask_token_id": 1,
                "max_anchors": 3072,
                "sliding_window_non_causal": false,
                "transformer_layer_config": {
                    "hc_mult": 4,
                    "head_dim": 256,
                    "hidden_size": 4096,
                    "intermediate_size": 2048,
                    "layer_types": vec!["sliding_attention"; 5],
                    "num_attention_heads": 64,
                    "num_hidden_layers": 5,
                    "num_key_value_heads": 1,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
                    "sliding_window": 2048,
                    "vocab_size": 129280
                }
            })
            .to_string(),
        )
        .unwrap();
        let mut cfg = DFlashConfig::real_default();
        // Perturb everything the file should restore.
        cfg.window = None;
        cfg.causal = false;
        cfg.block_size = 4;
        cfg.aux_layers = vec![1, 2, 3, 4, 5];
        cfg.apply_config_json(&path).unwrap();
        assert_eq!(cfg.window, Some(2048));
        assert!(cfg.causal, "sliding_window_non_causal=false means causal");
        assert_eq!(cfg.block_size, 8);
        // vLLM's [i - 1] aux-id -> target-layer conversion (update_dflash).
        assert_eq!(cfg.aux_layers, vec![2, 12, 22, 31, 41]);
        assert_eq!(cfg.aux_width, 16384);
        assert_eq!(cfg.draft_vocab, 32000);
        assert_eq!(cfg.embed_vocab, 129_280);
        assert_eq!(cfg.mask_token, 1);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.rope_theta, 10_000.0);

        // Absent sliding_window_non_causal defaults to true -> non-causal
        // (vLLM update_dflash), and all-full layer_types drop the window.
        std::fs::write(
            &path,
            serde_json::json!({
                "transformer_layer_config": {
                    "layer_types": vec!["full_attention"; 5],
                }
            })
            .to_string(),
        )
        .unwrap();
        let mut cfg = DFlashConfig::real_default();
        cfg.apply_config_json(&path).unwrap();
        assert!(!cfg.causal);
        assert_eq!(cfg.window, None);

        // Mixed layer types are rejected.
        std::fs::write(
            &path,
            serde_json::json!({
                "transformer_layer_config": {
                    "layer_types": ["full_attention", "sliding_attention"],
                    "sliding_window": 64,
                }
            })
            .to_string(),
        )
        .unwrap();
        let mut cfg = DFlashConfig::real_default();
        assert!(cfg.apply_config_json(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dflash_embed_rows_decode_bf16_checkpoints() {
        let cfg = tiny_cfg();
        let path = tempfile_path("bf16");
        // Quarter-step values are exactly representable in bf16.
        let weights_gen = gen_with(0.4, |name, len| {
            if name == EMBED_TENSOR {
                Some((0..len).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect())
            } else {
                Some(
                    lcg_values(fnv(name), len, 0.4)
                        .into_iter()
                        .map(|v| bf16_to_f32((v.to_bits() >> 16) as u16))
                        .collect(),
                )
            }
        });
        write_checkpoint_as(
            &path,
            &cfg,
            &weights_gen,
            &vec![0i64; cfg.draft_vocab],
            true,
        );
        let drafter = DFlashDrafter::load_host(&path, cfg.clone()).unwrap();
        let row = drafter.embed_row(2).unwrap();
        let expected: Vec<f32> = (16..24).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
        assert_close(&row, &expected, 0.0, "bf16 embed row");
        std::fs::remove_file(&path).ok();
    }

    // -----------------------------------------------------------------------
    // Context-KV path
    // -----------------------------------------------------------------------

    #[test]
    fn dflash_context_kv_applies_fc_then_hidden_norm_then_kv_pipeline() {
        // hidden 4, one layer, identity projections. fc = 2*I so that a
        // wrongly-ordered hidden_norm(fc(x)) vs fc(hidden_norm(x)) differs by
        // exactly 2x (RMS norm is scale-invariant, so the correct order
        // cancels fc's factor).
        let cfg = DFlashConfig {
            hidden: 4,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 4,
            embed_vocab: 8,
            draft_vocab: 4,
            aux_layers: vec![0],
            aux_width: 4,
            rope_theta: 10_000.0,
            rms_eps: 1e-9,
            window: None,
            causal: true,
            block_size: 8,
            mask_token: 1,
            max_anchors: 64,
        };
        let weights_gen = gen_with(0.4, |name, len| match name {
            "fc.weight" => Some(eye(4, 2.0)),
            "hidden_norm.weight" => Some(vec![1.0, 2.0, 1.0, 2.0]),
            "layers.0.self_attn.k_proj.weight" | "layers.0.self_attn.v_proj.weight" => {
                Some(eye(4, 1.0))
            }
            "layers.0.self_attn.k_norm.weight" => Some(ones(len)),
            _ => None,
        });
        let path = tempfile_path("ctx-order");
        write_checkpoint(&path, &cfg, &weights_gen, &vec![0i64; 4]);
        let mut drafter = DFlashDrafter::load_host(&path, cfg).unwrap();
        std::fs::remove_file(&path).ok();

        // aux [1,2,2,4]: fc -> [2,4,4,8]; rms = 5 -> [0.4,0.8,0.8,1.6];
        // * hidden_norm -> [0.4,1.6,0.8,3.2] = the V row (identity v_proj).
        drafter.append_context(&[1.0, 2.0, 2.0, 4.0], 1, 0).unwrap();
        let expected_hidden = [0.4, 1.6, 0.8, 3.2];
        assert_close(&drafter.ctx_v[0], &expected_hidden, 1e-5, "ctx V");
        // K re-normalizes per head: ms = 13.6/4 = 3.4 -> inv 0.542326.
        let inv = (13.6f32 / 4.0).sqrt().recip();
        let expected_k: Vec<f32> = expected_hidden.iter().map(|v| v * inv).collect();
        // Position 0 rope is the identity rotation.
        assert_close(&drafter.ctx_k[0], &expected_k, 1e-5, "ctx K");
        // The wrong order (hidden_norm before fc) would have doubled V
        // (RMS is scale-invariant, so fc's 2x only survives if fc runs last).
        assert!((drafter.ctx_v[0][3] - 3.2).abs() < 1e-4);
        assert!((drafter.ctx_v[0][3] - 6.4).abs() > 1.0);
    }

    #[test]
    fn dflash_context_kv_rope_uses_absolute_positions_and_batches_match_incremental() {
        // head_dim 2 -> inv_freq [1]; identical aux rows at positions 0..3 so
        // the pre-rope K is constant and the stored rows are pure rotations.
        let cfg = DFlashConfig {
            hidden: 2,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            intermediate: 2,
            embed_vocab: 8,
            draft_vocab: 4,
            aux_layers: vec![0],
            aux_width: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-9,
            window: None,
            causal: true,
            block_size: 8,
            mask_token: 1,
            max_anchors: 64,
        };
        let weights_gen = gen_with(0.4, |name, len| match name {
            "fc.weight"
            | "layers.0.self_attn.k_proj.weight"
            | "layers.0.self_attn.v_proj.weight" => Some(eye(2, 1.0)),
            "hidden_norm.weight" | "layers.0.self_attn.k_norm.weight" => Some(ones(len)),
            _ => None,
        });
        let path = tempfile_path("ctx-rope");
        write_checkpoint(&path, &cfg, &weights_gen, &vec![0i64; 4]);
        let mut incremental = DFlashDrafter::load_host(&path, cfg.clone()).unwrap();
        let mut batch = DFlashDrafter::load_host(&path, cfg).unwrap();
        std::fs::remove_file(&path).ok();

        // aux [1,0] -> fc identity -> rms -> [sqrt(2), 0]; k_norm keeps it
        // (rms of [sqrt(2),0] is 1). Position p stores [√2 cos p, √2 sin p].
        let row = [1.0f32, 0.0];
        incremental.append_context(&row, 1, 0).unwrap();
        incremental
            .append_context(&[row, row].concat(), 2, 1)
            .unwrap();
        batch
            .append_context(&[row, row, row].concat(), 3, 0)
            .unwrap();

        let r2 = 2.0f32.sqrt();
        let mut expected = Vec::new();
        for pos in 0..3 {
            let (sin, cos) = (pos as f32).sin_cos();
            expected.extend([r2 * cos, r2 * sin]);
        }
        assert_close(&incremental.ctx_k[0], &expected, 1e-5, "incremental K");
        assert_close(&batch.ctx_k[0], &expected, 1e-5, "batch K");
        assert_close(&incremental.ctx_v[0], &batch.ctx_v[0], 1e-6, "V parity");
    }

    // -----------------------------------------------------------------------
    // Mask-query block
    // -----------------------------------------------------------------------

    #[test]
    fn dflash_mask_block_draft_count_and_d2t_offset_mapping() {
        // Null out attention (v_proj = 0) and the MLP (gate = 0): every mask
        // row keeps exactly the mask embedding, so the draft logits are
        // lm_head * rms(embed[mask]) and every draft is the same id.
        let cfg = DFlashConfig {
            hidden: 4,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 4,
            embed_vocab: 16,
            draft_vocab: 5,
            aux_layers: vec![0],
            aux_width: 4,
            rope_theta: 10_000.0,
            rms_eps: 1e-9,
            window: None,
            causal: true,
            block_size: 8,
            mask_token: 1,
            max_anchors: 64,
        };
        let weights_gen = gen_with(0.4, |name, len| match name {
            "layers.0.self_attn.v_proj.weight" | "layers.0.mlp.gate_proj.weight" => {
                Some(zeros(len))
            }
            "norm.weight" => Some(ones(len)),
            EMBED_TENSOR => {
                let mut table = lcg_values(fnv(name), len, 0.4);
                // mask row (id 1) = [2,0,0,0] -> rms-normalized [2,0,0,0].
                table[4..8].copy_from_slice(&[2.0, 0.0, 0.0, 0.0]);
                Some(table)
            }
            "lm_head.weight" => {
                // Column 0 decides: logits = 2 * [0,1,2,5,3] -> argmax id 3.
                let mut head = zeros(len);
                for (row, &value) in [0.0, 1.0, 2.0, 5.0, 3.0].iter().enumerate() {
                    head[row * 4] = value;
                }
                Some(head)
            }
            _ => None,
        });
        let path = tempfile_path("block-d2t");
        // d2t[3] = 7 -> target id 10.
        write_checkpoint(&path, &cfg, &weights_gen, &[0, 0, 0, 7, 0]);
        let mut drafter = DFlashDrafter::load_host(&path, cfg.clone()).unwrap();
        std::fs::remove_file(&path).ok();

        let tokens = [4u32, 9, 3];
        let aux = owned_aux(&cfg.aux_layers, cfg.aux_width, &tokens[..2]);
        let drafts = drafter.propose_inner(&tokens, &aux, 3).unwrap();
        assert_eq!(drafts, vec![10, 10, 10], "d2t offset applied to argmax 3");

        // K is capped by block_size - 1, and k = 0 proposes nothing.
        let drafts = drafter.propose_inner(&tokens, &aux, 100).unwrap();
        assert_eq!(drafts.len(), 7);
        assert!(drafter.propose_inner(&tokens, &aux, 0).unwrap().is_empty());

        // Token/tap disagreement is an error, not garbage drafts.
        assert!(drafter.propose_inner(&tokens[..2], &aux, 3).is_err());
    }

    #[test]
    fn dflash_single_layer_block_attention_hand_computed() {
        // Full by-hand forward: hidden 2, one layer, one head, head_dim 2,
        // identity projections, MLP nulled, lm_head = identity. Context is a
        // single position built from aux [0,1]; the block is [anchor, mask]
        // at positions 1 and 2 under CAUSAL full attention.
        let cfg = DFlashConfig {
            hidden: 2,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            intermediate: 2,
            embed_vocab: 4,
            draft_vocab: 2,
            aux_layers: vec![0],
            aux_width: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-9,
            window: None,
            causal: true,
            block_size: 8,
            mask_token: 1,
            max_anchors: 64,
        };
        let weights_gen = gen_with(0.4, |name, len| match name {
            "fc.weight"
            | "layers.0.self_attn.q_proj.weight"
            | "layers.0.self_attn.k_proj.weight"
            | "layers.0.self_attn.v_proj.weight"
            | "layers.0.self_attn.o_proj.weight"
            | "lm_head.weight" => Some(eye(2, 1.0)),
            "hidden_norm.weight"
            | "norm.weight"
            | "layers.0.input_layernorm.weight"
            | "layers.0.post_attention_layernorm.weight"
            | "layers.0.self_attn.q_norm.weight"
            | "layers.0.self_attn.k_norm.weight" => Some(ones(len)),
            "layers.0.mlp.gate_proj.weight" => Some(zeros(len)),
            EMBED_TENSOR => Some(vec![
                0.0, 0.0, // id 0
                1.0, 0.0, // id 1 = mask
                1.0, 1.0, // id 2 = anchor
                0.0, 1.0, // id 3
            ]),
            _ => None,
        });
        let path = tempfile_path("hand-attn");
        write_checkpoint(&path, &cfg, &weights_gen, &[0, 0]);
        let mut drafter = DFlashDrafter::load_host(&path, cfg.clone()).unwrap();
        std::fs::remove_file(&path).ok();

        // Context from aux row [0,1]: fc identity -> rms -> [0, sqrt(2)];
        // k_norm keeps it; rope at pos 0 is identity.
        let aux = OwnedAux {
            layers: vec![0],
            base: 0,
            rows: vec![vec![vec![0.0, 1.0]]],
        };
        let tokens = [0u32, 2]; // one context token, anchor id 2
        let drafts = drafter.propose_inner(&tokens, &aux, 1).unwrap();
        assert_eq!(drafts.len(), 1);

        // ---- Independent hand computation ----
        let r2 = 2.0f32.sqrt();
        let ctx_k = [0.0, r2];
        let ctx_v = [0.0, r2];
        // Rows after input_layernorm (pure RMS): anchor [1,1] -> [1,1];
        // mask [1,0] -> [sqrt(2),0]. q_norm/k_norm leave both unchanged.
        let (s1, c1) = 1.0f32.sin_cos();
        let (s2, c2) = 2.0f32.sin_cos();
        // Roped q/k at pos 1 (anchor) and pos 2 (mask), inv_freq = [1].
        let q0 = [c1 - s1, s1 + c1];
        let k0 = q0;
        let q1 = [r2 * c2, r2 * s2];
        let k1 = q1;
        let v0 = [1.0, 1.0];
        let v1 = [r2, 0.0];
        let scale = 1.0 / r2;
        // Anchor row (pos 1): sees ctx (pos 0) and itself; the mask (pos 2)
        // is causally hidden. (Feeds nothing we assert, but part of the math.)
        // Mask row (pos 2): sees all three keys.
        let s10 = (q1[0] * ctx_k[0] + q1[1] * ctx_k[1]) * scale;
        let s11 = (q1[0] * k0[0] + q1[1] * k0[1]) * scale;
        let s12 = (q1[0] * k1[0] + q1[1] * k1[1]) * scale;
        let max = s10.max(s11).max(s12);
        let (e0, e1, e2) = ((s10 - max).exp(), (s11 - max).exp(), (s12 - max).exp());
        let sum = e0 + e1 + e2;
        let (p0, p1, p2) = (e0 / sum, e1 / sum, e2 / sum);
        let attn = [
            p0 * ctx_v[0] + p1 * v0[0] + p2 * v1[0],
            p0 * ctx_v[1] + p1 * v0[1] + p2 * v1[1],
        ];
        // Residual (o_proj identity, MLP nulled), then final RMS norm.
        let x = [1.0 + attn[0], 0.0 + attn[1]];
        let ms = (x[0] * x[0] + x[1] * x[1]) / 2.0;
        let inv = ms.sqrt().recip();
        let expected_logits = [x[0] * inv, x[1] * inv];

        assert_eq!(drafter.last_mask_logits.len(), 1);
        assert_close(
            &drafter.last_mask_logits[0],
            &expected_logits,
            1e-4,
            "hand-computed mask logits",
        );
        let expected_draft = argmax_row(&expected_logits) as u32;
        assert_eq!(drafts[0], expected_draft);
    }

    #[test]
    fn dflash_swa_window_and_causal_flag_change_draft_logits() {
        // Same weights + context; only the mask semantics differ. The window
        // hides early context (w=2 over 5 positions) and the causal flag
        // hides in-block future masks, so both must move the logits.
        let base = tiny_cfg();
        let tokens = [3u32, 4, 5, 6, 7, 2];
        let aux = owned_aux(&base.aux_layers, base.aux_width, &tokens[..5]);
        let mut logits = Vec::new();
        for (window, causal) in [(Some(4), true), (Some(2), true), (Some(4), false)] {
            let mut cfg = base.clone();
            cfg.window = window;
            cfg.causal = causal;
            let mut drafter = load_tiny("swa-flags", &cfg);
            let drafts = drafter.propose_inner(&tokens, &aux, 3).unwrap();
            assert_eq!(drafts.len(), 3);
            logits.push(drafter.last_mask_logits.clone());
        }
        let max_delta = |a: &[Vec<f32>], b: &[Vec<f32>]| {
            a.iter()
                .flatten()
                .zip(b.iter().flatten())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            max_delta(&logits[0], &logits[1]) > 1e-4,
            "shrinking the window must change what context is attended"
        );
        assert!(
            max_delta(&logits[0], &logits[2]) > 1e-4,
            "the causal flag must change in-block visibility"
        );
    }

    // -----------------------------------------------------------------------
    // Incremental bookkeeping
    // -----------------------------------------------------------------------

    #[test]
    fn dflash_incremental_insert_divergence_and_new_request_reuse() {
        let cfg = tiny_cfg();
        let mut drafter = load_tiny("incremental", &cfg);
        let kv_dim = cfg.kv_dim();

        // Fresh request: 3 context positions inserted.
        let t1 = [10u32, 11, 12, 5];
        let aux1 = owned_aux(&cfg.aux_layers, cfg.aux_width, &t1[..3]);
        let d1 = drafter.propose_inner(&t1, &aux1, 4).unwrap();
        assert_eq!(d1.len(), 4);
        assert_eq!(drafter.rows_inserted, 3);
        assert_eq!(drafter.inserted, &t1[..3]);
        assert_eq!(drafter.ctx_k[0].len(), 3 * kv_dim);

        // The verify loop accepted 2 tokens: only the delta is inserted.
        let t2 = [10u32, 11, 12, 5, 13, 6];
        let aux2 = owned_aux(&cfg.aux_layers, cfg.aux_width, &t2[..5]);
        drafter.propose_inner(&t2, &aux2, 4).unwrap();
        assert_eq!(drafter.rows_inserted, 5, "only 2 new rows projected");
        assert_eq!(drafter.ctx_k[0].len(), 5 * kv_dim);
        assert_eq!(drafter.truncations, 0);

        // A different conversation sharing one token of prefix: truncate to
        // the common prefix and rebuild the rest.
        let t3 = [10u32, 3, 2, 5];
        let aux3 = owned_aux(&cfg.aux_layers, cfg.aux_width, &t3[..3]);
        let d3 = drafter.propose_inner(&t3, &aux3, 4).unwrap();
        assert_eq!(drafter.truncations, 1);
        assert_eq!(drafter.rows_inserted, 7, "kept 1, inserted 2");
        assert_eq!(drafter.inserted, &t3[..3]);

        // Incremental state == a fresh drafter fed the same final context.
        let mut fresh = load_tiny("incremental-fresh", &cfg);
        let df = fresh.propose_inner(&t3, &aux3, 4).unwrap();
        assert_eq!(d3, df, "incremental and fresh drafts agree");
        for l in 0..cfg.layers {
            assert_close(
                &drafter.ctx_k[l],
                &fresh.ctx_k[l],
                1e-6,
                "incremental ctx K",
            );
            assert_close(
                &drafter.ctx_v[l],
                &fresh.ctx_v[l],
                1e-6,
                "incremental ctx V",
            );
        }

        // A shorter re-prompt of the same conversation truncates without
        // re-inserting anything (cross-request context reuse).
        let t4 = [10u32, 3, 9];
        let aux4 = owned_aux(&cfg.aux_layers, cfg.aux_width, &t4[..2]);
        drafter.propose_inner(&t4, &aux4, 4).unwrap();
        assert_eq!(drafter.rows_inserted, 7, "no new rows");
        assert_eq!(drafter.ctx_k[0].len(), 2 * kv_dim);
    }

    /// Restore-based taps: context building floors at the taps base (no row
    /// below it is ever fabricated), rows already built below the base stay
    /// reusable across requests, divergence below the base rebases instead
    /// of misaligning, and a base outside every query's SWA window leaves the
    /// drafts BIT-IDENTICAL to a full-context run (the missing rows were
    /// invisible anyway).
    #[test]
    fn dflash_context_floors_at_taps_base() {
        let cfg = tiny_cfg();
        let kv_dim = cfg.kv_dim();

        // (a) Base outside the window: identical drafts. tokens = 8 context
        // + anchor; window 4 => the anchor at pos 8 sees keys 5.. only, so a
        // base of 3 hides nothing visible.
        let tokens = [10u32, 11, 12, 5, 13, 6, 9, 3, 2];
        let n_ctx = tokens.len() - 1;
        let mut full = load_tiny("base-full", &cfg);
        let aux_full = owned_aux(&cfg.aux_layers, cfg.aux_width, &tokens[..n_ctx]);
        let drafts_full = full.propose_inner(&tokens, &aux_full, 4).unwrap();

        let base = 3usize;
        let mut floored = load_tiny("base-floored", &cfg);
        let aux_based = owned_aux_at(&cfg.aux_layers, cfg.aux_width, &tokens[..n_ctx], base);
        assert!(aux_based.aux_row(0, base - 1).is_none());
        let drafts_based = floored.propose_inner(&tokens, &aux_based, 4).unwrap();
        assert_eq!(
            drafts_based, drafts_full,
            "rows below an out-of-window base must not change the drafts"
        );
        assert_eq!(floored.ctx_base, base);
        assert_eq!(floored.inserted, &tokens[base..n_ctx]);
        assert_eq!(floored.ctx_k[0].len(), (n_ctx - base) * kv_dim);

        // (b) Base inside the window: still drafts (fewer context keys — the
        // documented approximation), flooring the build at the base.
        let deep_base = 6usize;
        let mut deep = load_tiny("base-deep", &cfg);
        let aux_deep = owned_aux_at(&cfg.aux_layers, cfg.aux_width, &tokens[..n_ctx], deep_base);
        let drafts_deep = deep.propose_inner(&tokens, &aux_deep, 4).unwrap();
        assert_eq!(drafts_deep.len(), 4);
        assert_eq!(deep.ctx_base, deep_base);
        assert_eq!(deep.ctx_k[0].len(), (n_ctx - deep_base) * kv_dim);

        // (c) Warm cross-request reuse: rows built with full taps stay valid
        // below a LATER request's base — the extended request only appends
        // the delta, and the result equals a fresh full-context drafter.
        let mut warm = load_tiny("base-warm", &cfg);
        let turn1 = &tokens[..7];
        let aux1 = owned_aux(&cfg.aux_layers, cfg.aux_width, &turn1[..6]);
        warm.propose_inner(turn1, &aux1, 4).unwrap();
        assert_eq!(warm.rows_inserted, 6);
        let aux2 = owned_aux_at(&cfg.aux_layers, cfg.aux_width, &tokens[..n_ctx], 5);
        let drafts_warm = warm.propose_inner(&tokens, &aux2, 4).unwrap();
        assert_eq!(warm.ctx_base, 0, "reusable rows below the base survive");
        assert_eq!(warm.rows_inserted, 8, "only the delta is appended");
        assert_eq!(drafts_warm, drafts_full, "reused context equals full");

        // (d) Divergence BELOW the new base: nothing at/above the base can be
        // trusted to align, so the cache rebases there and rebuilds.
        let mut other = tokens;
        other[1] = 7; // diverges below the base of 5
        let aux3 = owned_aux_at(&cfg.aux_layers, cfg.aux_width, &other[..n_ctx], 5);
        let drafts_rebased = warm.propose_inner(&other, &aux3, 4).unwrap();
        assert_eq!(drafts_rebased.len(), 4);
        assert_eq!(warm.ctx_base, 5, "divergence below the base rebases");
        assert_eq!(warm.inserted, &other[5..n_ctx]);
        assert_eq!(warm.ctx_k[0].len(), (n_ctx - 5) * kv_dim);

        // (e) A restore GAP: the new base lies beyond every built row, so the
        // kept prefix cannot reach it contiguously — rebase, never bail on
        // the missing rows in between.
        let mut extended = other.to_vec();
        extended.extend([4u32, 8]);
        let n_ext = extended.len() - 1; // 10 context positions
        let gap_base = n_ext - 1; // rows end at 8, base 9 leaves a gap
        let aux4 = owned_aux_at(&cfg.aux_layers, cfg.aux_width, &extended[..n_ext], gap_base);
        let drafts_gap = warm.propose_inner(&extended, &aux4, 4).unwrap();
        assert_eq!(drafts_gap.len(), 4);
        assert_eq!(warm.ctx_base, gap_base, "gap must rebase, not bail");
        assert_eq!(warm.inserted, &extended[gap_base..n_ext]);
        assert_eq!(warm.ctx_k[0].len(), (n_ext - gap_base) * kv_dim);
    }

    #[test]
    fn dflash_propose_rejects_width_and_vocab_mismatches() {
        let cfg = tiny_cfg();
        let mut drafter = load_tiny("mismatch", &cfg);

        // Wrong tap width (engine hc*embed disagrees with the fc).
        let tokens = [1u32, 2, 3];
        let bad = owned_aux(&cfg.aux_layers, cfg.aux_width - 1, &tokens[..2]);
        let err = format!("{:#}", drafter.propose_inner(&tokens, &bad, 4).unwrap_err());
        assert!(err.contains("wide"), "got: {err}");

        // Anchor outside the drafter's embedding table.
        let aux = owned_aux(&cfg.aux_layers, cfg.aux_width, &tokens[..2]);
        let far = [1u32, 2, 4000];
        let err = format!("{:#}", drafter.propose_inner(&far, &aux, 4).unwrap_err());
        assert!(err.contains("embed vocab"), "got: {err}");
    }

    // -----------------------------------------------------------------------
    // GPU provider parity + verify-loop integration (need a CUDA device)
    // -----------------------------------------------------------------------

    #[cfg(feature = "native-cuda")]
    mod native {
        use futures_util::StreamExt;
        use hi_local_core::backend::{GenerationEvent, GenerationRequest, InferenceBackend};

        use super::*;
        use crate::dsv4_backend::DeepSeekV4Backend;
        use crate::dsv4_cpu::fixture::write_deepseek4_spec_gguf;

        /// GQA geometry (4 q heads over 2 kv heads) with an SWA window that
        /// genuinely truncates the 9-position context.
        fn parity_cfg() -> DFlashConfig {
            DFlashConfig {
                hidden: 32,
                layers: 3,
                heads: 4,
                kv_heads: 2,
                head_dim: 8,
                intermediate: 16,
                embed_vocab: 24,
                draft_vocab: 12,
                aux_layers: vec![0, 1],
                aux_width: 12,
                rope_theta: 10_000.0,
                rms_eps: 1e-6,
                window: Some(6),
                causal: true,
                block_size: 8,
                mask_token: 1,
                max_anchors: 64,
            }
        }

        #[test]
        fn dflash_gpu_matches_cpu_reference_on_synthetic_checkpoint() {
            let cfg = parity_cfg();
            let path = tempfile_path("parity");
            let d2t: Vec<i64> = (0..cfg.draft_vocab as i64).map(|i| i % 3).collect();
            write_checkpoint(&path, &cfg, &gen_with(0.3, |_, _| None), &d2t);
            let mut host = DFlashDrafter::load_host(&path, cfg.clone()).unwrap();
            let mut gpu = DFlashDrafter::load_gpu(&path, cfg.clone()).unwrap();
            std::fs::remove_file(&path).ok();
            assert!(gpu.provider.resident_bytes() > 0);

            // Two proposals in lockstep: the first inserts 5 context rows,
            // the second appends 4 more (incremental on both providers).
            let tokens_a = [7u32, 8, 9, 10, 11, 3];
            let tokens_b = [7u32, 8, 9, 10, 11, 3, 12, 13, 14, 4];
            for tokens in [&tokens_a[..], &tokens_b[..]] {
                let n_ctx = tokens.len() - 1;
                let aux = owned_aux(&cfg.aux_layers, cfg.aux_width, &tokens[..n_ctx]);
                let host_drafts = host.propose_inner(tokens, &aux, 7).unwrap();
                let gpu_drafts = gpu.propose_inner(tokens, &aux, 7).unwrap();
                assert_eq!(host_drafts.len(), 7);
                assert_eq!(host_drafts, gpu_drafts, "draft ids must agree");
                for (h, g) in host.last_mask_logits.iter().zip(&gpu.last_mask_logits) {
                    assert_close(g, h, 0.06, "cpu/gpu mask logits");
                }
                for l in 0..cfg.layers {
                    assert_close(&gpu.ctx_k[l], &host.ctx_k[l], 5e-3, "ctx K parity");
                    assert_close(&gpu.ctx_v[l], &host.ctx_v[l], 5e-3, "ctx V parity");
                }
            }
        }

        #[test]
        fn dflash_tap_config_requests_checkpoint_aux_layers() {
            let drafter = load_tiny("tap-config", &tiny_cfg());
            let tap = Drafter::tap_config(&drafter);
            assert!(!tap.pre_hc_head);
            assert_eq!(tap.aux_layers, vec![0, 1]);
            assert_eq!(
                DFlashConfig::real_default().aux_layers,
                DFLASH_DEFAULT_AUX_LAYERS.to_vec()
            );
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
            }
        }

        async fn collect_tokens(
            backend: &DeepSeekV4Backend,
            prompt: &str,
            max_tokens: u32,
        ) -> (Vec<u32>, String) {
            let mut stream = backend
                .stream_generate(generation_request(prompt, max_tokens))
                .await
                .unwrap();
            let mut ids = Vec::new();
            let mut finished = None;
            while let Some(event) = stream.next().await {
                match event.unwrap() {
                    GenerationEvent::TokenDelta { token_id, .. } => ids.push(token_id),
                    GenerationEvent::Finished { output } => finished = Some(output.text),
                }
            }
            (ids, finished.expect("stream must finish"))
        }

        fn health_stat(quantization: &str, key: &str) -> u64 {
            let start = quantization
                .find(key)
                .unwrap_or_else(|| panic!("{key} missing in {quantization}"))
                + key.len();
            quantization[start..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .unwrap()
        }

        /// End-to-end wiring over the fixture engine: a real (synthetic
        /// checkpoint) DFlash drafter feeds the greedy verify loop, whose
        /// output must be byte-identical to the drafter-less backend, with
        /// taps flowing (proposals observed in /health stats).
        #[tokio::test]
        async fn dflash_fixture_backend_output_identical_with_dflash_drafter() {
            // The fixture engine: hc 2 x embed 4 -> flat tap rows 8 wide,
            // 3 target layers, spec vocab a b c d (ids 0..=3, eos d).
            let gguf = crate::dsv4_cpu::fixture::tempfile_path("dflash-fixture");
            write_deepseek4_spec_gguf(std::path::Path::new(&gguf));

            let cfg = DFlashConfig {
                hidden: 8,
                layers: 2,
                heads: 2,
                kv_heads: 1,
                head_dim: 4,
                intermediate: 4,
                embed_vocab: 4,
                draft_vocab: 4,
                aux_layers: vec![0, 2],
                aux_width: 8,
                rope_theta: 10_000.0,
                rms_eps: 1e-6,
                window: Some(4),
                causal: true,
                block_size: 8,
                mask_token: 1,
                max_anchors: 64,
            };
            let checkpoint = tempfile_path("fixture-drafter");
            write_checkpoint(
                &checkpoint,
                &cfg,
                &gen_with(0.4, |_, _| None),
                &[0, 0, 0, 0],
            );

            let factory_ckpt = checkpoint.clone();
            let factory_cfg = cfg.clone();
            let spec = DeepSeekV4Backend::load_with_drafter(
                &gguf,
                Some("dsv4-dflash".to_string()),
                8,
                1 << 20,
                Box::new(move |_| {
                    Some(
                        Box::new(DFlashDrafter::load_host(&factory_ckpt, factory_cfg).unwrap())
                            as Box<dyn Drafter>,
                    )
                }),
            )
            .unwrap();
            let plain = DeepSeekV4Backend::load_with_prefix_config(
                &gguf,
                Some("dsv4-plain".to_string()),
                8,
                1 << 20,
            )
            .unwrap();

            for prompt in ["abcab", "aabbccab", "cab"] {
                let (spec_ids, spec_text) = collect_tokens(&spec, prompt, 12).await;
                let (plain_ids, plain_text) = collect_tokens(&plain, prompt, 12).await;
                assert_eq!(spec_ids, plain_ids, "prompt {prompt:?}");
                assert_eq!(spec_text, plain_text, "prompt {prompt:?}");
            }

            let health = spec.health().quantization;
            assert!(health_stat(&health, "spec_verify_steps=") > 0, "{health}");
            assert!(health_stat(&health, "spec_proposed=") > 0, "{health}");

            std::fs::remove_file(&gguf).ok();
            std::fs::remove_file(&checkpoint).ok();
        }

        // -------------------------------------------------------------------
        // Real-checkpoint gates (ignored: need the artifacts + a free GPU)
        // -------------------------------------------------------------------

        fn real_dflash_path() -> Option<PathBuf> {
            let path = dflash_checkpoint_path();
            path.exists().then_some(path)
        }

        fn real_gguf_path() -> Option<PathBuf> {
            let home = std::env::var_os("HOME")?;
            let path = PathBuf::from(home).join(
                ".hi/models/deepseek-v4-flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf",
            );
            path.exists().then_some(path)
        }

        /// Loads the real 3.6 GB checkpoint onto the GPU and times proposals
        /// against fabricated taps (real shapes, nonsense values): the
        /// context-KV insert cost for a 512-token prompt and the steady-state
        /// per-propose latency, independent of the target model.
        ///
        /// `CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release --features
        ///  native-cuda dflash_real_checkpoint_gpu_load_and_propose_latency
        ///  -- --ignored --nocapture`
        #[test]
        #[ignore = "needs the RedHat DFlash checkpoint and ~2.6 GiB of free VRAM"]
        fn dflash_real_checkpoint_gpu_load_and_propose_latency() {
            let Some(path) = real_dflash_path() else {
                eprintln!("skipping: DFlash checkpoint not found");
                return;
            };
            let load_start = Instant::now();
            let mut drafter = build_gpu_drafter(&path).unwrap();
            let cfg = drafter.cfg.clone();
            eprintln!(
                "loaded in {:.1}s: {:.2} GiB resident",
                load_start.elapsed().as_secs_f64(),
                drafter.provider.resident_bytes() as f64 / f64::from(1u32 << 30)
            );
            assert_eq!(cfg.layers, 5);
            assert_eq!(cfg.heads, 64);
            assert_eq!(cfg.head_dim, 256);
            assert_eq!(cfg.window, Some(2048));
            assert!(cfg.causal, "RedHat config: sliding_window_non_causal=false");
            assert_eq!(cfg.block_size, 8);
            assert_eq!(cfg.aux_width, 16384);

            let n_ctx = 512usize;
            let tokens: Vec<u32> = (0..=n_ctx as u32).map(|i| 1000 + i).collect();
            let aux = owned_aux(&cfg.aux_layers, cfg.aux_width, &tokens[..n_ctx]);

            let insert_start = Instant::now();
            let drafts = drafter.propose_inner(&tokens, &aux, 7).unwrap();
            eprintln!(
                "first propose ({} context rows + block): {:.1} ms, drafts {:?}",
                n_ctx,
                insert_start.elapsed().as_secs_f64() * 1e3,
                drafts
            );
            assert_eq!(drafts.len(), 7);
            assert!(drafts.iter().all(|&t| (t as usize) < cfg.embed_vocab));

            let steady_start = Instant::now();
            let reps = 10;
            for _ in 0..reps {
                let drafts = drafter.propose_inner(&tokens, &aux, 7).unwrap();
                assert_eq!(drafts.len(), 7);
            }
            eprintln!(
                "steady-state propose (no new context): {:.2} ms avg over {reps}",
                steady_start.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
            );
        }

        /// The full Stage-C acceptance gate: real target + real drafter, one
        /// backend, the same prompt decoded with K=7 speculation and then
        /// with K=0 (sequential-equivalent). Output must be identical and the
        /// health stats report the acceptance rate (a faithful port sees ~3-4
        /// accepted per verify step on natural text).
        ///
        /// `CUDA_VISIBLE_DEVICES=0 HI_DSV4_EXPERT_POOL_GB=16 cargo test -p
        ///  hi-cuda --release --features native-cuda
        ///  dflash_real_model_e2e_acceptance -- --ignored --nocapture`
        /// (shrink the pool if the GPU is shared).
        #[tokio::test]
        #[ignore = "needs the real DeepSeek-V4-Flash checkpoint + DFlash drafter and tens of GB of VRAM"]
        async fn dflash_real_model_e2e_acceptance() {
            let Some(gguf) = real_gguf_path() else {
                eprintln!("skipping: real V4-Flash GGUF not found");
                return;
            };
            let Some(ckpt) = real_dflash_path() else {
                eprintln!("skipping: DFlash checkpoint not found");
                return;
            };
            let backend = DeepSeekV4Backend::load_with_drafter(
                &gguf,
                Some("dsv4-real".to_string()),
                256,
                64 << 20,
                Box::new(move |_| match build_gpu_drafter(&ckpt) {
                    Ok(drafter) => Some(Box::new(drafter) as Box<dyn Drafter>),
                    Err(err) => {
                        eprintln!("drafter load failed: {err:#}");
                        None
                    }
                }),
            )
            .unwrap();

            // One raw-text continuation plus one prompt in the exact shape the
            // serving path renders (`render_deepseek_v4_template`): the
            // drafter is trained on chat traces, so acceptance on templated
            // prompts is the production-relevant number.
            let raw_prompt = "The Rust programming language guarantees memory safety without a \
                              garbage collector. It achieves this through an ownership system: \
                              every value has a single owner, and";
            let chat_prompt = "<｜begin▁of▁sentence｜><｜User｜>Explain in a few sentences how \
                               Rust's ownership system guarantees memory safety without a \
                               garbage collector.<｜Assistant｜></think>";
            let max_tokens = 48;

            for (label, prompt) in [("raw", raw_prompt), ("chat", chat_prompt)] {
                // Speculative run (K = 7). Stats are cumulative across every
                // run (the K = 0 runs still count verify steps), so diff the
                // health counters around exactly this generation.
                let before = backend.health().quantization;
                unsafe { std::env::set_var("HI_DSV4_SPEC_K", "7") };
                let spec_start = Instant::now();
                let (spec_ids, spec_text) = collect_tokens(&backend, prompt, max_tokens).await;
                let spec_secs = spec_start.elapsed().as_secs_f64();
                let health = backend.health().quantization;
                let stat = |key: &str| health_stat(&health, key) - health_stat(&before, key);
                let proposed = stat("spec_proposed=");
                let accepted = stat("spec_accepted=");
                let steps = stat("spec_verify_steps=");

                // Sequential-equivalent run (K = 0 degrades the loop to
                // 1-token verify chunks; emitted tokens are the argmax chain
                // either way).
                unsafe { std::env::set_var("HI_DSV4_SPEC_K", "0") };
                let seq_start = Instant::now();
                let (seq_ids, seq_text) = collect_tokens(&backend, prompt, max_tokens).await;
                let seq_secs = seq_start.elapsed().as_secs_f64();
                unsafe { std::env::remove_var("HI_DSV4_SPEC_K") };

                assert_eq!(spec_ids, seq_ids, "speculative output must be lossless");
                assert_eq!(spec_text, seq_text);
                eprintln!("[{label}] text: {spec_text:?}");
                eprintln!(
                    "[{label}] acceptance: {accepted}/{proposed} drafts over {steps} verify \
                     steps ({:.2} accepted/step, {:.1}% of proposed)",
                    accepted as f64 / steps.max(1) as f64,
                    100.0 * accepted as f64 / proposed.max(1) as f64,
                );
                eprintln!(
                    "[{label}] decode wall: spec {:.1}s vs sequential {:.1}s for {} tokens \
                     ({:.2}x)",
                    spec_secs,
                    seq_secs,
                    spec_ids.len(),
                    seq_secs / spec_secs.max(1e-9),
                );
            }
        }
    }
}
