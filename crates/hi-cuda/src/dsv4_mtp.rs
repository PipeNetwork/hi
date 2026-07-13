//! DeepSeek-V4-Flash MTP self-speculation drafter — Stage B of
//! `docs/deepseek-v4-spec-decode-plan.md`.
//!
//! Owns: loading the official `mtp.0.*` safetensors shard (via
//! [`crate::safetensors`]), the compress-ratio-1 draft layer forward
//! (SWA-128 latent MQA + sinks, hyperconnections, 256-expert MoE — no
//! indexer/compressor), and the [`crate::dsv4_backend::Drafter`]
//! implementation selected by `HI_DSV4_SPEC=mtp`.
//!
//! # The module (ported from vLLM `models/deepseek_v4/nvidia/mtp.py`)
//!
//! One full V4 decoder layer at compress_ratio = 1 — the same math as the
//! trunk's ratio-0 layers (raw sliding-window ring, rope base
//! `rope.freq_base`, attention sinks, no compressor/indexer), with the MTP
//! module's own input/output shells:
//!
//! - Input per slot: `h_proj(hnorm(prev_stream_s)) + e_proj(enorm(embed))`
//!   per hc stream `s` — `prev` is the target's flat pre-hc-head residual
//!   `(hc·embed)` reshaped to streams, and the single `e_proj` activation
//!   broadcasts across streams (vLLM's `unsqueeze(-2)`). The embedding is
//!   zero-masked at rope position 0 (`fused_mtp_input_rmsnorm`).
//! - The decoder block runs exactly [`crate::dsv4_cpu::DsV4Engine`]'s
//!   per-layer body (hc_attn pre/post around attention, hc_ffn pre/post
//!   around the 256-expert MoE with `gate.bias` selection bias and the
//!   trunk's swiglu clamps); vLLM defers the final `mhc_post` to the caller,
//!   which is the same composition.
//! - Output: the flat post-block streams are the module's own pre-hc-head
//!   residual (re-fed for K>1 recurrence); logits apply the module's own
//!   `hc_head_{fn,base,scale}` collapse + `norm`, then the TARGET's lm head
//!   (shard 46 carries no embedding/head — both bind the GGUF's tensors).
//!
//! # The (token, hidden, position) pairing — where MTP ports die
//!
//! Derived from vLLM `v1/spec_decode/llm_base_proposer.py`:
//! `set_inputs_first_pass` (lines 838-859) left-shifts the target token ids
//! and writes the sampled token into each request's last slot while keeping
//! `target_positions` and `target_hidden_states` UNSHIFTED; the K>1 loop
//! (lines 682-767) then feeds each drafted token with the module's own
//! previous flat output at position+1. In this engine's terms, with
//! `tokens = prompt + emitted + pending` (length n) and
//! `taps.pre_hc_head(i)` = the target's residual after forwarding
//! `tokens[i]` at position i:
//!
//! ```text
//! draft slot i  =  ( embed(tokens[i+1]),  taps.pre_hc_head(i),  rope pos i )
//! ```
//!
//! for i in 0..n-1 — the LAST real slot (i = n-2) pairs the pending token's
//! embedding with the residual of the token before it, and its logits argmax
//! is the first draft (the token at position n, continuing AFTER pending).
//! Speculative slot t >= 1 then feeds (embed(draft_{t-1}), own flat output,
//! rope pos n-2+t). The slot at rope position 0 zero-masks its embedding,
//! exactly like vLLM. `mtp_pairing_alignment_produces_expected_draft` pins
//! this alignment against a straight-line derivation and rejects both
//! off-by-one hypotheses.
//!
//! # Weights and residency
//!
//! `HI_DSV4_MTP_PATH` (default
//! `~/.hi/models/deepseek-v4-flash/mtp/model-00046-of-00046.safetensors`)
//! is loaded through [`crate::safetensors`]: FP8-e4m3 + ue8m0 128x128 block
//! scales dequantize to f16 (~335 MB resident), the BF16 router stays bf16,
//! small norms/mixers/sinks materialize f32 host-side, and the 256 fp4
//! experts repack bit-exactly into the GGUF MXFP4 layout (~3.42 GB) and
//! register in the trunk's [`crate::dsv4_gpu`] expert pool as layer 43 with
//! PERMANENT (never-evicted) slots — every draft touches them.

// The production consumer (the drafter) is native-cuda-gated and the CPU
// reference paths are exercised by tests; a bare default build uses nothing
// here, which is expected rather than a smell.
#![cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{GgufFile, GgufTensorType};

use crate::dsv4_cpu::{
    DsV4Engine, DsV4Geometry, DsV4HcFunc, DsV4Layer, DsV4LayerState, DsV4Linear, DsV4MoeBlockCtx,
    DsV4MoeShared, DsV4Taps, HcParams, HcWeights, RawExperts, RawMatrix, SharedExpertWeights,
    TensorKey, dsv4_attention_step, dsv4_embed_row, hc_post, hc_pre_math, hyper_head_math,
};
use crate::qwen_cpu::{argmax, dot, rms_norm_in_place};
use crate::safetensors::SafetensorsFile;

#[cfg(feature = "native-cuda")]
use crate::dsv4_backend::{DraftContext, Drafter};
#[cfg(feature = "native-cuda")]
use crate::dsv4_gpu::{DeepSeekV4GpuEngine, DsV4GpuLinear, HostDenseData};

/// Every dimension the loader validates shard shapes against and the module
/// forwards with. Copied from the trunk engine (the MTP block shares the
/// trunk's geometry exactly); constructible directly for the real-shard
/// census test.
#[derive(Clone, Debug)]
pub(crate) struct MtpDims {
    pub(crate) geometry: DsV4Geometry,
    /// Routed-expert FFN width (`gate_exps.out_dim` on the trunk).
    pub(crate) expert_ff: usize,
    /// Shared-expert FFN width; the V4 MTP block always has shared experts.
    pub(crate) shared_ff: usize,
    /// Trunk decoder layer count — the MTP block registers its experts as
    /// this layer index (43 on the real model, matching the GGUF's vestigial
    /// `compress_ratios` slot).
    pub(crate) trunk_layers: usize,
    /// Non-compress rope base (`rope.freq_base`; vLLM selects
    /// `config.rope_theta` for compress_ratio <= 1 layers).
    pub(crate) rope_base: f32,
    /// Trunk swiglu clamp (all layers carry 10.0 on the real model).
    pub(crate) swiglu_clamp: f32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_eps: f32,
}

impl MtpDims {
    /// Derive from a loaded trunk engine. Fails when the trunk lacks the
    /// pieces the MTP block shares (a shared expert, a sliding window).
    pub(crate) fn from_engine<L: DsV4Linear>(engine: &DsV4Engine<L>) -> Result<Self> {
        let layers = engine.layers();
        let first = layers
            .first()
            .ok_or_else(|| anyhow!("trunk engine has no layers"))?;
        let last = layers.last().expect("non-empty checked above");
        if first.gate_exps.out_dim == 0 {
            bail!("trunk expert tensors have zero FFN width");
        }
        let shared = last
            .shared
            .as_ref()
            .ok_or_else(|| anyhow!("MTP module requires the trunk's shared expert shape"))?;
        let geometry = engine.geometry().clone();
        if geometry.window.is_none() {
            bail!("MTP module requires a sliding window (attention.sliding_window)");
        }
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
        })
    }

    fn hc_mix_rows(&self) -> usize {
        let hc = self.geometry.hc;
        hc * hc + 2 * hc
    }

    fn q_dim(&self) -> usize {
        self.geometry.heads * self.geometry.head_dim
    }

    /// Synthesized GGUF-style name for the pooled expert tensors ("blk.43."
    /// on the real model — the layer index the pool keys the slices under).
    fn expert_name(&self, proj: &str) -> String {
        format!("blk.{}.ffn_{proj}_exps.weight", self.trunk_layers)
    }
}

/// A dense weight payload produced by the loader, in the dtype it should be
/// served in: F32 stays exact (the synthetic fixture, and the CPU host
/// reference), F16 carries fp8-block-dequantized shard weights, Bf16 carries
/// the shard's bf16 router verbatim.
pub(crate) enum MtpPayload {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
}

impl MtpPayload {
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
            Self::F16(bits) => bits
                .iter()
                .map(|&bits| crate::safetensors::f16_to_f32(bits))
                .collect(),
            Self::Bf16(bits) => bits
                .iter()
                .map(|&bits| crate::safetensors::bf16_to_f32(bits))
                .collect(),
        }
    }
}

/// One dense matrix destined for provider residency.
pub(crate) struct MtpDenseTensor {
    pub(crate) matrix: RawMatrix,
    /// `Some(rank)` uploads block-diagonally (the grouped output projection).
    /// Consumed by the native-cuda registration; the CPU host reference
    /// serves grouped keys off the same row-major payload directly.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) grouped_rank: Option<usize>,
    pub(crate) payload: MtpPayload,
}

/// One packed rank-3 expert tensor in the GGUF layout (expert-major slices,
/// innermost = the in dim): MXFP4 from the shard's fp4 (bit-exact repack), or
/// raw F32 for the synthetic fixture.
pub(crate) struct MtpExpertTensor {
    pub(crate) experts: RawExperts,
    pub(crate) expert_count: usize,
    pub(crate) dtype: GgufTensorType,
    pub(crate) bytes: Vec<u8>,
}

/// Everything [`load_mtp`] produces: the module's weight handles + small host
/// tensors, and the heavy payloads a provider takes residency of.
pub(crate) struct MtpLoad {
    pub(crate) weights: MtpWeights,
    pub(crate) dense: Vec<MtpDenseTensor>,
    pub(crate) experts: Vec<MtpExpertTensor>,
}

impl MtpLoad {
    /// Resident (dense f16/f32/bf16) payload bytes.
    pub(crate) fn resident_bytes(&self) -> usize {
        self.dense
            .iter()
            .map(|entry| entry.payload.byte_len())
            .sum()
    }

    /// Packed expert payload bytes.
    pub(crate) fn expert_bytes(&self) -> usize {
        self.experts.iter().map(|entry| entry.bytes.len()).sum()
    }
}

/// The MTP module's weight handles: a full [`DsV4Layer`] (the decoder block —
/// same struct as a trunk layer, compressor/indexer `None`) plus the module's
/// input/output shells. Heavy matrices are name-keyed [`RawMatrix`] handles
/// served by whatever provider took residency of the payloads.
pub(crate) struct MtpWeights {
    pub(crate) layer: DsV4Layer,
    pub(crate) e_proj: RawMatrix,
    pub(crate) h_proj: RawMatrix,
    pub(crate) enorm: Vec<f32>,
    pub(crate) hnorm: Vec<f32>,
    pub(crate) hc_head: HcWeights,
    /// The module's own final norm (`mtp.0.norm`, vLLM's shared_head.norm).
    pub(crate) norm: Vec<f32>,
}

/// Shard tensor names (census-verified 2026-07-12; see the plan doc).
mod names {
    pub(super) const WQ_A: &str = "mtp.0.attn.wq_a.weight";
    pub(super) const Q_NORM: &str = "mtp.0.attn.q_norm.weight";
    pub(super) const WQ_B: &str = "mtp.0.attn.wq_b.weight";
    pub(super) const WKV: &str = "mtp.0.attn.wkv.weight";
    pub(super) const KV_NORM: &str = "mtp.0.attn.kv_norm.weight";
    pub(super) const ATTN_SINK: &str = "mtp.0.attn.attn_sink";
    pub(super) const WO_A: &str = "mtp.0.attn.wo_a.weight";
    pub(super) const WO_B: &str = "mtp.0.attn.wo_b.weight";
    pub(super) const ATTN_NORM: &str = "mtp.0.attn_norm.weight";
    pub(super) const FFN_NORM: &str = "mtp.0.ffn_norm.weight";
    pub(super) const E_PROJ: &str = "mtp.0.e_proj.weight";
    pub(super) const H_PROJ: &str = "mtp.0.h_proj.weight";
    pub(super) const ENORM: &str = "mtp.0.enorm.weight";
    pub(super) const HNORM: &str = "mtp.0.hnorm.weight";
    pub(super) const GATE: &str = "mtp.0.ffn.gate.weight";
    pub(super) const GATE_BIAS: &str = "mtp.0.ffn.gate.bias";
    pub(super) const SHARED_W1: &str = "mtp.0.ffn.shared_experts.w1.weight";
    pub(super) const SHARED_W2: &str = "mtp.0.ffn.shared_experts.w2.weight";
    pub(super) const SHARED_W3: &str = "mtp.0.ffn.shared_experts.w3.weight";
    pub(super) const NORM: &str = "mtp.0.norm.weight";

    pub(super) fn hc(prefix: &str, part: &str) -> String {
        format!("mtp.0.hc_{prefix}_{part}")
    }

    pub(super) fn expert(index: usize, proj: &str) -> String {
        format!("mtp.0.ffn.experts.{index}.{proj}.weight")
    }
}

/// Load and shape-validate the `mtp.0.*` shard against the trunk-derived
/// dims. Dense fp8 dequantizes to f16 (scale siblings auto-resolved,
/// MULTIPLIER semantics), bf16/f32 pass through, fp4 experts repack
/// bit-exactly into the GGUF MXFP4 stream the expert pool consumes.
pub(crate) fn load_mtp(file: &SafetensorsFile, dims: &MtpDims) -> Result<MtpLoad> {
    let g = &dims.geometry;
    let mut dense = Vec::new();
    let mut load_dense = |name: &str, rows: usize, cols: usize, grouped_rank: Option<usize>| {
        let payload = dense_payload(file, name, rows, cols)?;
        let matrix = RawMatrix {
            name: name.to_string(),
            rows,
            cols,
        };
        dense.push(MtpDenseTensor {
            matrix: matrix.clone(),
            grouped_rank,
            payload,
        });
        Ok::<RawMatrix, anyhow::Error>(matrix)
    };

    let q_a = load_dense(names::WQ_A, g.q_lora, g.embed, None)?;
    let q_b = load_dense(names::WQ_B, dims.q_dim(), g.q_lora, None)?;
    let kv = load_dense(names::WKV, g.head_dim, g.embed, None)?;
    let out_a = load_dense(
        names::WO_A,
        g.o_groups * g.o_rank,
        dims.q_dim() / g.o_groups,
        Some(g.o_rank),
    )?;
    let out_b = load_dense(names::WO_B, g.embed, g.o_groups * g.o_rank, None)?;
    let e_proj = load_dense(names::E_PROJ, g.embed, g.embed, None)?;
    let h_proj = load_dense(names::H_PROJ, g.embed, g.embed, None)?;
    let router = load_dense(names::GATE, g.experts, g.embed, None)?;
    let shared_gate = load_dense(names::SHARED_W1, dims.shared_ff, g.embed, None)?;
    let shared_down = load_dense(names::SHARED_W2, g.embed, dims.shared_ff, None)?;
    let shared_up = load_dense(names::SHARED_W3, dims.shared_ff, g.embed, None)?;

    let layer = DsV4Layer {
        attn_norm: small_vector(file, names::ATTN_NORM, g.embed)?,
        ffn_norm: small_vector(file, names::FFN_NORM, g.embed)?,
        hc_attn: load_hc(file, dims, "attn", dims.hc_mix_rows(), 3)?,
        hc_ffn: load_hc(file, dims, "ffn", dims.hc_mix_rows(), 3)?,
        q_a,
        q_a_norm: small_vector(file, names::Q_NORM, g.q_lora)?,
        q_b,
        kv,
        kv_norm: small_vector(file, names::KV_NORM, g.head_dim)?,
        sinks: Some(small_vector(file, names::ATTN_SINK, g.heads)?),
        out_a,
        out_b,
        rope_base: dims.rope_base,
        compressor: None,
        indexer: None,
        router,
        probs_bias: Some(small_vector(file, names::GATE_BIAS, g.experts)?),
        tid2eid: None,
        gate_exps: RawExperts {
            name: dims.expert_name("gate"),
            in_dim: g.embed,
            out_dim: dims.expert_ff,
        },
        up_exps: RawExperts {
            name: dims.expert_name("up"),
            in_dim: g.embed,
            out_dim: dims.expert_ff,
        },
        down_exps: RawExperts {
            name: dims.expert_name("down"),
            in_dim: dims.expert_ff,
            out_dim: g.embed,
        },
        shared: Some(SharedExpertWeights {
            gate: shared_gate,
            up: shared_up,
            down: shared_down,
        }),
        swiglu_clamp: dims.swiglu_clamp,
    };

    let experts = vec![
        load_expert_tensor(file, dims, "w1", "gate", dims.expert_ff, g.embed)?,
        load_expert_tensor(file, dims, "w3", "up", dims.expert_ff, g.embed)?,
        load_expert_tensor(file, dims, "w2", "down", g.embed, dims.expert_ff)?,
    ];

    let weights = MtpWeights {
        layer,
        e_proj,
        h_proj,
        enorm: small_vector(file, names::ENORM, g.embed)?,
        hnorm: small_vector(file, names::HNORM, g.embed)?,
        hc_head: load_hc(file, dims, "head", g.hc, 1)?,
        norm: small_vector(file, names::NORM, g.embed)?,
    };
    Ok(MtpLoad {
        weights,
        dense,
        experts,
    })
}

/// Read a dense 2-D weight in serving dtype, validating the safetensors
/// `[out, in]` shape (row-major — identical memory layout to the GGUF's
/// `[ne0 = in, ne1 = out]`).
fn dense_payload(
    file: &SafetensorsFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<MtpPayload> {
    let info = file
        .info(name)
        .ok_or_else(|| anyhow!("MTP shard is missing tensor {name}"))?;
    if info.shape != [rows, cols] {
        bail!(
            "MTP tensor {name} has shape {:?}; expected [{rows}, {cols}]",
            info.shape
        );
    }
    use crate::safetensors::SafetensorsDtype as Dtype;
    let payload = match info.dtype {
        Dtype::F8E4M3 => MtpPayload::F16(file.fp8_block_scaled_f16(name)?),
        Dtype::F32 => MtpPayload::F32(file.tensor_f32(name)?),
        Dtype::F16 => MtpPayload::F16(file.tensor_f16(name)?),
        Dtype::BF16 => MtpPayload::Bf16(
            file.bytes(name)?
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect(),
        ),
        other => bail!("MTP tensor {name} has unsupported dtype {}", other.name()),
    };
    if payload.len() != rows * cols {
        bail!(
            "MTP tensor {name} dequantized to {} values; expected {}",
            payload.len(),
            rows * cols
        );
    }
    Ok(payload)
}

/// Read a small 1-D tensor (norms, sinks, bias) as exact f32.
fn small_vector(file: &SafetensorsFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let info = file
        .info(name)
        .ok_or_else(|| anyhow!("MTP shard is missing tensor {name}"))?;
    if info.shape != [len] {
        bail!(
            "MTP tensor {name} has shape {:?}; expected [{len}]",
            info.shape
        );
    }
    file.tensor_f32(name)
}

/// Load one hyper-connection mixer triple (`hc_{attn,ffn,head}_*`; F32 in the
/// shard, unsuffixed names).
fn load_hc(
    file: &SafetensorsFile,
    dims: &MtpDims,
    which: &str,
    rows: usize,
    scale_len: usize,
) -> Result<HcWeights> {
    let cols = dims.geometry.hc * dims.geometry.embed;
    let fn_name = names::hc(which, "fn");
    let info = file
        .info(&fn_name)
        .ok_or_else(|| anyhow!("MTP shard is missing tensor {fn_name}"))?;
    if info.shape != [rows, cols] {
        bail!(
            "MTP tensor {fn_name} has shape {:?}; expected [{rows}, {cols}]",
            info.shape
        );
    }
    Ok(HcWeights {
        func: DsV4HcFunc::from_parts(rows, cols, file.tensor_f32(&fn_name)?)?,
        base: small_vector(file, &names::hc(which, "base"), rows)?,
        scale: small_vector(file, &names::hc(which, "scale"), scale_len)?,
    })
}

/// Load the 256 per-expert `w{1,3,2}` weights of one projection into a single
/// packed rank-3 blob in the GGUF layout: fp4 shards repack bit-exactly to
/// MXFP4 (one 17-byte block per 32 in-dim values, expert-major), the f32
/// fixture concatenates raw little-endian rows.
fn load_expert_tensor(
    file: &SafetensorsFile,
    dims: &MtpDims,
    shard_proj: &str,
    gguf_proj: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<MtpExpertTensor> {
    use crate::safetensors::SafetensorsDtype as Dtype;
    let expert_count = dims.geometry.experts;
    let mut bytes = Vec::new();
    let mut dtype = None;
    for index in 0..expert_count {
        let name = names::expert(index, shard_proj);
        let info = file
            .info(&name)
            .ok_or_else(|| anyhow!("MTP shard is missing tensor {name}"))?;
        match info.dtype {
            Dtype::I8 => {
                // Packed fp4 [out, in/2] + ue8m0 scale sibling.
                if info.shape != [out_dim, in_dim / 2] {
                    bail!(
                        "MTP tensor {name} has shape {:?}; expected [{out_dim}, {}] (packed fp4)",
                        info.shape,
                        in_dim / 2
                    );
                }
                if dtype.replace(GgufTensorType::MXFP4) == Some(GgufTensorType::F32) {
                    bail!("MTP expert tensors mix fp4 and f32 payloads");
                }
                bytes.extend_from_slice(&file.fp4_to_gguf_mxfp4(&name)?);
            }
            Dtype::F32 => {
                if info.shape != [out_dim, in_dim] {
                    bail!(
                        "MTP tensor {name} has shape {:?}; expected [{out_dim}, {in_dim}]",
                        info.shape
                    );
                }
                if dtype.replace(GgufTensorType::F32) == Some(GgufTensorType::MXFP4) {
                    bail!("MTP expert tensors mix fp4 and f32 payloads");
                }
                bytes.extend_from_slice(file.bytes(&name)?);
            }
            other => bail!("MTP tensor {name} has unsupported dtype {}", other.name()),
        }
    }
    Ok(MtpExpertTensor {
        experts: RawExperts {
            name: dims.expert_name(gguf_proj),
            in_dim,
            out_dim,
        },
        expert_count,
        dtype: dtype.ok_or_else(|| anyhow!("MTP shard has zero experts"))?,
        bytes,
    })
}

/// The MTP draft module: weights + the trunk scalars its forward needs. The
/// heavy linears go through whatever [`DsV4Linear`] provider registered the
/// payloads ([`MtpHostLinear`] for the CPU reference, the trunk's
/// [`DsV4GpuLinear`] handle in production), passed per call so the module
/// itself stays provider-agnostic.
pub(crate) struct MtpModule {
    dims: MtpDims,
    weights: MtpWeights,
    /// Target-shared embedding + lm head (the shard has neither).
    gguf: Arc<GgufFile>,
    token_embd: RawMatrix,
    output_head: RawMatrix,
}

impl MtpModule {
    pub(crate) fn new(
        dims: MtpDims,
        weights: MtpWeights,
        gguf: Arc<GgufFile>,
        token_embd: RawMatrix,
        output_head: RawMatrix,
    ) -> Self {
        Self {
            dims,
            weights,
            gguf,
            token_embd,
            output_head,
        }
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

    fn moe_ctx(&self) -> DsV4MoeBlockCtx<'_> {
        let g = &self.dims.geometry;
        let layer = &self.weights.layer;
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

    /// One draft slot at rope position `pos`: build the input streams from
    /// (token embedding, previous flat residual), run the decoder block over
    /// the module's own SWA ring, and return the flat post-block streams
    /// (the module's pre-hc-head residual, the K>1 recurrence input) plus —
    /// when asked — the vocab logits through hc_head + norm + the target's
    /// lm head.
    pub(crate) fn forward_slot<L: DsV4Linear>(
        &self,
        linear: &L,
        ring: &mut DsV4LayerState,
        pos: usize,
        token: u32,
        prev_flat: &[f32],
        want_logits: bool,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let g = &self.dims.geometry;
        let rms_eps = self.dims.rms_eps;
        if prev_flat.len() != g.hc * g.embed {
            bail!(
                "MTP previous hidden has {} values; expected {}",
                prev_flat.len(),
                g.hc * g.embed
            );
        }

        // vLLM `fused_mtp_input_rmsnorm`: the slot at position 0 zero-masks
        // its embedding (enorm of zeros is zeros), so e_proj contributes
        // nothing there; hnorm applies per stream.
        let mut emb = if pos == 0 {
            vec![0.0f32; g.embed]
        } else {
            dsv4_embed_row(&self.gguf, &self.token_embd, g.vocab, token)?
        };
        rms_norm_in_place(&mut emb, &self.weights.enorm, rms_eps)?;
        let e = linear.mul_vec(TensorKey::Dense(&self.weights.e_proj), &emb)?;
        let mut streams = Vec::with_capacity(g.hc);
        for stream in prev_flat.chunks(g.embed) {
            let mut h = stream.to_vec();
            rms_norm_in_place(&mut h, &self.weights.hnorm, rms_eps)?;
            let mut h = linear.mul_vec(TensorKey::Dense(&self.weights.h_proj), &h)?;
            for (h, e) in h.iter_mut().zip(&e) {
                *h += *e;
            }
            streams.push(h);
        }

        // The decoder block — the exact per-layer body of the trunk's host
        // step (vLLM's DeepseekV4DecoderLayer + the deferred mhc_post).
        let layer = &self.weights.layer;
        let residual = streams.clone();
        let (mut y, post, comb) = hc_pre_math(&layer.hc_attn, &streams, self.hc_params())?;
        rms_norm_in_place(&mut y, &layer.attn_norm, rms_eps)?;
        let attn = dsv4_attention_step(linear, g, rms_eps, layer, ring, &y, pos, 0)?;
        streams = hc_post(&attn, &residual, &post, &comb);

        let residual = streams.clone();
        let (mut y, post, comb) = hc_pre_math(&layer.hc_ffn, &streams, self.hc_params())?;
        rms_norm_in_place(&mut y, &layer.ffn_norm, rms_eps)?;
        let ys = [y];
        let ffn = linear
            .moe_block(&self.moe_ctx(), &ys, &[token])?
            .pop()
            .ok_or_else(|| anyhow!("MTP moe_block returned no output rows"))?;
        streams = hc_post(&ffn, &residual, &post, &comb);

        let mut flat = Vec::with_capacity(g.hc * g.embed);
        for stream in &streams {
            flat.extend_from_slice(stream);
        }
        let logits = if want_logits {
            let mut hidden = hyper_head_math(
                &self.weights.hc_head,
                &streams,
                g.embed,
                rms_eps,
                self.dims.hc_eps,
            )?;
            rms_norm_in_place(&mut hidden, &self.weights.norm, rms_eps)?;
            Some(linear.mul_vec(TensorKey::Dense(&self.output_head), &hidden)?)
        } else {
            None
        };
        Ok((flat, logits))
    }

    /// The drafter core: catch up on every (token, hidden) pair the target
    /// has produced since the last call, then draft up to `k` tokens. See the
    /// module docs for the pairing; `tokens` and `taps` are the Stage-A
    /// [`DraftContext`] fields (tokens = prompt + emitted + pending, taps
    /// covering ABSOLUTE positions `taps.base()..tokens.len()-1` — a request
    /// resumed from a prefix-cache restore has no rows below the base, and
    /// the catch-up cold-starts its ring at the first available pair; drafts
    /// there are an approximation of the warm state — acceptance may dip for
    /// the first ~window slots, never correctness (verify is lossless) — and
    /// the ring converges to the full-history contents after `window` pairs,
    /// because each ring entry is a pure per-slot projection).
    pub(crate) fn propose_tokens<L: DsV4Linear>(
        &self,
        linear: &L,
        state: &mut MtpDrafterState,
        tokens: &[u32],
        taps: &DsV4Taps,
        k: usize,
    ) -> Result<Vec<u32>> {
        let n = tokens.len();
        if n < 2 {
            return Ok(Vec::new());
        }
        // Pair i = (embed(tokens[i+1]), taps.pre_hc_head(i), rope pos i) for
        // i in base..n-1; the taps buffer must cover through the last pair.
        let pairs = n - 1;
        if taps.positions() < pairs {
            bail!(
                "MTP drafter needs taps through position {pairs} but they end at {}",
                taps.positions()
            );
        }
        let base = taps.base();
        if base >= pairs {
            // The restore covered everything up to the pending token; nothing
            // is pairable yet (first proposal right after a deep restore).
            return Ok(Vec::new());
        }

        // Warm continuation: the pairs consumed so far (absolute
        // start..start+fed) must embed the matching token range, not overshoot
        // the pairable range, and leave no tap-less gap before the next pair.
        // Chat-style follow-ups extend the previous transcript, so the ring
        // warm-start survives across turns — even when the new request's taps
        // base moved (the ring was built while its own taps existed); anything
        // else resets and cold-starts at the current base. The hiddens need no
        // check — equal token prefixes produce equal residuals in this
        // deterministic engine.
        let end = state.start + state.fed.len();
        let warm = end <= pairs && end >= base && state.fed[..] == tokens[1 + state.start..1 + end];
        if !warm {
            state.reset();
            state.start = base;
        }

        let mut last: Option<(Vec<f32>, Vec<f32>)> = None;
        while state.start + state.fed.len() < pairs {
            let i = state.start + state.fed.len();
            let token = tokens[i + 1];
            let prev = taps
                .pre_hc_head(i)
                .ok_or_else(|| anyhow!("taps buffer has no pre-hc-head row for position {i}"))?;
            let want = i + 1 == pairs && k > 0;
            let (flat, logits) =
                self.forward_slot(linear, &mut state.ring, i, token, prev, want)?;
            state.fed.push(token);
            if let Some(logits) = logits {
                last = Some((flat, logits));
            }
        }
        let Some((mut flat, logits)) = last else {
            // k == 0 (budget-capped verify) — or an already-consumed context,
            // which the strictly-growing verify loop never produces.
            return Ok(Vec::new());
        };

        let mut drafts = Vec::with_capacity(k);
        drafts.push(argmax(&logits)?);
        if drafts.len() < k {
            // Speculative slots re-feed (draft token, own flat residual) at
            // the next rope positions — vLLM's K>1 recurrence. They must not
            // pollute the ring the next catch-up builds on: restore it after.
            let ring_snapshot = state.ring.clone();
            for t in 1..k {
                let token = *drafts.last().expect("drafts is non-empty");
                let (next_flat, logits) =
                    self.forward_slot(linear, &mut state.ring, pairs - 1 + t, token, &flat, true)?;
                flat = next_flat;
                drafts.push(argmax(
                    &logits.expect("want_logits was set for the recurrence slot"),
                )?);
            }
            state.ring = ring_snapshot;
        }
        Ok(drafts)
    }
}

/// The drafter's own mutable state: the draft layer's SWA ring plus the
/// embedding-side token of every consumed pair
/// (`tokens[1 + start ..= start + len]`), which both indexes the next pair to
/// feed and detects request switches. `start` is the absolute index of the
/// first consumed pair — 0 for full-history requests, the taps base after a
/// prefix-cache restore cold-started the ring mid-sequence.
pub(crate) struct MtpDrafterState {
    ring: DsV4LayerState,
    fed: Vec<u32>,
    start: usize,
}

impl MtpDrafterState {
    pub(crate) fn new() -> Self {
        Self {
            ring: DsV4LayerState {
                ring: std::collections::VecDeque::new(),
                compressor: None,
                indexer: None,
            },
            fed: Vec::new(),
            start: 0,
        }
    }

    fn reset(&mut self) {
        self.ring.ring.clear();
        self.fed.clear();
        self.start = 0;
    }

    /// Pairs consumed so far (tests).
    #[cfg(test)]
    pub(crate) fn consumed(&self) -> usize {
        self.fed.len()
    }

    /// Absolute index of the first consumed pair (tests).
    #[cfg(test)]
    pub(crate) fn start(&self) -> usize {
        self.start
    }
}

/// The CPU host reference [`DsV4Linear`] for the MTP module: the shard's
/// dense payloads and packed expert blobs served with exact f32 host math, and
/// everything else (the target-shared lm head) falling through to the plain
/// GGUF-streaming CPU provider. The CPU==GPU parity gate drives the SAME
/// [`MtpModule`] through this and the CUDA provider. Test-only today: the
/// fixture invariants, the pairing gate, and the parity gate all run on it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct MtpHostLinear {
    fallback: crate::dsv4_cpu::DsV4CpuLinear,
    mats: HashMap<String, MtpHostMat>,
    experts: HashMap<String, MtpHostExperts>,
}

struct MtpHostMat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

struct MtpHostExperts {
    dtype: GgufTensorType,
    bytes: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl MtpHostLinear {
    pub(crate) fn new(gguf: Arc<GgufFile>, load: &MtpLoad) -> Self {
        let mats = load
            .dense
            .iter()
            .map(|entry| {
                (
                    entry.matrix.name.clone(),
                    MtpHostMat {
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
                    MtpHostExperts {
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

impl DsV4Linear for MtpHostLinear {
    fn mul_vec(&self, key: TensorKey<'_>, x: &[f32]) -> Result<Vec<f32>> {
        match key {
            TensorKey::Dense(matrix) => {
                let Some(mat) = self.mats.get(&matrix.name) else {
                    return self.fallback.mul_vec(key, x);
                };
                if x.len() != mat.cols || matrix.rows != mat.rows {
                    bail!(
                        "matvec shapes do not match host MTP tensor {} [{}, {}]",
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

/// `HI_DSV4_MTP_PATH`, defaulting to the documented local shard location.
#[cfg(feature = "native-cuda")]
fn mtp_shard_path() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("HI_DSV4_MTP_PATH") {
        return Some(std::path::PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")?;
    Some(
        std::path::PathBuf::from(home)
            .join(".hi/models/deepseek-v4-flash/mtp/model-00046-of-00046.safetensors"),
    )
}

/// Drafts per verify step: an explicit `HI_DSV4_SPEC_K` wins (the loop caps
/// proposals at its own k anyway); unset defaults to 1 —
/// `num_nextn_predict_layers = 1`, so deeper drafts reuse the module beyond
/// its training depth and decay in quality.
#[cfg(feature = "native-cuda")]
fn mtp_k_cap() -> usize {
    match std::env::var("HI_DSV4_SPEC_K") {
        Ok(raw) => raw.trim().parse::<usize>().unwrap_or(1),
        Err(_) => 1,
    }
}

/// The `HI_DSV4_SPEC=mtp` drafter: the module + the trunk provider handle +
/// its own catch-up state.
#[cfg(feature = "native-cuda")]
pub(crate) struct MtpDrafter {
    module: MtpModule,
    linear: DsV4GpuLinear,
    state: MtpDrafterState,
    k_cap: usize,
    warned: bool,
}

#[cfg(feature = "native-cuda")]
impl MtpDrafter {
    /// Build the drafter over an already-loaded trunk engine: derive dims,
    /// load + validate the shard, take GPU residency of the dense payloads,
    /// and register the packed experts in the trunk's pool as the MTP layer
    /// (pinned). Construction runs on the engine worker thread.
    pub(crate) fn from_shard(
        engine: &DeepSeekV4GpuEngine,
        path: &std::path::Path,
        k_cap: usize,
    ) -> Result<Self> {
        let inner = engine.engine();
        let dims = MtpDims::from_engine(inner)?;
        let file = SafetensorsFile::open(path)?;
        let load = load_mtp(&file, &dims)
            .with_context(|| format!("loading MTP shard {}", path.display()))?;
        let linear = inner.linear().clone();

        for entry in &load.dense {
            let payload = match &entry.payload {
                MtpPayload::F32(values) => HostDenseData::F32(values.clone()),
                MtpPayload::F16(bits) => HostDenseData::F16(bits.clone()),
                MtpPayload::Bf16(bits) => HostDenseData::Bf16(bits.clone()),
            };
            match entry.grouped_rank {
                Some(rank) => linear.register_host_grouped(&entry.matrix, rank, &payload)?,
                None => linear.register_host_dense(&entry.matrix, &payload)?,
            }
        }
        let mut pinned = 0usize;
        let mut pooled = 0usize;
        for (proj, entry) in load.experts.iter().enumerate() {
            let pin = linear.register_host_experts(
                &entry.experts,
                entry.expert_count,
                u32::try_from(dims.trunk_layers).context("layer index does not fit u32")?,
                proj as u8,
                entry.dtype,
                entry.bytes.clone(),
                true,
            )?;
            pooled += entry.expert_count;
            if pin {
                pinned += entry.expert_count;
            }
        }
        eprintln!(
            "dsv4 mtp drafter: loaded {} ({:.0} MiB resident dense, {:.2} GiB experts; {pinned}/{pooled} expert slices pool-pinned as layer {}; k_cap {k_cap})",
            path.display(),
            load.resident_bytes() as f64 / (1u64 << 20) as f64,
            load.expert_bytes() as f64 / (1u64 << 30) as f64,
            dims.trunk_layers,
        );

        let module = MtpModule::new(
            dims,
            load.weights,
            inner.gguf().clone(),
            inner.token_embd_matrix().clone(),
            inner.output_head_matrix().clone(),
        );
        Ok(Self {
            module,
            linear,
            state: MtpDrafterState::new(),
            k_cap,
            warned: false,
        })
    }
}

#[cfg(feature = "native-cuda")]
impl Drafter for MtpDrafter {
    fn tap_config(&self) -> crate::dsv4_cpu::DsV4TapConfig {
        crate::dsv4_cpu::DsV4TapConfig {
            pre_hc_head: true,
            aux_layers: Vec::new(),
        }
    }

    fn propose(&mut self, ctx: &DraftContext<'_>) -> Vec<u32> {
        let Some(taps) = ctx.taps else {
            return Vec::new();
        };
        let k = ctx.k.min(self.k_cap);
        match self
            .module
            .propose_tokens(&self.linear, &mut self.state, ctx.tokens, taps, k)
        {
            Ok(drafts) => drafts,
            Err(err) => {
                if !self.warned {
                    eprintln!("dsv4 mtp drafter error: {err:#}; proposing nothing");
                    self.warned = true;
                }
                self.state.reset();
                Vec::new()
            }
        }
    }
}

/// `HI_DSV4_SPEC=mtp` entry point, constructed on the engine worker thread
/// (device resources are allowed here). Returning `None` leaves speculative
/// decoding off; every failure path explains itself on stderr.
#[cfg(feature = "native-cuda")]
pub(crate) fn mtp_drafter_from_env(engine: &DeepSeekV4GpuEngine) -> Option<Box<dyn Drafter>> {
    let Some(path) = mtp_shard_path() else {
        eprintln!("HI_DSV4_SPEC=mtp: HOME is unset and HI_DSV4_MTP_PATH not given; spec off");
        return None;
    };
    if !path.exists() {
        eprintln!(
            "HI_DSV4_SPEC=mtp: MTP shard not found at {} (set HI_DSV4_MTP_PATH); spec off",
            path.display()
        );
        return None;
    }
    match MtpDrafter::from_shard(engine, &path, mtp_k_cap()) {
        Ok(drafter) => Some(Box::new(drafter)),
        Err(err) => {
            eprintln!("HI_DSV4_SPEC=mtp: failed to build the MTP drafter: {err:#}; spec off");
            None
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::dsv4_cpu::fixture::{tempfile_path, write_deepseek4_spec_gguf};
    use crate::dsv4_cpu::{DsV4CpuLinear, DsV4TapConfig};

    // ---- Fixture plumbing --------------------------------------------------

    pub(super) fn mtp_tempfile(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-cuda-dsv4-mtp-{name}-{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
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

    /// How the pairing fixtures craft a projection.
    #[derive(Clone, Copy)]
    pub(super) enum Proj {
        Random(u32),
        ScaledIdentity(f32),
        Zero,
    }

    impl Proj {
        fn values(self, rows: usize, cols: usize) -> Vec<f32> {
            match self {
                Proj::Random(seed) => vals(seed, rows * cols),
                Proj::ScaledIdentity(scale) => {
                    let mut out = vec![0.0f32; rows * cols];
                    for i in 0..rows.min(cols) {
                        out[i * cols + i] = scale;
                    }
                    out
                }
                Proj::Zero => vec![0.0f32; rows * cols],
            }
        }
    }

    /// Crafting knobs for the MTP safetensors fixture (dims mirror
    /// `write_deepseek4_spec_gguf`: embed 4, hc 2, heads 2, head_dim 8,
    /// rope 4, q_lora 4, groups 2x4, experts 4 top-2 ff 4 + shared 4,
    /// window 4, vocab 4).
    pub(super) struct Craft {
        pub(super) e_proj: Proj,
        pub(super) h_proj: Proj,
        /// wo_b = 0: attention contributes EXACTLY nothing to the streams.
        pub(super) zero_attn_out: bool,
        /// down experts + shared w2 = 0: the MoE contributes EXACTLY nothing.
        pub(super) zero_moe_down: bool,
    }

    impl Craft {
        pub(super) fn random() -> Self {
            Self {
                e_proj: Proj::Random(207),
                h_proj: Proj::Random(208),
                zero_attn_out: false,
                zero_moe_down: false,
            }
        }
    }

    /// Write a syntactically valid safetensors file (the writer pattern from
    /// `safetensors.rs` tests).
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

    /// Tiny `mtp.0.*` safetensors matching the spec-GGUF fixture dims.
    pub(super) fn write_mtp_fixture(path: &Path, craft: &Craft) {
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

        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            (names::WQ_A.into(), vec![q_lora, embed], vals(210, 16)),
            (names::Q_NORM.into(), vec![q_lora], vec![1.0; 4]),
            (names::WQ_B.into(), vec![q_dim, q_lora], vals(211, 64)),
            (names::WKV.into(), vec![head_dim, embed], vals(212, 32)),
            (names::KV_NORM.into(), vec![head_dim], vec![1.0; 8]),
            (names::ATTN_SINK.into(), vec![2], vals(213, 2)),
            (
                names::WO_A.into(),
                vec![groups * rank, q_dim / groups],
                vals(214, 64),
            ),
            (
                names::WO_B.into(),
                vec![embed, groups * rank],
                if craft.zero_attn_out {
                    vec![0.0; 32]
                } else {
                    vals(215, 32)
                },
            ),
            (names::ATTN_NORM.into(), vec![embed], vec![1.0; 4]),
            (names::FFN_NORM.into(), vec![embed], vec![1.0; 4]),
            (
                names::E_PROJ.into(),
                vec![embed, embed],
                craft.e_proj.values(embed, embed),
            ),
            (
                names::H_PROJ.into(),
                vec![embed, embed],
                craft.h_proj.values(embed, embed),
            ),
            (names::ENORM.into(), vec![embed], vec![1.0; 4]),
            (names::HNORM.into(), vec![embed], vec![1.0; 4]),
            (
                names::hc("attn", "fn"),
                vec![mix, hc * embed],
                vals(216, mix * hc * embed),
            ),
            (names::hc("attn", "base"), vec![mix], vals(217, mix)),
            (names::hc("attn", "scale"), vec![3], vec![0.6, 0.4, 0.8]),
            (
                names::hc("ffn", "fn"),
                vec![mix, hc * embed],
                vals(218, mix * hc * embed),
            ),
            (names::hc("ffn", "base"), vec![mix], vals(219, mix)),
            (names::hc("ffn", "scale"), vec![3], vec![0.5, 0.7, 0.3]),
            (
                names::hc("head", "fn"),
                vec![hc, hc * embed],
                vals(220, hc * hc * embed),
            ),
            (names::hc("head", "base"), vec![hc], vals(221, hc)),
            (names::hc("head", "scale"), vec![1], vec![0.7]),
            (names::GATE.into(), vec![experts, embed], vals(222, 16)),
            (names::GATE_BIAS.into(), vec![experts], vals(223, 4)),
            (
                names::SHARED_W1.into(),
                vec![shared_ff, embed],
                vals(224, 16),
            ),
            (
                names::SHARED_W2.into(),
                vec![embed, shared_ff],
                if craft.zero_moe_down {
                    vec![0.0; 16]
                } else {
                    vals(225, 16)
                },
            ),
            (
                names::SHARED_W3.into(),
                vec![shared_ff, embed],
                vals(226, 16),
            ),
            (names::NORM.into(), vec![embed], vec![1.0; 4]),
        ];
        for index in 0..experts {
            let seed = 230 + index as u32 * 3;
            tensors.push((
                names::expert(index, "w1"),
                vec![ff, embed],
                vals(seed, ff * embed),
            ));
            tensors.push((
                names::expert(index, "w3"),
                vec![ff, embed],
                vals(seed + 1, ff * embed),
            ));
            tensors.push((
                names::expert(index, "w2"),
                vec![embed, ff],
                if craft.zero_moe_down {
                    vec![0.0; embed * ff]
                } else {
                    vals(seed + 2, embed * ff)
                },
            ));
        }
        write_safetensors(path, &tensors);
    }

    /// A CPU trunk engine + MTP module + host provider over fresh fixtures.
    pub(super) struct CpuRig {
        pub(super) engine: DsV4Engine<DsV4CpuLinear>,
        pub(super) module: MtpModule,
        pub(super) host: MtpHostLinear,
    }

    pub(super) fn cpu_rig(name: &str, craft: &Craft) -> CpuRig {
        let gguf_path = tempfile_path(&format!("mtp-{name}"));
        write_deepseek4_spec_gguf(&gguf_path);
        let mtp_path = mtp_tempfile(name);
        write_mtp_fixture(&mtp_path, craft);
        let gguf = Arc::new(hi_gguf::GgufFile::open(&gguf_path).unwrap());
        let engine = DsV4Engine::new(
            gguf.clone(),
            DsV4CpuLinear::from_gguf(gguf.clone()),
            "cpu-reference",
        )
        .unwrap();
        let dims = MtpDims::from_engine(&engine).unwrap();
        let file = SafetensorsFile::open(&mtp_path).unwrap();
        let load = load_mtp(&file, &dims).unwrap();
        let host = MtpHostLinear::new(gguf.clone(), &load);
        let module = MtpModule::new(
            dims,
            load.weights,
            gguf,
            engine.token_embd_matrix().clone(),
            engine.output_head_matrix().clone(),
        );
        CpuRig {
            engine,
            module,
            host,
        }
    }

    /// Prefill `tokens` through the trunk with pre-hc-head capture.
    pub(super) fn taps_for(engine: &DsV4Engine<DsV4CpuLinear>, tokens: &[u32]) -> DsV4Taps {
        taps_for_at(engine, tokens, 0)
    }

    /// [`taps_for`] with the first `base` positions forwarded UNTAPPED — a
    /// prefix-cache restore's view of the same sequence (rows exist for
    /// absolute positions `base..tokens.len()` only, bit-identical to the
    /// full buffer's rows there).
    pub(super) fn taps_for_at(
        engine: &DsV4Engine<DsV4CpuLinear>,
        tokens: &[u32],
        base: usize,
    ) -> DsV4Taps {
        let mut taps = engine
            .new_taps_at(
                DsV4TapConfig {
                    pre_hc_head: true,
                    aux_layers: Vec::new(),
                },
                base,
            )
            .unwrap();
        let mut state = engine.new_state();
        if base > 0 {
            engine.prefill(&mut state, &tokens[..base]).unwrap();
        }
        engine
            .prefill_with_taps(&mut state, &tokens[base..], Some(&mut taps))
            .unwrap();
        taps
    }

    // ---- Loader + CPU forward invariants ----------------------------------

    #[test]
    fn mtp_loader_validates_dims_and_rejects_bad_shapes() {
        let rig = cpu_rig("loader", &Craft::random());
        let dims = &rig.module.dims;
        assert_eq!(dims.trunk_layers, 3);
        assert_eq!(dims.expert_ff, 4);
        assert_eq!(dims.shared_ff, 4);
        assert_eq!(dims.geometry.hc, 2);
        assert_eq!(dims.rope_base, 10_000.0);
        assert_eq!(dims.swiglu_clamp, 10.0);
        assert_eq!(
            rig.module.weights.layer.gate_exps.name,
            "blk.3.ffn_gate_exps.weight"
        );
        assert!(rig.module.weights.layer.compressor.is_none());
        assert!(rig.module.weights.layer.indexer.is_none());
        assert!(rig.module.weights.layer.tid2eid.is_none());
        assert!(rig.module.weights.layer.probs_bias.is_some());

        // A wrong-shaped tensor must fail loudly, naming the tensor.
        let bad_path = mtp_tempfile("loader-bad");
        write_mtp_fixture(&bad_path, &Craft::random());
        let mut file_bytes = std::fs::read(&bad_path).unwrap();
        // Rewrite the header's wq_a shape [4,4] -> [2,8] (same byte span).
        let header_len = u64::from_le_bytes(file_bytes[0..8].try_into().unwrap()) as usize;
        let header = String::from_utf8(file_bytes[8..8 + header_len].to_vec()).unwrap();
        let bad_header = header.replace(
            "\"mtp.0.attn.wq_a.weight\":{\"data_offsets\"",
            "\"mtp.0.attn.wq_a.weight\":{\"data_offsets\"",
        );
        assert_eq!(header, bad_header, "marker sanity");
        let bad_header = header.replacen("\"shape\":[4,4]", "\"shape\":[2,8]", 1);
        assert_ne!(header, bad_header, "fixture header must contain [4,4]");
        assert_eq!(header.len(), bad_header.len());
        file_bytes[8..8 + header_len].copy_from_slice(bad_header.as_bytes());
        std::fs::write(&bad_path, file_bytes).unwrap();
        let file = SafetensorsFile::open(&bad_path).unwrap();
        let err = match load_mtp(&file, dims) {
            Ok(_) => panic!("loader accepted a mis-shaped wq_a"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("expected [4, 4]") || err.contains("shape"),
            "unhelpful shape error: {err}"
        );
    }

    #[test]
    fn mtp_cpu_forward_invariants_hold() {
        let rig = cpu_rig("invariants", &Craft::random());
        let tokens = [0u32, 1, 2, 0, 1, 2];
        let taps = taps_for(&rig.engine, &tokens);
        let mut state = MtpDrafterState::new();
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &taps, 3)
            .unwrap();
        assert_eq!(drafts.len(), 3);
        assert!(drafts.iter().all(|&token| token < 4), "{drafts:?}");
        assert_eq!(state.consumed(), tokens.len() - 1);
        // The draft ring holds the catch-up pairs only (speculative slots are
        // snapshot-restored), capped by the window (4).
        assert_eq!(state.ring.ring.len(), 4.min(tokens.len() - 1));

        // Deterministic: a fresh state over the same context re-drafts the
        // same tokens.
        let mut fresh = MtpDrafterState::new();
        let again = rig
            .module
            .propose_tokens(&rig.host, &mut fresh, &tokens, &taps, 3)
            .unwrap();
        assert_eq!(drafts, again);

        // Same context twice on one state: nothing new to consume -> no
        // drafts (the verify loop always grows the context between calls).
        let empty = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &taps, 3)
            .unwrap();
        assert!(empty.is_empty());

        // A DIVERGED context (same length, different history) resets and
        // still drafts deterministically.
        let other = [0u32, 2, 1, 0, 2, 1];
        let other_taps = taps_for(&rig.engine, &other);
        let diverged = rig
            .module
            .propose_tokens(&rig.host, &mut state, &other, &other_taps, 2)
            .unwrap();
        let mut fresh = MtpDrafterState::new();
        let expected = rig
            .module
            .propose_tokens(&rig.host, &mut fresh, &other, &other_taps, 2)
            .unwrap();
        assert_eq!(diverged, expected);

        // k = 0 (budget-capped) still catches up but proposes nothing.
        let longer = [0u32, 1, 2, 0, 1, 2, 3];
        let longer_taps = taps_for(&rig.engine, &longer);
        let mut state = MtpDrafterState::new();
        let none = rig
            .module
            .propose_tokens(&rig.host, &mut state, &longer, &longer_taps, 0)
            .unwrap();
        assert!(none.is_empty());
        assert_eq!(state.consumed(), longer.len() - 1);
    }

    /// Restore-based taps (rows only from `base` on): the catch-up
    /// cold-starts its ring at the first available pair. Ring entries are
    /// pure per-slot projections, so once `window` pairs have been fed the
    /// ring equals the full-history ring and the drafts are BIT-IDENTICAL;
    /// shallower bases still draft validly (the documented approximation).
    /// Warm continuation and divergence detection keep working with a
    /// mid-sequence start.
    #[test]
    fn mtp_cold_start_at_taps_base() {
        let rig = cpu_rig("base", &Craft::random());
        let tokens: Vec<u32> = (0..10).map(|idx| idx % 3).collect();
        let pairs = tokens.len() - 1;
        let window = rig.module.dims.geometry.window.unwrap();

        let full_taps = taps_for(&rig.engine, &tokens);
        let mut full_state = MtpDrafterState::new();
        let full = rig
            .module
            .propose_tokens(&rig.host, &mut full_state, &tokens, &full_taps, 2)
            .unwrap();
        assert_eq!(full.len(), 2);

        // Deep-enough base (>= window pairs available): identical drafts.
        let base = pairs - window - 1;
        let based_taps = taps_for_at(&rig.engine, &tokens, base);
        assert_eq!(based_taps.base(), base);
        assert!(based_taps.pre_hc_head(base - 1).is_none());
        let mut state = MtpDrafterState::new();
        let cold = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &based_taps, 2)
            .unwrap();
        assert_eq!(
            cold, full,
            "after window pairs the cold-start ring equals full history"
        );
        assert_eq!(state.start(), base);
        assert_eq!(state.consumed(), pairs - base);

        // Warm continuation across a verify iteration: accepted tokens extend
        // the transcript; the mid-sequence state neither resets nor re-feeds.
        let mut longer = tokens.clone();
        longer.extend([1u32, 2]);
        let longer_taps = taps_for_at(&rig.engine, &longer, base);
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &longer, &longer_taps, 2)
            .unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(
            state.start(),
            base,
            "warm continuation keeps the cold-start point"
        );
        assert_eq!(state.consumed(), longer.len() - 1 - base);

        // Divergence mid-sequence (same length, different tail) must reset —
        // and cold-start at the NEW request's base, not misfire on offsets.
        let mut other = longer.clone();
        let flip = other.len() - 2;
        other[flip] = (other[flip] + 1) % 3;
        let other_base = 6;
        let other_taps = taps_for_at(&rig.engine, &other, other_base);
        let diverged = rig
            .module
            .propose_tokens(&rig.host, &mut state, &other, &other_taps, 2)
            .unwrap();
        let mut fresh = MtpDrafterState::new();
        let expected = rig
            .module
            .propose_tokens(&rig.host, &mut fresh, &other, &other_taps, 2)
            .unwrap();
        assert_eq!(diverged, expected, "diverged state must re-draft fresh");
        assert_eq!(state.start(), other_base);

        // A restore GAP: the new base lies beyond the warm state's consumed
        // pairs, so the tap-less pairs in between force a reset + cold-start
        // at the new base (never a bail, never a misaligned catch-up).
        let mut warm_state = MtpDrafterState::new();
        rig.module
            .propose_tokens(
                &rig.host,
                &mut warm_state,
                &longer,
                &taps_for_at(&rig.engine, &longer, base),
                2,
            )
            .unwrap();
        let mut gapped = longer.clone();
        gapped.extend([0u32, 1]);
        let gap_pairs = gapped.len() - 1;
        let gap_base = gap_pairs - 1;
        assert!(gap_base > longer.len() - 1, "gap must clear the warm end");
        let gap_taps = taps_for_at(&rig.engine, &gapped, gap_base);
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut warm_state, &gapped, &gap_taps, 2)
            .unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(warm_state.start(), gap_base, "gap must cold-start at base");
        assert_eq!(warm_state.consumed(), gap_pairs - gap_base);

        // A base at/after the last pair (deep restore, first proposal):
        // nothing is pairable yet — propose nothing rather than misalign.
        let all_restored = taps_for_at(&rig.engine, &tokens, pairs);
        let mut state = MtpDrafterState::new();
        assert!(
            rig.module
                .propose_tokens(&rig.host, &mut state, &tokens, &all_restored, 2)
                .unwrap()
                .is_empty()
        );

        // Shallow base (< window pairs): valid drafts, approximated ring.
        let shallow_base = pairs - 2;
        let shallow = taps_for_at(&rig.engine, &tokens, shallow_base);
        let mut state = MtpDrafterState::new();
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &shallow, 2)
            .unwrap();
        assert_eq!(drafts.len(), 2);
        assert!(drafts.iter().all(|&token| token < 4), "{drafts:?}");
        assert_eq!(state.start(), shallow_base);
        assert_eq!(state.ring.ring.len(), pairs - shallow_base);
    }

    #[test]
    fn mtp_position_zero_masks_the_embedding() {
        let rig = cpu_rig("pos0", &Craft::random());
        // Two 2-token contexts share tokens[0] (same hidden h0) and differ in
        // tokens[1] — the pair-0 embedding. vLLM zero-masks the embedding at
        // rope position 0, so the drafts MUST be identical.
        let taps_a = taps_for(&rig.engine, &[0u32, 1]);
        let taps_b = taps_for(&rig.engine, &[0u32, 2]);
        let mut state = MtpDrafterState::new();
        let draft_a = rig
            .module
            .propose_tokens(&rig.host, &mut state, &[0, 1], &taps_a, 1)
            .unwrap();
        let mut state = MtpDrafterState::new();
        let draft_b = rig
            .module
            .propose_tokens(&rig.host, &mut state, &[0, 2], &taps_b, 1)
            .unwrap();
        assert_eq!(draft_a, draft_b, "position 0 must ignore the embedding");

        // Control: at position >= 1 the embedding is live — the same
        // divergence one slot later must be able to change the draft. (The
        // random fixture separates the two embeddings' logits by
        // construction; if this ever ties, reseed the fixture.)
        let taps_a = taps_for(&rig.engine, &[0u32, 1, 1]);
        let taps_b = taps_for(&rig.engine, &[0u32, 1, 2]);
        let mut state = MtpDrafterState::new();
        let flat_a = {
            rig.module
                .propose_tokens(&rig.host, &mut state, &[0, 1, 1], &taps_a, 1)
                .unwrap()
        };
        let mut state = MtpDrafterState::new();
        let flat_b = {
            rig.module
                .propose_tokens(&rig.host, &mut state, &[0, 1, 2], &taps_b, 1)
                .unwrap()
        };
        // Hiddens h0, h1 agree (same prefix [0, 1]); only the pair-1
        // embedding differs. We assert on the module's slot output rather
        // than the argmax to keep this robust to argmax ties.
        let prev = taps_a.pre_hc_head(1).unwrap();
        let mut ring = DsV4LayerState {
            ring: std::collections::VecDeque::new(),
            compressor: None,
            indexer: None,
        };
        let (out_a, _) = rig
            .module
            .forward_slot(&rig.host, &mut ring, 1, 1, prev, false)
            .unwrap();
        let mut ring = DsV4LayerState {
            ring: std::collections::VecDeque::new(),
            compressor: None,
            indexer: None,
        };
        let (out_b, _) = rig
            .module
            .forward_slot(&rig.host, &mut ring, 1, 2, prev, false)
            .unwrap();
        assert_ne!(out_a, out_b, "position >= 1 must see the embedding");
        let _ = (flat_a, flat_b);
    }

    // ---- The pairing gate --------------------------------------------------

    /// Straight-line derivation of one slot's (flat, argmax) under the
    /// crafted fixtures (wo_b = 0 and MoE-down = 0 make attention and the
    /// MoE contribute EXACT zeros, so the block reduces to the two hc
    /// comb-mixes of the input streams). Independent of the module's
    /// catch-up/pairing code — it hardcodes WHICH (embedding, hidden) it was
    /// given, which is exactly what the pairing test discriminates.
    fn straight_line(
        module: &MtpModule,
        host: &MtpHostLinear,
        emb_token: Option<u32>,
        prev_flat: &[f32],
    ) -> (Vec<f32>, u32) {
        let g = &module.dims.geometry;
        let rms_eps = module.dims.rms_eps;
        let mut emb = match emb_token {
            Some(token) => {
                dsv4_embed_row(&module.gguf, &module.token_embd, g.vocab, token).unwrap()
            }
            None => vec![0.0; g.embed],
        };
        rms_norm_in_place(&mut emb, &module.weights.enorm, rms_eps).unwrap();
        let e = host
            .mul_vec(TensorKey::Dense(&module.weights.e_proj), &emb)
            .unwrap();
        let mut streams = Vec::new();
        for stream in prev_flat.chunks(g.embed) {
            let mut h = stream.to_vec();
            rms_norm_in_place(&mut h, &module.weights.hnorm, rms_eps).unwrap();
            let mut h = host
                .mul_vec(TensorKey::Dense(&module.weights.h_proj), &h)
                .unwrap();
            for (h, e) in h.iter_mut().zip(&e) {
                *h += *e;
            }
            streams.push(h);
        }
        let params = HcParams {
            hc: g.hc,
            embed: g.embed,
            rms_eps,
            hc_eps: module.dims.hc_eps,
            sinkhorn_iterations: g.sinkhorn_iterations,
        };
        let zero = vec![0.0f32; g.embed];
        let layer = &module.weights.layer;
        let (_, post, comb) = hc_pre_math(&layer.hc_attn, &streams, params).unwrap();
        let streams = hc_post(&zero, &streams, &post, &comb);
        let (_, post, comb) = hc_pre_math(&layer.hc_ffn, &streams, params).unwrap();
        let streams = hc_post(&zero, &streams, &post, &comb);
        let mut hidden = hyper_head_math(
            &module.weights.hc_head,
            &streams,
            g.embed,
            rms_eps,
            module.dims.hc_eps,
        )
        .unwrap();
        rms_norm_in_place(&mut hidden, &module.weights.norm, rms_eps).unwrap();
        let logits = host
            .mul_vec(TensorKey::Dense(&module.output_head), &hidden)
            .unwrap();
        let mut flat = Vec::new();
        for stream in &streams {
            flat.extend_from_slice(stream);
        }
        (flat, argmax(&logits).unwrap())
    }

    /// THE pairing test: with attention and MoE outputs crafted to exact
    /// zero, the draft is a pure function of the (embedding, hidden) pair the
    /// drafter feeds the last slot. The correct pairing is
    /// (embed(tokens[n-1]), taps(n-2)); both off-by-one hypotheses must
    /// produce a DIFFERENT draft, and the drafter must produce the correct
    /// one. K>1 recurrence chains are pinned the same way.
    #[test]
    fn mtp_pairing_alignment_produces_expected_draft() {
        // Craft A: e-dominant (h_proj = 0) — discriminates the TOKEN side.
        let rig = cpu_rig(
            "pairing-token",
            &Craft {
                e_proj: Proj::ScaledIdentity(3.0),
                h_proj: Proj::Zero,
                zero_attn_out: true,
                zero_moe_down: true,
            },
        );
        let tokens = [0u32, 1, 2, 0, 1];
        let n = tokens.len();
        let taps = taps_for(&rig.engine, &tokens);
        let prev = taps.pre_hc_head(n - 2).unwrap();
        let (_, expected) = straight_line(&rig.module, &rig.host, Some(tokens[n - 1]), prev);
        let (_, wrong_token) = straight_line(&rig.module, &rig.host, Some(tokens[n - 2]), prev);
        assert_ne!(
            expected, wrong_token,
            "fixture must discriminate the token-side off-by-one (reseed if this ties)"
        );
        let mut state = MtpDrafterState::new();
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &taps, 3)
            .unwrap();
        assert_eq!(
            drafts[0], expected,
            "slot pairing must be (embed(tokens[n-1]), taps(n-2))"
        );
        // K>1 recurrence: with the hidden path zeroed the chain is a pure
        // token map d_t = f(d_{t-1}).
        let (_, d1) = straight_line(&rig.module, &rig.host, Some(drafts[0]), prev);
        let (_, d2) = straight_line(&rig.module, &rig.host, Some(d1), prev);
        assert_eq!(drafts, vec![expected, d1, d2], "K>1 token recurrence");

        // Craft B: h-dominant (e_proj = 0) — discriminates the HIDDEN side.
        let rig = cpu_rig(
            "pairing-hidden",
            &Craft {
                e_proj: Proj::Zero,
                h_proj: Proj::ScaledIdentity(1.0),
                zero_attn_out: true,
                zero_moe_down: true,
            },
        );
        let taps = taps_for(&rig.engine, &tokens);
        let prev = taps.pre_hc_head(n - 2).unwrap();
        let stale = taps.pre_hc_head(n - 3).unwrap();
        let (flat0, expected) = straight_line(&rig.module, &rig.host, None, prev);
        let (_, wrong_hidden) = straight_line(&rig.module, &rig.host, None, stale);
        assert_ne!(
            expected, wrong_hidden,
            "fixture must discriminate the hidden-side off-by-one (reseed if this ties)"
        );
        let mut state = MtpDrafterState::new();
        let drafts = rig
            .module
            .propose_tokens(&rig.host, &mut state, &tokens, &taps, 2)
            .unwrap();
        assert_eq!(drafts[0], expected, "hidden pairing must be taps(n-2)");
        // K>1 recurrence: the second draft must come from the module's OWN
        // flat output re-fed as the hidden (embedding path zeroed).
        let (_, d1) = straight_line(&rig.module, &rig.host, None, &flat0);
        assert_eq!(drafts, vec![expected, d1], "K>1 hidden recurrence");
    }

    // ---- Real-shard census (ignored: needs the downloaded checkpoint) -----

    pub(super) fn real_shard_path() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home)
            .join(".hi/models/deepseek-v4-flash/mtp/model-00046-of-00046.safetensors");
        path.exists().then_some(path)
    }

    /// Real dims, stated independently of any engine (byte-verified census in
    /// docs/deepseek-v4-spec-decode-plan.md).
    fn real_dims() -> MtpDims {
        MtpDims {
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
                context: 1_048_576,
            },
            expert_ff: 2048,
            shared_ff: 2048,
            trunk_layers: 43,
            rope_base: 10_000.0,
            swiglu_clamp: 10.0,
            rms_eps: 1.0e-6,
            hc_eps: 1.0e-6,
        }
    }

    /// Census + full-load dequant sanity over the official shard. Run:
    /// `cargo test -p hi-cuda --release mtp_real_shard -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the downloaded DeepSeek-V4-Flash MTP shard (~3.6 GB)"]
    fn mtp_real_shard_census_and_dequant_sanity() {
        let Some(path) = real_shard_path() else {
            eprintln!("skipping: MTP shard not found");
            return;
        };
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(file.tensors().len(), 1575, "shard census drifted");
        let info = |name: &str| file.info(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info(names::WQ_B).shape, [32768, 1024]);
        assert_eq!(info(names::WO_A).shape, [8192, 4096]);
        assert_eq!(info(names::ATTN_SINK).shape, [64]);
        assert_eq!(info(&names::hc("head", "fn")).shape, [4, 16384]);
        assert_eq!(info(&names::expert(255, "w2")).shape, [4096, 1024]);

        let dims = real_dims();
        let started = std::time::Instant::now();
        let load = load_mtp(&file, &dims).unwrap();
        eprintln!(
            "real shard loaded in {:.1}s: {:.1} MiB dense resident, {:.2} GiB packed experts",
            started.elapsed().as_secs_f64(),
            load.resident_bytes() as f64 / (1u64 << 20) as f64,
            load.expert_bytes() as f64 / (1u64 << 30) as f64,
        );
        // Census sizes from the plan doc: ~335 MB dense f16, 3 packed expert
        // tensors of 256 x 4,456,448 bytes each.
        let dense_mb = load.resident_bytes() as f64 / (1u64 << 20) as f64;
        assert!(
            (300.0..380.0).contains(&dense_mb),
            "dense resident {dense_mb:.1} MiB left the expected band"
        );
        assert_eq!(load.experts.len(), 3);
        for entry in &load.experts {
            assert_eq!(entry.dtype, GgufTensorType::MXFP4);
            assert_eq!(entry.expert_count, 256);
            assert_eq!(entry.bytes.len(), 256 * 4_456_448);
        }

        // Dequant sanity: norms near 1, sinks finite, fp8 weights finite and
        // non-degenerate, expert nibbles decode to finite values.
        let norm = &load.weights.norm;
        assert!(norm.iter().all(|v| v.is_finite()));
        let mean = norm.iter().sum::<f32>() / norm.len() as f32;
        assert!((0.05..20.0).contains(&mean), "final norm mean {mean}");
        assert!(
            load.weights
                .layer
                .sinks
                .as_ref()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        let bias = load.weights.layer.probs_bias.as_ref().unwrap();
        assert!(bias.iter().all(|v| v.is_finite()));
        let wq_a = load
            .dense
            .iter()
            .find(|entry| entry.matrix.name == names::WQ_A)
            .unwrap();
        let wq_a_f32 = wq_a.payload.to_f32();
        assert!(wq_a_f32.iter().all(|v| v.is_finite()));
        let nonzero = wq_a_f32.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero * 2 > wq_a_f32.len(),
            "wq_a dequant looks degenerate ({nonzero} nonzero of {})",
            wq_a_f32.len()
        );
        // MXFP4 repack cross-check: expert 0's first row decoded through the
        // GGUF dequantizer must match the safetensors-side fp4 dequant.
        let gate = &load.experts[0];
        let per_expert = gate.experts.in_dim * gate.experts.out_dim;
        let row = hi_gguf::dequantize_tensor_as_f32(
            &gate.bytes[..gate.experts.in_dim / 32 * 17],
            GgufTensorType::MXFP4,
            gate.experts.in_dim,
        )
        .unwrap();
        let reference = file.fp4_block_scaled_f32(&names::expert(0, "w1")).unwrap();
        assert_eq!(row, reference[..gate.experts.in_dim], "repack drifted");
        let _ = per_expert;
    }
}

/// Native-CUDA gates: CPU==GPU module parity, the fixture backend running the
/// REAL MtpDrafter end to end (lossless by construction), and the ignored
/// real-model acceptance measurement.
#[cfg(all(test, feature = "native-cuda"))]
mod native_tests {
    use std::sync::Arc;

    use futures_util::StreamExt;
    use hi_local_core::backend::{GenerationEvent, GenerationRequest, InferenceBackend};

    use super::tests::{Craft, cpu_rig, mtp_tempfile, taps_for, write_mtp_fixture};
    use super::*;
    use crate::dsv4_backend::DeepSeekV4Backend;
    use crate::dsv4_cpu::fixture::{tempfile_path, write_deepseek4_spec_gguf};

    /// A GPU engine + MtpDrafter over fixtures byte-identical to `cpu_rig`'s
    /// (the writers are deterministic).
    fn gpu_drafter(name: &str, craft: &Craft, k_cap: usize) -> (DeepSeekV4GpuEngine, MtpDrafter) {
        let gguf_path = tempfile_path(&format!("mtp-{name}"));
        write_deepseek4_spec_gguf(&gguf_path);
        let mtp_path = mtp_tempfile(name);
        write_mtp_fixture(&mtp_path, craft);
        let gpu = DeepSeekV4GpuEngine::load(&gguf_path).unwrap();
        let drafter = MtpDrafter::from_shard(&gpu, &mtp_path, k_cap).unwrap();
        (gpu, drafter)
    }

    /// The module forward must agree between the CPU host reference and the
    /// CUDA provider — same slots, same taps (the CPU engine's, so trunk
    /// GEMV reduction differences cannot leak into the comparison).
    #[test]
    fn mtp_fixture_cpu_gpu_parity() {
        let craft = Craft::random();
        let rig = cpu_rig("gpu-parity", &craft);
        let (_gpu, mut drafter) = gpu_drafter("gpu-parity", &craft, 4);

        let tokens = [0u32, 1, 2, 0, 1, 2, 0, 1];
        let taps = taps_for(&rig.engine, &tokens);

        // Slot-by-slot: identical inputs through both providers.
        let mut cpu_ring = MtpDrafterState::new();
        let mut gpu_ring = MtpDrafterState::new();
        for i in 0..tokens.len() - 1 {
            let prev = taps.pre_hc_head(i).unwrap();
            let (cpu_flat, cpu_logits) = rig
                .module
                .forward_slot(&rig.host, &mut cpu_ring.ring, i, tokens[i + 1], prev, true)
                .unwrap();
            let (gpu_flat, gpu_logits) = drafter
                .module
                .forward_slot(
                    &drafter.linear,
                    &mut gpu_ring.ring,
                    i,
                    tokens[i + 1],
                    prev,
                    true,
                )
                .unwrap();
            assert_close(&cpu_flat, &gpu_flat, &format!("slot {i} flat"));
            assert_close(
                &cpu_logits.unwrap(),
                &gpu_logits.unwrap(),
                &format!("slot {i} logits"),
            );
        }

        // Whole-proposal parity: identical drafts from identical contexts.
        let mut cpu_state = MtpDrafterState::new();
        let cpu_drafts = rig
            .module
            .propose_tokens(&rig.host, &mut cpu_state, &tokens, &taps, 4)
            .unwrap();
        let ctx = DraftContext {
            tokens: &tokens,
            taps: Some(&taps),
            k: 4,
        };
        let gpu_drafts = drafter.propose(&ctx);
        assert_eq!(cpu_drafts, gpu_drafts, "draft tokens diverged");
        assert_eq!(gpu_drafts.len(), 4);
    }

    fn assert_close(cpu: &[f32], gpu: &[f32], what: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{what}: length mismatch");
        for (idx, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (c - g).abs() <= 1.0e-5,
                "{what}[{idx}]: cpu {c} vs gpu {g} exceeds 1e-5"
            );
        }
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

    /// The full serving loop with the REAL MtpDrafter over the fixture: the
    /// emitted stream must be byte-identical to the sequential baseline (the
    /// verify loop guarantees losslessness regardless of draft quality), the
    /// drafter must actually run, and a warm rerun (prefix restore) must
    /// reproduce it again.
    #[tokio::test]
    async fn mtp_fixture_backend_speculative_stream_matches_sequential() {
        let prompt = "abcabcab";
        let max_tokens = 8u32;

        let baseline_path = tempfile_path("mtp-e2e-baseline");
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

        let spec_gguf = tempfile_path("mtp-e2e-spec");
        write_deepseek4_spec_gguf(&spec_gguf);
        let mtp_path = mtp_tempfile("e2e");
        write_mtp_fixture(&mtp_path, &Craft::random());
        let factory_path = mtp_path.clone();
        let backend = DeepSeekV4Backend::load_with_drafter(
            &spec_gguf,
            Some("dsv4-fixture".to_string()),
            4,
            1 << 20,
            Box::new(move |engine| {
                Some(Box::new(
                    MtpDrafter::from_shard(engine, &factory_path, 4)
                        .expect("fixture MTP drafter must build"),
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
            "the MTP drafter never proposed"
        );

        // Warm rerun: same conversation again (prefix snapshots were written
        // at accepted boundaries only). Taps no longer disable the restore —
        // the rerun MUST reuse cached blocks, the drafter cold-starts its
        // catch-up at the taps base, and the stream stays byte-identical.
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

    /// Real-model MTP end-to-end: identical output with the drafter on vs
    /// off, printed acceptance stats. Run explicitly on a quiet-enough GPU:
    /// `HI_DSV4_EXPERT_POOL_GB=16 CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release \
    ///  --features native-cuda mtp_real_model_e2e -- --ignored --nocapture --test-threads=1`
    #[tokio::test]
    #[ignore = "needs the real checkpoint + MTP shard and a mostly-idle GPU"]
    async fn mtp_real_model_e2e_lossless_and_acceptance() {
        let Some(gguf_path) = crate::dsv4_gpu::tests::real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let Some(shard_path) = super::tests::real_shard_path() else {
            eprintln!("skipping: MTP shard not found");
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

        // K comes from HI_DSV4_SPEC_K exactly like production (default 1).
        let k_cap = mtp_k_cap();
        eprintln!("loading speculative backend (MTP drafter, K={k_cap})...");
        let started = std::time::Instant::now();
        let factory_path = shard_path.clone();
        let backend = DeepSeekV4Backend::load_with_drafter(
            &gguf_path,
            Some("dsv4-real".to_string()),
            256,
            1 << 30,
            Box::new(move |engine| {
                Some(Box::new(
                    MtpDrafter::from_shard(engine, &factory_path, k_cap)
                        .expect("real MTP drafter must build"),
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
            "mtp spec: {} tokens in {:.1}s ({:.2} tok/s); proposed {proposed} accepted {accepted} ({:.1}%) over {steps} verify steps",
            ids.len(),
            spec_elapsed.as_secs_f64(),
            ids.len() as f64 / spec_elapsed.as_secs_f64(),
            100.0 * accepted as f64 / proposed.max(1) as f64,
        );
        assert_eq!(ids, base_ids, "speculative output must be lossless");
        assert_eq!(text, base_text);
        assert!(steps >= 1 && proposed >= 1, "the drafter never engaged");
        // A faithful port accepts most drafts on natural text; near-zero
        // acceptance means a pairing/porting bug (see the module docs).
        assert!(
            accepted * 5 >= proposed,
            "acceptance {accepted}/{proposed} is bug-level low"
        );

        // The backend worker thread tears the real engine down asynchronously
        // after the drop (nothing joins it); give it a moment so the test
        // harness's process exit does not race CUDA driver teardown mid-free
        // (observed as a post-"ok" SIGSEGV on the 29-GB real engine).
        drop(backend);
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
