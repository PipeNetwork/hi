//! GPU engine for DeepSeek-V4-Flash (`deepseek4`), host-orchestrated bring-up
//! (Stage 1 of `docs/deepseek-v4-flash-gpu-bringup.md`).
//!
//! Runs the exact same per-token state machine as the CPU oracle — this module
//! only supplies the heavy-linear side of [`crate::dsv4_cpu::DsV4Linear`]:
//!
//! - Every NON-expert matrix is made GPU-resident once at load. Float tensors
//!   upload as-is (F32/F16/BF16 served natively by cuBLAS); quantized tensors
//!   dequantize ONCE on device to f16 (dequant kernel + narrowing cast, the
//!   same recipe as gpu.rs's `into_f16`) and are then served by f16 GEMV with
//!   f32 accumulation. Resident budget for the real model is ~13 GB.
//! - The block-diagonal output projection (`attn_output_a`) is split into one
//!   buffer per group at load, so each group runs as an ordinary GEMV against
//!   its input slice (cuBLAS wrappers take whole buffers, not offsets).
//! - Packed rank-3 experts (~140 GB of MXFP4 in the real model) stay in the
//!   mmap'd GGUF on host, but their packed slices are served through a
//!   GPU-resident LRU pool (Stage 1b): equal-size slots carved out of a few
//!   large arena chunks, keyed by (layer, projection, expert). A hit runs the
//!   fused MXFP4 GEMV (M=1) straight against the resident slot; a miss evicts
//!   the least-recently-used slot and uploads the packed bytes into it first.
//!   Gate/up [4096→2048] and down [2048→4096] slices have the same element
//!   count, so one uniform slot size covers all projections. Budget comes from
//!   `HI_DSV4_EXPERT_POOL_GB` (default: min(72 GiB, free VRAM − 6 GiB));
//!   `HI_DSV4_EXPERT_PREFILL_POOL=1` fills the pool layer-major at load.
//!   Non-MXFP4 expert dtypes (the F32 test fixture, exotic quants) keep the
//!   Stage-1a per-call streaming path through a grow-only scratch buffer.
//!   Opt-in copy-stream prefetch (roadmap item 5, `HI_DSV4_COPY_STREAM=1`)
//!   issues a layer's missing-slice H2D copies on a dedicated non-blocking copy
//!   stream after routing, staged through double-buffered pinned host memory and
//!   ordered against the engine stream by CUDA events (fork before the copies,
//!   copy-done before the expert GEMVs); slices being copied are pinned in the
//!   LRU so they cannot be evicted mid-flight. `HI_DSV4_SPEC_PREFETCH=1` also
//!   warms the next layer's same-expert-id slices speculatively. Both default
//!   off: on the measured model copy-stream prefetch is throughput-neutral
//!   (bit-identical tokens either way) because decode is bottlenecked on the
//!   host-synchronous GEMV orchestration, not the expert copies — the mechanism
//!   is kept wired and tested for when that bottleneck is addressed.
//!   `HI_DSV4_NO_COPY_STREAM=1` hard-disables it for bisection.
//! - Wave-2 Stage 2a (roadmap items 1+4): the whole MoE BLOCK of each layer
//!   runs device-side through [`DsV4Linear::moe_block`] — ONE H2D upload of
//!   the batch activations, per-token router GEMVs (M=1, the host path's
//!   exact cuBLAS calls), a scoring+selection kernel (sqrt-softplus, hash
//!   tables / selection bias, serial per-token top-k with the lower-index
//!   tie-break), ONE small D2H readback of topk_ids (4*top_k bytes/token, the
//!   only interior sync — it services the expert-pool LRU), back-to-back
//!   fused MXFP4 expert GEMVs chained device-side through the SwiGLU-clamp
//!   and weighted-accumulate kernels, the shared expert, and ONE ys download.
//!   That replaces ~22 host-blocking GEMV round-trips per layer per token
//!   with 2 syncs per layer per batch. Every numeric op mirrors the host path
//!   exactly — the elementwise kernels use bit-exact ports of glibc 2.39's
//!   expf/logf (validated exhaustively over all 2^32 f32 inputs) with
//!   explicit roundings, so device-MoE activations are BIT-IDENTICAL to the
//!   host path and the greedy-token parity gate is deterministic, not
//!   statistical. `HI_DSV4_NO_DEVICE_MOE=1` kills the device path (exact host
//!   fallback, for bisection); non-MXFP4 experts (the F32 fixture), a missing
//!   pool, or off-grid dims fall back automatically. Counters in
//!   [`DsV4MoeBlockStats`] via [`DeepSeekV4GpuEngine::moe_block_stats`].
//! - Wave-2 Stage 2b (roadmap items 2-3): DECODE runs as a fully
//!   device-resident step ([`DsV4Linear::try_device_step`]). Per token the
//!   host uploads only the token id (+ a ~1 KB rope sin/cos table computed
//!   with the host libm); the embedding row is gathered from a packed device
//!   copy of `token_embd` (bit-exact dequant), all 43 layers run device-side
//!   — hyper-connection pre/post with sinkhorn, exact rms norms, resident-f16
//!   GEMVs at fixed 256-byte-aligned addresses, rope, latent-MQA attention
//!   over a device KV ring + compressed-block cache, APE compressor
//!   completions, the lightning indexer, and the inlined Stage-2a MoE core —
//!   then the hyper head + lm head. The only host syncs are the per-layer MoE
//!   topk_ids readback (24 B, services the expert-pool LRU), expert-pool miss
//!   uploads (inherent H2D), and ONE end-of-step arena download carrying the
//!   logits plus the step's state delta, which is replayed into the host
//!   [`crate::dsv4_cpu::DsV4State`] mirror immediately — the mirror stays
//!   authoritative at every position, so prefix-cache snapshots/truncation
//!   keep working unchanged. Every kernel is a bit-exact port of the host
//!   math (serial folds, the glibc expf/logf ports, host-computed rope
//!   tables), so a device step reproduces the host step bit for bit; the
//!   fixture gate measures literal 0.0 logit drift over 14 steps and the
//!   real-model gate identical 64-token greedy continuations.
//!   `HI_DSV4_HOST_STEP=1` declines every device step (the ENTIRE
//!   pre-Stage-2b host path stays selectable and bit-identical); unsupported
//!   shapes decline automatically. Counters in [`DsV4StepStats`] via
//!   [`DeepSeekV4GpuEngine::step_stats`].
//! - Prefill keeps the host-orchestrated chunked path (rayon host math +
//!   batched heavy linears): at chunk 64 it is ~2x faster than routing prompt
//!   tokens through sequential device steps, so retiring host attention for
//!   prefill awaits batched (B×) variants of the Stage-2b kernels.
//!
//! Chunked-prefill batching modes: by default `mul_mat` serves a chunk as
//! per-token `mul_vec` calls — the exact cuBLAS/kernel invocations of the
//! sequential path, so a chunked prefill reproduces a sequential run
//! bit-for-bit (the real-model parity gate measures literal 0.0 drift).
//! `HI_DSV4_PREFILL_GEMM=1` opts into true GEMM batching (one GEMM per
//! resident matrix per chunk; experts dequantized once per unique expert then
//! GEMM'd over their tokens): ~2x faster prefill, but GEMM reduction order
//! perturbs activations by ~1e-6, which can flip near-tied top-k routing
//! (256-expert MoE, indexer top-512) — outputs then drift from a sequential
//! run at equivalent quality. `HI_DSV4_MULMAT_LOOP=dense,grouped,expert`
//! downgrades individual operator kinds back to loops inside GEMM mode (drift
//! bisection aid).
//!
//! Parity discipline: the CPU reference is the oracle; the test at the bottom
//! runs the same greedy generation on both providers and requires identical
//! tokens and near-identical logits (the fixture is F32, so the GPU path stays
//! in f32 end-to-end and only reduction order differs).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{GgufFile, GgufTensorType, GgufTokenizer, QwenGgufConfig, TensorInfo, TensorView};

use crate::dsv4_cpu::{
    CompressorWeights, DsV4Engine, DsV4ExpertTensors, DsV4Linear, DsV4MoeBlockCtx, DsV4State,
    DsV4StepMirror, RawExperts, RawMatrix, TensorKey, dequantize_elem_range, host_moe_block,
    v4_rope_sincos,
};
use crate::qwen_cpu::{QwenCpuRunOptions, QwenCpuRunOutput};
use crate::runtime::{Cublas, CudaRuntime, DeviceBuffer, Event, GemmDType, PinnedBuffer, Stream};

/// Public GPU entry point: [`DsV4Engine`] driven by [`DsV4GpuLinear`]. Same
/// API and output contract as `DeepSeekV4CpuReference`, backend-labelled
/// `cuda-dsv4`.
pub struct DeepSeekV4GpuEngine {
    engine: DsV4Engine<DsV4GpuLinear>,
}

impl DeepSeekV4GpuEngine {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(GgufFile::open(path)?)
    }

    /// Takes ownership of the GGUF: expert weights are streamed from the mmap
    /// per token, so the file must stay open for the engine's lifetime.
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        // Fail with the runtime's clear diagnostics (no device / no driver)
        // before any allocation is attempted.
        CudaRuntime::probe()?;
        let gguf = Arc::new(gguf);
        let linear = DsV4GpuLinear::new(gguf.clone())?;
        let engine = DsV4Engine::new(gguf, linear, "cuda-dsv4")?;
        // The engine enumerates every dense/grouped matrix its forward pass
        // can touch; uploading exactly that set keeps the provider honest (a
        // mul_vec against anything else is a bug and fails fast).
        engine
            .linear()
            .upload_resident(&engine.resident_matrices())?;
        // Expert pool sizing runs after the resident upload so the auto budget
        // sees the free VRAM the residents actually left behind.
        engine.linear().init_expert_pool()?;
        Ok(Self { engine })
    }

    pub fn config(&self) -> &QwenGgufConfig {
        self.engine.config()
    }

    pub fn tokenizer(&self) -> &GgufTokenizer {
        self.engine.tokenizer()
    }

    /// The shared per-token state machine, for callers that drive decoding one
    /// step at a time (`new_state` / `step`) instead of through the monolithic
    /// run entry points — the serving backend streams tokens this way.
    pub(crate) fn engine(&self) -> &DsV4Engine<DsV4GpuLinear> {
        &self.engine
    }

    /// Expert-pool counters; `None` when the pool is disabled or the model has
    /// no packed MXFP4 experts (everything then streams per call).
    pub fn pool_stats(&self) -> Option<DsV4ExpertPoolStats> {
        self.engine.linear().pool_stats()
    }

    /// Device MoE-block counters (Wave-2 Stage 2a): how many `moe_block`
    /// calls ran device-side vs fell back to the host path, and the CUDA ops
    /// enqueued / host syncs paid by the device path.
    pub fn moe_block_stats(&self) -> DsV4MoeBlockStats {
        self.engine.linear().moe_block_stats()
    }

    /// Device decode-step counters (Wave-2 Stage 2b): steps served
    /// device-side vs host, state restores, and per-step launch/sync tallies.
    pub fn step_stats(&self) -> DsV4StepStats {
        self.engine.linear().step_stats()
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
}

/// One GPU-resident matrix: row-major [rows, cols] in `dtype` (F32 uploaded
/// as-is, F16 for dequantized quants, BF16 native).
struct ResidentMatrix {
    buffer: DeviceBuffer,
    rows: usize,
    cols: usize,
    dtype: GemmDType,
}

enum ResidentEntry {
    Dense(ResidentMatrix),
    /// Block-diagonal matrix split into per-group [rank, cols] buffers, in
    /// group order.
    Grouped(Vec<ResidentMatrix>),
}

/// Stage-1b expert pool tuning. The auto budget caps at 72 GiB and leaves
/// 6 GiB of VRAM headroom; slots pack into ~1 GiB arena chunks because a
/// cudaMalloc per ~4.25 MiB slot rounds up to 2 MiB granularity and measurably
/// wastes ~40% of the budget. Slot offsets stay 256-byte aligned for any
/// future vectorized weight loads (the current kernel reads bytes).
const DSV4_POOL_DEFAULT_MAX_BYTES: usize = 72 << 30;
const DSV4_POOL_HEADROOM_BYTES: usize = 6 << 30;
const DSV4_POOL_CHUNK_TARGET_BYTES: usize = 1 << 30;
const DSV4_POOL_SLOT_ALIGN: usize = 256;
/// One stats line on stderr every this many pooled expert GEMVs (~1.5x per
/// decoded token on the real model), plus a final line at provider drop.
const DSV4_POOL_LOG_EVERY_CALLS: u64 = 512;
/// Minimum tokens routed to one expert in a prefill chunk before the batched
/// expert path dequantizes the whole slice for a GEMM. Below this the
/// per-token fused GEMV moves less memory: it reads the ~4.25 MiB packed
/// slice per token, while the GEMM path writes and re-reads its ~32 MiB f32
/// image once regardless of token count.
const DSV4_EXPERT_GEMM_MIN_TOKENS: usize = 8;
/// LRU list terminator for [`PoolSlot::prev`]/[`PoolSlot::next`].
const POOL_NIL: u32 = u32::MAX;
/// Copy-stream prefetch (roadmap item 5): pinned staging capacity, in slots.
/// Each pinned region is one `slot_stride` (~4.25 MiB on the real model), so
/// the default 48 pins ~204 MiB of host memory — enough to stage a full decode
/// layer's within-layer misses (top-k experts x 3 projections) plus the
/// speculative next-layer batch. Copies beyond this cap per layer fall back to
/// the synchronous demand path. Override with `HI_DSV4_PREFETCH_STAGING_SLOTS`.
const DSV4_PREFETCH_STAGING_SLOTS: usize = 48;

/// Identity of one pooled expert slice: (layer, projection, expert) with
/// projection 0 = gate, 1 = up, 2 = down — the same key shape as
/// `expert_pool::ExpertKey`, but this pool serves per-slice fused GEMVs
/// instead of rewriting grouped-kernel pointer tables.
type ExpertSlotKey = (u32, u8, u32);

/// Load-time identity + geometry of one packed rank-3 MXFP4 expert tensor.
#[derive(Clone, Copy, Debug)]
struct ExpertTensorMeta {
    layer: u32,
    proj: u8,
    bytes_per_expert: usize,
    expert_count: usize,
}

/// Cumulative [`DsV4ExpertPool`] counters, exposed through
/// [`DeepSeekV4GpuEngine::pool_stats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DsV4ExpertPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Slots loaded by the `HI_DSV4_EXPERT_PREFILL_POOL=1` warmup (deliberately
    /// not counted as misses so hit rate reflects the demand path only).
    pub prefilled: u64,
    /// Packed bytes copied host-to-device (misses + prefill + prefetch).
    pub bytes_uploaded: u64,
    /// Slices uploaded ahead of demand on the copy stream by the within-layer
    /// prefetch (roadmap item 5a). A slice prefetched here then lands as a
    /// `hit` when the expert GEMV demands it, so these do not double-count as
    /// misses — they are the copies the synchronous path would have blocked on.
    pub prefetch_uploads: u64,
    /// Slices uploaded speculatively for the next layer's same-expert-ids
    /// (roadmap item 5b, `HI_DSV4_SPEC_PREFETCH=1`).
    pub spec_uploads: u64,
    /// Speculative uploads that were still resident when their layer actually
    /// routed to them: `spec_hits / spec_uploads` is the speculative hit rate.
    pub spec_hits: u64,
}

/// One pool slot: which expert slice it holds and its links in the intrusive
/// LRU list (head = coldest, tail = hottest; free slots start at the head).
struct PoolSlot {
    key: Option<ExpertSlotKey>,
    prev: u32,
    next: u32,
    /// Set while an in-flight copy-stream prefetch is filling this slot for the
    /// current layer's batch; a pinned slot is never chosen as an eviction
    /// victim, so a slice being copied cannot be clobbered before it is used.
    /// Cleared at the next batch ([`DsV4ExpertPool::clear_pins`]) and when a
    /// demand GEMV consumes the slice.
    pinned: bool,
    /// Set when this slice was uploaded speculatively for the next layer
    /// (roadmap item 5b); cleared (and counted as a speculative hit) the first
    /// time its layer actually re-touches it.
    spec: bool,
    /// Permanently resident (never an eviction victim): the MTP drafter's
    /// layer-43 slices, touched every draft. Set only by
    /// [`DsV4GpuLinearInner::register_host_experts`], which guarantees enough
    /// unpinned slots remain for the trunk's working set.
    permanent: bool,
}

/// Outcome of [`DsV4ExpertPool::acquire_prefetch`].
enum PrefetchAcquire {
    /// The slice is already resident (nothing to copy); slot pinned for the batch.
    Resident(u32),
    /// A slot was evicted and pinned; the caller stages and copies the bytes.
    Loaded(u32),
    /// Every slot is pinned by the current batch — stop prefetching this layer.
    Full,
}

/// Outcome of one [`DsV4GpuLinear::prefetch_one`] slice.
enum PrefetchStep {
    /// A slice was staged and its async copy enqueued (consumed a staging region).
    Copied,
    /// Nothing to do: slice already resident, tensor unmanaged, or expert out of
    /// range. No staging region consumed.
    Skipped,
    /// The pool is fully pinned by this batch — stop prefetching.
    Full,
}

/// GPU-resident LRU pool of packed MXFP4 expert slices (Stage 1b). All slots
/// share one uniform stride: on the real model gate/up [4096→2048] and down
/// [2048→4096] slices carry the same 8M elements, hence the same packed size
/// (verified at build; a mixed-size model would just under-fill max-size
/// slots). Single-threaded like the rest of the provider.
struct DsV4ExpertPool {
    /// Slot backing store, `slots_per_chunk` slots per chunk (the last chunk
    /// may be short). Slot `i` lives in chunk `i / slots_per_chunk` at byte
    /// offset `(i % slots_per_chunk) * slot_stride`.
    chunks: Vec<DeviceBuffer>,
    slots_per_chunk: usize,
    slot_stride: usize,
    slots: Vec<PoolSlot>,
    lru_head: u32,
    lru_tail: u32,
    /// Slots pinned by the current prefetch batch (see [`PoolSlot::pinned`]),
    /// drained by [`DsV4ExpertPool::clear_pins`] at the start of the next batch.
    pinned_slots: Vec<u32>,
    resident: HashMap<ExpertSlotKey, u32>,
    /// Expert tensors the pool manages, by GGUF name; anything else falls back
    /// to the per-call streaming path.
    tensors: HashMap<String, ExpertTensorMeta>,
    stats: DsV4ExpertPoolStats,
}

impl DsV4ExpertPool {
    /// Resolve `key` to a slot for a DEMAND expert GEMV and mark it
    /// most-recently-used, evicting the coldest slot on a miss. Returns
    /// (slot, hit); on a miss the caller must upload the packed bytes before
    /// launching against the slot. Consuming a slice unpins it (a demand GEMV is
    /// the "use" a prefetch was pinned against) and, on the first re-touch of a
    /// speculatively-loaded slice, records a speculative hit. Safe against
    /// in-flight reads of the victim: demand uploads and GEMVs share the
    /// engine stream, so the copy is ordered after any queued use; prefetch
    /// copies on the copy stream are gated into the engine stream by a copy-done
    /// event before any demand GEMV runs.
    fn acquire(&mut self, key: ExpertSlotKey) -> (u32, bool) {
        if let Some(&idx) = self.resident.get(&key) {
            self.consume_hit(idx);
            return (idx, true);
        }
        // Prefer an unpinned victim; a fully-pinned pool (pathological tiny
        // pool) force-evicts the coldest non-permanent slot — safe because the
        // engine stream has already waited on the copy-done event, so that
        // slot's prefetch copy has completed and its GEMV, if any, was
        // enqueued earlier on the same stream. Permanent (MTP) slots are never
        // victims; registration guarantees non-permanent slots exist.
        let idx = self
            .find_victim()
            .or_else(|| self.find_forced_victim())
            .unwrap_or(self.lru_head);
        self.reassign(idx, key);
        self.slots[idx as usize].pinned = false;
        self.touch(idx);
        (idx, false)
    }

    /// Resolve `key` to a slot for a copy-stream PREFETCH. Unlike [`Self::acquire`]
    /// this never clobbers a slice pinned by the current batch: it returns
    /// [`PrefetchAcquire::Full`] when every slot is pinned, and the caller stops
    /// prefetching (the demand path serves the rest synchronously). A resident
    /// slice needs no copy; a loaded slot's packed bytes must be staged and
    /// copied. Either way the slot is pinned for the batch.
    fn acquire_prefetch(&mut self, key: ExpertSlotKey) -> PrefetchAcquire {
        if let Some(&idx) = self.resident.get(&key) {
            self.consume_hit(idx);
            self.pin(idx);
            return PrefetchAcquire::Resident(idx);
        }
        let Some(idx) = self.find_victim() else {
            return PrefetchAcquire::Full;
        };
        self.reassign(idx, key);
        self.pin(idx);
        self.touch(idx);
        PrefetchAcquire::Loaded(idx)
    }

    /// Touch a resident slot as most-recently-used and, on the first re-touch of
    /// a speculatively-loaded slice, retire its speculative mark as a hit.
    fn consume_hit(&mut self, idx: u32) {
        let slot = &mut self.slots[idx as usize];
        if slot.spec {
            slot.spec = false;
            self.stats.spec_hits += 1;
        }
        slot.pinned = false;
        self.touch(idx);
    }

    /// Evict whatever slice `idx` holds (if any) and give it `key`. Clears the
    /// speculative mark; the caller sets it again for a speculative load.
    fn reassign(&mut self, idx: u32, key: ExpertSlotKey) {
        if let Some(old) = self.slots[idx as usize].key.take() {
            self.resident.remove(&old);
            self.stats.evictions += 1;
        }
        self.slots[idx as usize].spec = false;
        self.slots[idx as usize].key = Some(key);
        self.resident.insert(key, idx);
    }

    /// Pin `idx` for the current prefetch batch (idempotent).
    fn pin(&mut self, idx: u32) {
        if !self.slots[idx as usize].pinned {
            self.slots[idx as usize].pinned = true;
            self.pinned_slots.push(idx);
        }
    }

    /// Release every pin from the previous prefetch batch. Called once at the
    /// start of each layer's prefetch so pins protect only the batch in flight.
    fn clear_pins(&mut self) {
        for idx in self.pinned_slots.drain(..) {
            self.slots[idx as usize].pinned = false;
        }
    }

    /// The coldest unpinned, non-permanent slot (LRU head walking toward the
    /// tail), or `None` when every slot is pinned by the current batch or
    /// permanently resident.
    fn find_victim(&self) -> Option<u32> {
        let mut idx = self.lru_head;
        while idx != POOL_NIL {
            let slot = &self.slots[idx as usize];
            if !slot.pinned && !slot.permanent {
                return Some(idx);
            }
            idx = slot.next;
        }
        None
    }

    /// The coldest non-permanent slot, ignoring batch pins — the force-evict
    /// fallback for a fully batch-pinned pool (see [`Self::acquire`]).
    fn find_forced_victim(&self) -> Option<u32> {
        let mut idx = self.lru_head;
        while idx != POOL_NIL {
            let slot = &self.slots[idx as usize];
            if !slot.permanent {
                return Some(idx);
            }
            idx = slot.next;
        }
        None
    }

    /// (chunk index, byte offset) backing slot `idx`.
    fn slot_location(&self, idx: u32) -> (usize, usize) {
        let idx = idx as usize;
        (
            idx / self.slots_per_chunk,
            (idx % self.slots_per_chunk) * self.slot_stride,
        )
    }

    fn touch(&mut self, idx: u32) {
        if self.lru_tail != idx {
            self.unlink(idx);
            self.push_tail(idx);
        }
    }

    fn unlink(&mut self, idx: u32) {
        let PoolSlot { prev, next, .. } = self.slots[idx as usize];
        match prev {
            POOL_NIL => self.lru_head = next,
            prev => self.slots[prev as usize].next = next,
        }
        match next {
            POOL_NIL => self.lru_tail = prev,
            next => self.slots[next as usize].prev = prev,
        }
    }

    fn push_tail(&mut self, idx: u32) {
        let tail = self.lru_tail;
        self.slots[idx as usize].prev = tail;
        self.slots[idx as usize].next = POOL_NIL;
        match tail {
            POOL_NIL => self.lru_head = idx,
            tail => self.slots[tail as usize].next = idx,
        }
        self.lru_tail = idx;
    }
}

/// Parse `blk.{layer}.ffn_{gate,up,down}_exps.weight` into pool metadata.
/// `None` keeps the tensor on the streaming path: not MXFP4, not rank-3, an
/// input dim off the 32-element block grid, or an unexpected name.
fn packed_expert_meta(info: &TensorInfo) -> Option<ExpertTensorMeta> {
    if info.dtype != GgufTensorType::MXFP4 || info.dimensions.len() != 3 {
        return None;
    }
    let rest = info.name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let layer: u32 = layer.parse().ok()?;
    let proj = match suffix {
        "ffn_gate_exps.weight" => 0u8,
        "ffn_up_exps.weight" => 1,
        "ffn_down_exps.weight" => 2,
        _ => return None,
    };
    let in_dim = usize::try_from(info.dimensions[0]).ok()?;
    let out_dim = usize::try_from(info.dimensions[1]).ok()?;
    let expert_count = usize::try_from(info.dimensions[2]).ok()?;
    if !in_dim.is_multiple_of(32) || u32::try_from(expert_count).is_err() {
        return None;
    }
    let per_expert = in_dim.checked_mul(out_dim)?;
    let bytes_per_expert = GgufTensorType::MXFP4.byte_len(per_expert as u64).ok()?;
    Some(ExpertTensorMeta {
        layer,
        proj,
        bytes_per_expert: usize::try_from(bytes_per_expert).ok()?,
        expert_count,
    })
}

/// `HI_DSV4_EXPERT_POOL_GB` (GiB, fractional allowed; 0 disables the pool):
/// device budget for the expert pool. `None` means unset — the caller
/// auto-sizes from free VRAM.
fn expert_pool_budget_env() -> Result<Option<usize>> {
    let Ok(raw) = std::env::var("HI_DSV4_EXPERT_POOL_GB") else {
        return Ok(None);
    };
    let gib: f64 = raw
        .trim()
        .parse()
        .with_context(|| format!("HI_DSV4_EXPERT_POOL_GB value '{raw}' is not a number"))?;
    if !gib.is_finite() || gib < 0.0 {
        bail!("HI_DSV4_EXPERT_POOL_GB must be a finite non-negative number, got '{raw}'");
    }
    Ok(Some((gib * (1u64 << 30) as f64) as usize))
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The one-line stats summary logged periodically and at provider drop.
fn format_pool_stats(stats: &DsV4ExpertPoolStats) -> String {
    let served = stats.hits + stats.misses;
    let rate = if served > 0 {
        100.0 * stats.hits as f64 / served as f64
    } else {
        0.0
    };
    let spec = if stats.spec_uploads > 0 {
        let spec_rate = 100.0 * stats.spec_hits as f64 / stats.spec_uploads as f64;
        format!(
            " spec {}/{} ({spec_rate:.1}% spec hit)",
            stats.spec_hits, stats.spec_uploads
        )
    } else {
        String::new()
    };
    format!(
        "hits {} misses {} ({rate:.1}% hit) evictions {} prefilled {} prefetched {} uploaded {:.2} GiB{spec}",
        stats.hits,
        stats.misses,
        stats.evictions,
        stats.prefilled,
        stats.prefetch_uploads,
        gib(stats.bytes_uploaded)
    )
}

/// Copy-stream prefetch machinery for the expert pool (roadmap item 5, patterned
/// on vLLM's `PrefetchOffloader`). A dedicated non-blocking copy stream issues a
/// layer's missing expert-slice H2D uploads ahead of the expert GEMVs — staged
/// through pinned host memory for true async DMA out of the pageable GGUF mmap —
/// with CUDA events ordering the copy and engine streams. Absent (via
/// `HI_DSV4_NO_COPY_STREAM=1`) the provider keeps the original synchronous
/// per-miss copy on the engine stream, for bisection and zero behaviour change.
struct CopyEngine {
    /// Non-blocking so a synchronous demand `cudaMemcpy` on the engine side does
    /// not serialize with prefetch copies; all ordering is via the events below.
    stream: Stream,
    /// Engine -> copy fork: prefetch copies wait for expert reads enqueued
    /// before the batch, so a copy never clobbers a slot still being read.
    fork: Event,
    /// Copy -> engine join: the engine stream waits on this before the layer's
    /// expert GEMVs, so a GEMV never reads a half-copied slot.
    done: Event,
    /// Double-buffered pinned staging: batch `b` uses buffer `b % len`, so its
    /// host-side reuse waits on a guard recorded `len` batches earlier (already
    /// long complete) instead of stalling on the immediately-previous batch's
    /// in-flight DMA. Allocated once the pool's slot stride is known.
    staging: Vec<StagingBuf>,
    slot_stride: usize,
    staging_slots: usize,
    /// Monotonic batch counter; `cur = (batch - 1) % staging.len()` selects the
    /// in-progress batch's buffer.
    batch: usize,
    cur: usize,
}

/// One generation of pinned staging plus the event gating its host-side reuse.
struct StagingBuf {
    pinned: PinnedBuffer,
    /// Recorded on the copy stream after this generation's copies; the batch
    /// that next reuses this buffer host-waits it before overwriting.
    guard: Event,
    armed: bool,
}

/// Pinned-staging generations (double buffering). Two is enough to decouple a
/// batch's staging memcpy from the previous batch's in-flight DMA.
const DSV4_STAGING_GENERATIONS: usize = 2;

impl CopyEngine {
    fn create(staging_slots: usize) -> Result<Self> {
        Ok(Self {
            stream: Stream::create_non_blocking()
                .context("creating expert prefetch copy stream")?,
            fork: Event::create().context("creating prefetch fork event")?,
            done: Event::create().context("creating prefetch done event")?,
            staging: Vec::new(),
            slot_stride: 0,
            staging_slots,
            batch: 0,
            cur: 0,
        })
    }

    /// Allocate the double-buffered pinned staging on first use, sized to the
    /// pool's slot stride.
    fn ensure_staging(&mut self, slot_stride: usize) -> Result<()> {
        if self.staging.is_empty() {
            let bytes = slot_stride
                .checked_mul(self.staging_slots)
                .context("prefetch staging byte count overflows usize")?;
            for _ in 0..DSV4_STAGING_GENERATIONS {
                self.staging.push(StagingBuf {
                    pinned: PinnedBuffer::alloc(bytes)
                        .context("allocating pinned prefetch staging")?,
                    guard: Event::create().context("creating prefetch staging guard event")?,
                    armed: false,
                });
            }
            self.slot_stride = slot_stride;
        }
        Ok(())
    }

    /// Select this batch's staging generation and wait (host-side) for its
    /// prior use to finish before it is overwritten.
    fn begin_batch(&mut self) -> Result<()> {
        self.cur = self.batch % self.staging.len();
        self.batch += 1;
        let buf = &self.staging[self.cur];
        if buf.armed {
            buf.guard.synchronize()?;
        }
        Ok(())
    }

    /// Guard this batch's staging generation for its next reuse.
    fn end_batch(&mut self) -> Result<()> {
        let buf = &mut self.staging[self.cur];
        buf.guard.record(&self.stream)?;
        buf.armed = true;
        Ok(())
    }

    /// Stage `bytes` into region `region` of this batch's generation and enqueue
    /// the async H2D into `dst[dst_offset..]` on the copy stream.
    fn stage_upload(
        &self,
        region: usize,
        dst: &DeviceBuffer,
        dst_offset: usize,
        bytes: &[u8],
    ) -> Result<()> {
        let staging = &self.staging[self.cur].pinned;
        let src_offset = region * self.slot_stride;
        staging.copy_in(src_offset, bytes)?;
        dst.copy_from_pinned_at_async(dst_offset, staging, src_offset, bytes.len(), &self.stream)
    }
}

/// Counters for the Wave-2 Stage 2a device MoE block, exposed through
/// [`DeepSeekV4GpuEngine::moe_block_stats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DsV4MoeBlockStats {
    /// `moe_block` calls served by the device path.
    pub device_blocks: u64,
    /// `moe_block` calls served by the host fallback (`HI_DSV4_NO_DEVICE_MOE=1`,
    /// non-MXFP4 experts, or an unsupported layer shape).
    pub host_blocks: u64,
    /// CUDA ops the device path enqueued (kernel launches, GEMMs, async
    /// copies) — none of them synchronize the host.
    pub launches: u64,
    /// Host synchronizations the device path paid: the per-layer topk_ids
    /// readback, the ys download, pool-miss uploads, and one-time
    /// bias/tid2eid table uploads.
    pub syncs: u64,
}

/// Outcome of [`DsV4GpuLinear::device_moe_core`]: either the block outputs
/// landed in `scratch.ys`, or the pool cannot hold the batch's slice set
/// simultaneously and the caller must run the (eviction-safe) host path.
/// Both variants carry the CUDA launches / host syncs the core performed.
enum MoeCoreOutcome {
    Done { launches: u64, syncs: u64 },
    PoolTooSmall { launches: u64, syncs: u64 },
}

/// Counters for the Wave-2 Stage 2b device decode step, exposed through
/// [`DeepSeekV4GpuEngine::step_stats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DsV4StepStats {
    /// Decode steps served fully device-side.
    pub device_steps: u64,
    /// Steps declined to the host path (`HI_DSV4_HOST_STEP=1` or an
    /// unsupported model shape).
    pub host_steps: u64,
    /// Full host→device state restores (fresh conversation, prefill handoff,
    /// truncation, snapshot resume).
    pub restores: u64,
    /// CUDA ops enqueued by device steps (kernels, GEMVs, async copies) —
    /// none of them synchronize the host.
    pub launches: u64,
    /// Host synchronizations paid by device steps: the per-layer MoE topk_ids
    /// readback, expert-pool miss uploads, the ONE end-of-step arena download
    /// (logits + state delta), and any in-step host-MoE fallback downloads.
    /// Restore uploads are counted separately under `restore_syncs`.
    pub syncs: u64,
    /// Host-blocking copies performed by state restores.
    pub restore_syncs: u64,
}

/// Grow-only device buffers reused across [`DsV4GpuLinear::device_moe_block`]
/// calls: sized for the largest batch seen, never shrunk (a few tens of MB at
/// prefill batch 64 on the real model). Freeing per call would pay a
/// synchronizing cudaFree per buffer per layer — the exact cost Stage 2a
/// removes. The `xs_f16_valid`/`xs_bf16_valid` flags are per-call (reset on
/// entry): they dedupe the activation casts when the router and shared expert
/// share a dtype.
#[derive(Default)]
struct DeviceMoeScratch {
    /// [b, embed] f32 activations, uploaded once per call.
    xs_f32: Option<DeviceBuffer>,
    /// [b, embed] f16 / bf16 casts of `xs_f32` (built on demand).
    xs_f16: Option<DeviceBuffer>,
    xs_bf16: Option<DeviceBuffer>,
    xs_f16_valid: bool,
    xs_bf16_valid: bool,
    /// [b, experts] f32 router logits.
    router_logits: Option<DeviceBuffer>,
    /// [b] i32 token ids (hash layers only).
    token_ids: Option<DeviceBuffer>,
    /// [b, top_k] i32 selected experts / f32 mixture weights.
    topk_ids: Option<DeviceBuffer>,
    topk_weights: Option<DeviceBuffer>,
    /// [b, top_k, ff] f32 gate/up projections; `gate` becomes the SwiGLU
    /// hidden in place.
    gate: Option<DeviceBuffer>,
    up: Option<DeviceBuffer>,
    /// [b, top_k, embed] f32 per-(token, rank) expert outputs.
    expert_out: Option<DeviceBuffer>,
    /// Shared expert: [b, sff] f32 gate/up (gate becomes the hidden in
    /// place), the hidden cast to the down matrix dtype, and [b, embed] out.
    shared_gate: Option<DeviceBuffer>,
    shared_up: Option<DeviceBuffer>,
    shared_cast: Option<DeviceBuffer>,
    shared_out: Option<DeviceBuffer>,
    /// [b, embed] f32 block output.
    ys: Option<DeviceBuffer>,
}

/// (Re)allocate a grow-only scratch slot to at least `min_bytes` and return
/// the buffer.
fn ensure_scratch<'a>(
    slot: &'a mut Option<DeviceBuffer>,
    min_bytes: usize,
    what: &str,
) -> Result<&'a DeviceBuffer> {
    if slot
        .as_ref()
        .is_none_or(|buffer| buffer.bytes() < min_bytes)
    {
        *slot = Some(
            DeviceBuffer::alloc(min_bytes).with_context(|| format!("allocating {what} scratch"))?,
        );
    }
    Ok(slot.as_ref().expect("scratch slot was just populated"))
}

// ---------------------------------------------------------------------------
// Wave-2 Stage 2b: device-resident decode step.
//
// One `try_device_step` call runs a WHOLE decode token on the GPU: embedding
// gather from a packed device copy, all 43 layers (hyper-connection pre/post
// + sinkhorn, exact rms norms, the resident-GEMV projections at fixed device
// addresses, rope from host-computed sin/cos tables, latent-MQA attention
// over the device KV ring + compressed blocks, APE compressor completions,
// the lightning indexer, and the inlined Stage-2a device MoE core), then the
// hyper head + lm head. The ONLY host syncs are the per-layer MoE topk_ids
// readback (24 B — it services the expert-pool LRU), expert-pool miss
// uploads, and ONE end-of-step download of the step arena (logits + the
// state delta, which is immediately replayed into the host `DsV4State`
// mirror so prefix-cache snapshots stay exact and always available).
//
// Numerics: every kernel is a bit-exact port of the host math (serial folds,
// glibc expf/logf ports, host-uploaded rope tables) and every cuBLAS call is
// the identical M=1 GEMV shape at 256-byte-aligned addresses, so a device
// step reproduces the host step (`HI_DSV4_HOST_STEP=1`) bit for bit.

/// Round a byte count up to 256 so every cuBLAS operand keeps the alignment
/// class of a fresh allocation (algorithm-selection reproducibility).
const fn align256(bytes: usize) -> usize {
    bytes.div_ceil(256) * 256
}

/// Round an f32-element count up to a 256-byte boundary (arena slots).
const fn align_elems(elems: usize) -> usize {
    elems.div_ceil(64) * 64
}

/// `HI_DSV4_HOST_STEP=1`: decline every device step so the engine runs the
/// pre-Stage-2b host step (bisection kill switch; bit-identical to today).
fn host_step_forced_from_env() -> bool {
    std::env::var("HI_DSV4_HOST_STEP").ok().as_deref() == Some("1")
}

fn upload_f32(values: &[f32], what: &str) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::alloc(std::mem::size_of_val(values).max(4))
        .with_context(|| format!("allocating {what}"))?;
    buffer.copy_from_host(values)?;
    Ok(buffer)
}

fn alloc_dev(bytes: usize, what: &str) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc(bytes.max(4)).with_context(|| format!("allocating {what}"))
}

/// Lazily-built device-step resources; `Unsupported` declines every step to
/// the host path (reason logged once at build).
enum DeviceStepBuild {
    Unsupported,
    Ready(Box<DeviceStepRes>),
}

/// Where the device step gathers embedding rows.
enum EmbedSrc {
    /// Packed GGUF bytes resident on device (bit-exact device dequant);
    /// code: 0 = F32, 1 = F16, 2 = BF16, 3 = Q8_0.
    Packed { buffer: DeviceBuffer, code: u32 },
    /// Exotic dtype: dequantize on host (the host path's exact bytes), upload
    /// into staging, broadcast from there.
    Host { staging: DeviceBuffer },
}

/// One uploaded hyper-connection mixer.
struct DevHc {
    func: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
    rows: usize,
}

/// Uploaded APE-compressor constants + device row geometry.
struct DevCompConsts {
    ape: DeviceBuffer,
    norm: DeviceBuffer,
    ratio: usize,
    dim: usize,
    width: usize,
    /// Row stride (f32 elements) of the gate/kv projection outputs in the
    /// shared scratch (padded to 256 B so every GEMV output stays aligned).
    out_stride: usize,
    /// Pending activation row stride in BYTES (embed elements in `x_dtype`,
    /// padded to 256 B so every GEMV input stays aligned).
    pending_row_bytes: usize,
}

struct DevLayerConsts {
    attn_norm: DeviceBuffer,
    ffn_norm: DeviceBuffer,
    q_a_norm: DeviceBuffer,
    kv_norm: DeviceBuffer,
    sinks: Option<DeviceBuffer>,
    hc_attn: DevHc,
    hc_ffn: DevHc,
    /// Index into the per-base rope table blocks.
    base_idx: usize,
    /// Common dtype of every matrix consuming the attn-normed activation
    /// (q_a, kv, compressor gate/kv, indexer gate/kv/proj).
    x_dtype: GemmDType,
    /// Common dtype of q_b and the indexer q_b (consume the q latent).
    qr_dtype: GemmDType,
    wo_a_dtype: GemmDType,
    wo_b_dtype: GemmDType,
    /// Both the per-group input slices and outputs of the block-diagonal
    /// projection are naturally 256-byte aligned, so the GEMVs can run
    /// straight off contiguous buffers (the real model); otherwise the step
    /// stages them through padded rows (the tiny fixture).
    wo_direct: bool,
    comp: Option<DevCompConsts>,
    idx: Option<DevCompConsts>,
}

/// Fixed step-arena layout (f32 element offsets, 256-byte aligned): logits
/// first, then per-layer mirror slots (raw-KV latent, attn-normed activation,
/// compressor/indexer block completions). One D2H copy per step downloads the
/// whole arena.
struct ArenaLayout {
    kv_off: Vec<usize>,
    x_off: Vec<usize>,
    comp_off: Vec<Option<(usize, usize)>>,
    idx_off: Vec<Option<(usize, usize)>>,
    total: usize,
}

/// Per-conversation device state.
struct DevState {
    tag: u64,
    pos: usize,
    layers: Vec<DevLayerState>,
}

struct DevLayerState {
    /// Circular raw-KV ring [window, head_dim] f32; position p in slot
    /// p % window (only the trailing window is ever attention-visible, so
    /// the device ring ignores host-side ring slack).
    ring: DeviceBuffer,
    comp: Option<DevCompState>,
    idx: Option<DevCompState>,
}

struct DevCompState {
    /// Compressed K/V rows [cap_blocks, dim] f32, tight rows; grown by
    /// doubling (old buffer retired until the end-of-step sync).
    keys: DeviceBuffer,
    values: DeviceBuffer,
    cap_blocks: usize,
    /// Pending activation rows in the layer's x_dtype (padded rows).
    pending: DeviceBuffer,
}

/// Everything the device step owns: uploaded small weights, fixed-address
/// activation buffers, the step arena, and the per-conversation device state.
struct DeviceStepRes {
    embed: usize,
    heads: usize,
    head_dim: usize,
    rope_dims: usize,
    q_lora: usize,
    o_groups: usize,
    o_rank: usize,
    group_features: usize,
    window: usize,
    hc: usize,
    sinkhorn: usize,
    idx_heads: usize,
    idx_key: usize,
    idx_top_k: usize,
    vocab: usize,
    rms_eps: f32,
    hc_eps: f32,
    attn_scale: f32,
    idx_head_scale: f32,
    idx_key_scale: f32,
    head_dtype: GemmDType,
    rope_bases: Vec<f32>,
    layers: Vec<DevLayerConsts>,
    hyper: DevHc,
    hyper_scale0: f32,
    output_norm: DeviceBuffer,
    embed_src: EmbedSrc,
    arena_layout: ArenaLayout,

    arena: DeviceBuffer,
    streams_a: DeviceBuffer,
    streams_b: DeviceBuffer,
    post: DeviceBuffer,
    comb: DeviceBuffer,
    x_cast: DeviceBuffer,
    qr: DeviceBuffer,
    qr_cast: DeviceBuffer,
    q: DeviceBuffer,
    kv_tmp: DeviceBuffer,
    attn_out: DeviceBuffer,
    attn_out_cast: DeviceBuffer,
    attn_final: DeviceBuffer,
    wo_pad_in: DeviceBuffer,
    wo_pad_in_stride: usize,
    proj_pad: DeviceBuffer,
    proj_pad_stride: usize,
    proj_flat: DeviceBuffer,
    proj_cast: DeviceBuffer,
    comp_gates: DeviceBuffer,
    comp_kvs: DeviceBuffer,
    ffn_y: DeviceBuffer,
    ffn_out: DeviceBuffer,
    idx_qi: DeviceBuffer,
    idx_w: DeviceBuffer,
    idx_sel: DeviceBuffer,
    idx_scores: Option<DeviceBuffer>,
    idx_marks: Option<DeviceBuffer>,
    attn_w: Option<DeviceBuffer>,
    attn_wn: Option<DeviceBuffer>,
    attn_max_keys: usize,
    hyper_hidden: DeviceBuffer,
    hyper_cast: DeviceBuffer,
    rope_buf: DeviceBuffer,
    restore_staging: Option<DeviceBuffer>,

    state: Option<DevState>,
    tag_counter: u64,
    /// Buffers replaced mid-step (compressed-cache growth); still referenced
    /// by enqueued kernels, so their cudaFree waits for the end-of-step sync.
    retired: Vec<DeviceBuffer>,
}

fn dense_entry<'r>(
    resident: &'r HashMap<String, ResidentEntry>,
    matrix: &RawMatrix,
) -> Result<&'r ResidentMatrix> {
    match resident.get(&matrix.name) {
        Some(ResidentEntry::Dense(entry)) => Ok(entry),
        _ => bail!(
            "tensor {} is not GPU-resident as a dense matrix",
            matrix.name
        ),
    }
}

fn grouped_entry<'r>(
    resident: &'r HashMap<String, ResidentEntry>,
    matrix: &RawMatrix,
) -> Result<&'r [ResidentMatrix]> {
    match resident.get(&matrix.name) {
        Some(ResidentEntry::Grouped(groups)) => Ok(groups),
        _ => bail!(
            "tensor {} is not GPU-resident as a grouped matrix",
            matrix.name
        ),
    }
}

/// The CUDA [`DsV4Linear`]: resident non-expert weights + streamed expert
/// slices. Single-threaded by design (RefCell state, thread-local buffer
/// pool); build, use, and drop it on one thread like the other GPU models.
///
/// A thin `Rc` handle over [`DsV4GpuLinearInner`]: the engine owns one handle,
/// and the MTP drafter (`dsv4_mtp`) clones another so its `'static`
/// [`crate::dsv4_backend::Drafter`] box can share the SAME device resources —
/// resident matrices, streams, and crucially the expert pool — without
/// borrowing the engine. `Rc` (not `Arc`) keeps the single-thread discipline
/// honest: the handle cannot leave the engine worker thread.
#[derive(Clone)]
pub(crate) struct DsV4GpuLinear {
    inner: std::rc::Rc<DsV4GpuLinearInner>,
}

impl DsV4GpuLinear {
    fn new(gguf: Arc<GgufFile>) -> Result<Self> {
        Ok(Self {
            inner: std::rc::Rc::new(DsV4GpuLinearInner::new(gguf)?),
        })
    }

    /// Rebuild the copy engine (tests only). Requires the sole handle — call
    /// before the provider is shared with an engine or drafter.
    #[cfg(test)]
    pub(crate) fn set_copy_stream_enabled(&mut self, enabled: bool) -> Result<()> {
        std::rc::Rc::get_mut(&mut self.inner)
            .expect("set_copy_stream_enabled needs the sole provider handle")
            .set_copy_stream_enabled(enabled)
    }
}

impl std::ops::Deref for DsV4GpuLinear {
    type Target = DsV4GpuLinearInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DsV4Linear for DsV4GpuLinear {
    fn mul_vec(&self, key: TensorKey<'_>, x: &[f32]) -> Result<Vec<f32>> {
        self.inner.mul_vec(key, x)
    }

    fn prefetch_experts(&self, tensors: DsV4ExpertTensors<'_>, expert_ids: &[usize]) -> Result<()> {
        self.inner.prefetch_experts(tensors, expert_ids)
    }

    fn set_exact_batching(&self, exact: bool) {
        self.inner.set_exact_batching(exact);
    }

    /// Wave-2 Stage 2b: run the WHOLE decode step device-side (embedding
    /// gather, hyper connections, attention, compressors, indexer, inlined
    /// MoE, hyper head + lm head; one end-of-step download of logits + state
    /// delta). Declines to the host step under `HI_DSV4_HOST_STEP=1` or when
    /// the model shape is unsupported. Lives on the handle (not the inner)
    /// because the engine's type parameter is the handle.
    fn try_device_step(
        &self,
        engine: &DsV4Engine<Self>,
        state: &mut DsV4State,
        token: u32,
    ) -> Option<Result<Vec<f32>>> {
        self.inner.try_device_step_impl(engine, state, token)
    }

    fn moe_block(
        &self,
        ctx: &DsV4MoeBlockCtx<'_>,
        xs: &[Vec<f32>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        self.inner.moe_block(ctx, xs, tokens)
    }

    fn mul_mat(&self, key: TensorKey<'_>, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.inner.mul_mat(key, xs)
    }
}

/// A host-side dense weight payload for [`DsV4GpuLinearInner::register_host_dense`]
/// (the MTP module's safetensors-sourced matrices), row-major `[rows, cols]`.
pub(crate) enum HostDenseData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
}

impl HostDenseData {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(bits) | Self::Bf16(bits) => bits.len(),
        }
    }
}

fn check_host_payload(matrix: &RawMatrix, len: usize) -> Result<()> {
    if len != matrix.rows * matrix.cols {
        bail!(
            "host payload for {} has {len} values; expected {} x {}",
            matrix.name,
            matrix.rows,
            matrix.cols
        );
    }
    Ok(())
}

/// One host-owned packed expert tensor registered by the MTP drafter
/// (`dsv4_mtp`): the official shard's experts repacked into the exact GGUF
/// layout (`fp4_to_gguf_mxfp4`), served through the SAME pool/streaming paths
/// as the GGUF-mmap trunk tensors — the byte source is the only difference.
pub(crate) struct HostExpertBlob {
    dtype: GgufTensorType,
    /// Rank-3 `[in, out, experts]` packed payload, expert-major like the GGUF.
    bytes: Vec<u8>,
}

/// Byte source for one packed expert tensor: the GGUF mmap (trunk layers) or
/// a registered host blob (the MTP module). Owning the `Rc` keeps the blob
/// borrowable past the registry's `RefCell` guard.
enum ExpertSource<'a> {
    Gguf(TensorView<'a>),
    Blob(std::rc::Rc<HostExpertBlob>),
}

impl ExpertSource<'_> {
    fn dtype(&self) -> GgufTensorType {
        match self {
            Self::Gguf(view) => view.info.dtype,
            Self::Blob(blob) => blob.dtype,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Gguf(view) => view.bytes,
            Self::Blob(blob) => &blob.bytes,
        }
    }

    /// Dequantize an element subrange (the [`dequantize_elem_range`] math over
    /// either source).
    fn dequant_range(&self, elem_offset: usize, elem_count: usize) -> Result<Vec<f32>> {
        match self {
            Self::Gguf(view) => dequantize_elem_range(view, elem_offset, elem_count),
            Self::Blob(blob) => {
                let dtype = blob.dtype;
                let start = usize::try_from(dtype.byte_len(elem_offset as u64)?)
                    .context("expert blob byte offset does not fit usize")?;
                let len = usize::try_from(dtype.byte_len(elem_count as u64)?)
                    .context("expert blob byte length does not fit usize")?;
                let bytes = blob
                    .bytes
                    .get(start..start + len)
                    .ok_or_else(|| anyhow!("expert blob slice is out of range"))?;
                hi_gguf::dequantize_tensor_as_f32(bytes, dtype, elem_count)
            }
        }
    }
}

/// The provider state behind [`DsV4GpuLinear`] (see there).
pub(crate) struct DsV4GpuLinearInner {
    gguf: Arc<GgufFile>,
    stream: Stream,
    cublas: Cublas,
    resident: RefCell<HashMap<String, ResidentEntry>>,
    /// Host-owned packed expert tensors (MTP shard), keyed like GGUF tensors;
    /// [`Self::expert_source`] checks here before the mmap.
    host_expert_blobs: RefCell<HashMap<String, std::rc::Rc<HostExpertBlob>>>,
    /// Stage-1b LRU pool for packed MXFP4 expert slices; `None` when disabled
    /// or the model has no such tensors (everything then streams per call).
    expert_pool: RefCell<Option<DsV4ExpertPool>>,
    /// Copy-stream prefetch (roadmap item 5); `None` unless opted in via
    /// `HI_DSV4_COPY_STREAM=1` / `HI_DSV4_SPEC_PREFETCH=1` (the default and
    /// `HI_DSV4_NO_COPY_STREAM=1` keep the synchronous demand path only).
    copy: Option<RefCell<CopyEngine>>,
    /// Runtime gate for the within-layer prefetch (default on when `copy` is
    /// present). `Cell` so the real-model A/B validation can flip it on one
    /// loaded model without a second multi-minute load; off is output-equivalent
    /// to `HI_DSV4_NO_COPY_STREAM=1` (both leave the demand path synchronous).
    prefetch_enabled: Cell<bool>,
    /// `HI_DSV4_SPEC_PREFETCH=1`: also prefetch the next layer's same-expert-id
    /// slices speculatively (roadmap item 5b). `Cell` for the same A/B reason.
    spec_prefetch: Cell<bool>,
    /// Grow-only scratch for streamed expert weights. MXFP4 slices are ~4.5 MB
    /// — above the runtime buffer pool's recycling cap — so without a held
    /// buffer every expert call would pay a synchronizing cudaFree.
    expert_scratch: RefCell<Option<DeviceBuffer>>,
    /// Grow-only f32 image of one dequantized expert slice (~32 MB on the real
    /// model), reused by the batched-prefill expert GEMM path.
    expert_dequant_scratch: RefCell<Option<DeviceBuffer>>,
    /// Chunked-prefill batching mode (see the module docs): false = bit-exact
    /// per-token loops (default), true = GEMM batching
    /// (`HI_DSV4_PREFILL_GEMM=1`). `Cell` so tests can force a mode without
    /// process-global env races.
    gemm_batching: Cell<bool>,
    /// Scoped [`DsV4Linear::set_exact_batching`] override: while set,
    /// `mul_mat` ignores `gemm_batching` and serves per-token `mul_vec` loops.
    /// The engine pins it around speculative verify chunks (and their rewind
    /// re-feeds), whose logits must be bit-exact with the sequential step
    /// path even under `HI_DSV4_PREFILL_GEMM=1` (which stays prefill-only).
    exact_batching: Cell<bool>,
    /// Wave-2 Stage 2a kill switch (`HI_DSV4_NO_DEVICE_MOE=1`): true routes
    /// every `moe_block` through the host path for bisection. `Cell` so the
    /// real-model A/B parity test can flip it on one loaded model.
    device_moe_disabled: Cell<bool>,
    /// Grow-only device buffers for the device MoE block.
    moe_scratch: RefCell<DeviceMoeScratch>,
    /// Device copies of per-layer `exp_probs_b.bias`, keyed by router tensor
    /// name (unique per layer), uploaded on first touch.
    moe_bias_tables: RefCell<HashMap<String, DeviceBuffer>>,
    /// Device copies of per-layer tid2eid hash tables (I32), keyed by table
    /// tensor name, uploaded on first touch.
    moe_tid2eid_tables: RefCell<HashMap<String, DeviceBuffer>>,
    moe_stats: RefCell<DsV4MoeBlockStats>,
    /// Wave-2 Stage 2b kill switch (`HI_DSV4_HOST_STEP=1`): true declines
    /// every device step so the engine runs the exact pre-Stage-2b host step.
    /// `Cell` so tests can A/B on one loaded model.
    host_step_forced: Cell<bool>,
    /// Device-step resources, built lazily on the first step (the engine's
    /// resident matrices must be uploaded first, which happens at load).
    device_step: RefCell<Option<DeviceStepBuild>>,
    step_stats: RefCell<DsV4StepStats>,
}

impl DsV4GpuLinearInner {
    fn new(gguf: Arc<GgufFile>) -> Result<Self> {
        let stream = Stream::create()?;
        let cublas = Cublas::create()?;
        cublas.set_stream(&stream)?;
        // Copy-stream prefetch is opt-in (`HI_DSV4_COPY_STREAM=1` /
        // `HI_DSV4_SPEC_PREFETCH=1`); the default keeps the synchronous demand
        // path (measured neutral-to-negative — see `copy_stream_enabled`).
        let copy = if copy_stream_enabled() {
            Some(RefCell::new(CopyEngine::create(prefetch_staging_slots())?))
        } else {
            None
        };
        Ok(Self {
            gguf,
            stream,
            cublas,
            resident: RefCell::new(HashMap::new()),
            host_expert_blobs: RefCell::new(HashMap::new()),
            expert_pool: RefCell::new(None),
            copy,
            prefetch_enabled: Cell::new(true),
            spec_prefetch: Cell::new(spec_prefetch_from_env()),
            expert_scratch: RefCell::new(None),
            expert_dequant_scratch: RefCell::new(None),
            gemm_batching: Cell::new(prefill_gemm_from_env()),
            exact_batching: Cell::new(false),
            device_moe_disabled: Cell::new(device_moe_disabled_from_env()),
            moe_scratch: RefCell::new(DeviceMoeScratch::default()),
            moe_bias_tables: RefCell::new(HashMap::new()),
            moe_tid2eid_tables: RefCell::new(HashMap::new()),
            moe_stats: RefCell::new(DsV4MoeBlockStats::default()),
            host_step_forced: Cell::new(host_step_forced_from_env()),
            device_step: RefCell::new(None),
            step_stats: RefCell::new(DsV4StepStats::default()),
        })
    }

    /// Device decode-step counters (see [`DsV4StepStats`]).
    pub(crate) fn step_stats(&self) -> DsV4StepStats {
        *self.step_stats.borrow()
    }

    /// Force the device decode step on/off (tests only; production reads
    /// `HI_DSV4_HOST_STEP` once at construction). Off runs the exact host
    /// step — the Stage-2b parity gate's A/B side.
    #[cfg(test)]
    pub(crate) fn set_device_step_enabled(&self, enabled: bool) {
        self.host_step_forced.set(!enabled);
    }

    /// Device MoE-block counters (see [`DsV4MoeBlockStats`]).
    pub(crate) fn moe_block_stats(&self) -> DsV4MoeBlockStats {
        *self.moe_stats.borrow()
    }

    /// Force the device MoE block on/off (tests only; production reads
    /// `HI_DSV4_NO_DEVICE_MOE` once at construction). Off is the exact host
    /// path — the A/B side of the Stage-2a parity gate.
    #[cfg(test)]
    pub(crate) fn set_device_moe_enabled(&self, enabled: bool) {
        self.device_moe_disabled.set(!enabled);
    }

    /// Force a chunked-prefill batching mode (tests only; production reads
    /// `HI_DSV4_PREFILL_GEMM` once at construction).
    #[cfg(test)]
    pub(crate) fn set_gemm_batching(&self, enabled: bool) {
        self.gemm_batching.set(enabled);
    }

    /// Force the copy-stream prefetch on/off (tests only; production reads
    /// `HI_DSV4_NO_COPY_STREAM` once at construction). Lets a test compare the
    /// async and synchronous paths deterministically without a process-global
    /// env race.
    #[cfg(test)]
    pub(crate) fn set_copy_stream_enabled(&mut self, enabled: bool) -> Result<()> {
        self.copy = if enabled {
            Some(RefCell::new(CopyEngine::create(prefetch_staging_slots())?))
        } else {
            None
        };
        Ok(())
    }

    /// Force speculative next-layer prefetch on/off (tests only).
    #[cfg(test)]
    pub(crate) fn set_spec_prefetch(&self, enabled: bool) {
        self.spec_prefetch.set(enabled);
    }

    /// Toggle the within-layer prefetch at runtime (tests only): off is
    /// output-equivalent to `HI_DSV4_NO_COPY_STREAM=1`. Lets the real-model A/B
    /// validation compare both paths on a single (multi-minute) model load.
    /// Only effective when the copy engine exists (opt-in via `HI_DSV4_COPY_STREAM=1`).
    #[cfg(test)]
    pub(crate) fn set_prefetch_enabled(&self, enabled: bool) {
        self.prefetch_enabled.set(enabled);
    }

    /// Whether the opt-in copy stream was built for this provider (tests only).
    #[cfg(test)]
    pub(crate) fn copy_stream_present(&self) -> bool {
        self.copy.is_some()
    }

    /// Upload every listed matrix (skipping duplicates — the tied lm head
    /// aliases `token_embd.weight`). Called once right after engine load.
    pub(crate) fn upload_resident(&self, specs: &[(RawMatrix, Option<usize>)]) -> Result<()> {
        let mut resident = self.resident.borrow_mut();
        for (matrix, grouped_rank) in specs {
            if resident.contains_key(&matrix.name) {
                continue;
            }
            let entry = match grouped_rank {
                None => ResidentEntry::Dense(self.upload_matrix(matrix)?),
                Some(rank) => ResidentEntry::Grouped(self.upload_grouped(matrix, *rank)?),
            };
            resident.insert(matrix.name.clone(), entry);
        }
        Ok(())
    }

    /// Env-driven expert-pool setup, called once at load after the resident
    /// upload: budget from `HI_DSV4_EXPERT_POOL_GB` (auto-sized from free VRAM
    /// when unset), optional layer-major prefill via
    /// `HI_DSV4_EXPERT_PREFILL_POOL=1`.
    pub(crate) fn init_expert_pool(&self) -> Result<()> {
        if self.scan_packed_expert_tensors().is_empty() {
            // No packed MXFP4 experts (float fixtures, exotic quants): stay on
            // the streaming path without touching env or VRAM accounting.
            return Ok(());
        }
        let budget = match expert_pool_budget_env()? {
            Some(bytes) => bytes,
            None => crate::runtime::free_memory_bytes()
                .context("measuring free VRAM for the expert pool budget")?
                .saturating_sub(DSV4_POOL_HEADROOM_BYTES)
                .min(DSV4_POOL_DEFAULT_MAX_BYTES),
        };
        self.init_expert_pool_with_budget(budget)?;
        if std::env::var("HI_DSV4_EXPERT_PREFILL_POOL").ok().as_deref() == Some("1") {
            self.prefill_expert_pool()?;
        }
        Ok(())
    }

    /// Build the LRU pool over the model's packed MXFP4 expert tensors with
    /// `budget_bytes` of device memory. No-op when the model has none; logs
    /// and stays disabled when the budget holds fewer than 2 slots.
    fn init_expert_pool_with_budget(&self, budget_bytes: usize) -> Result<()> {
        let tensors = self.scan_packed_expert_tensors();
        if tensors.is_empty() {
            return Ok(());
        }
        let slot_bytes = tensors
            .iter()
            .map(|(_, meta)| meta.bytes_per_expert)
            .max()
            .expect("expert tensor scan is non-empty");
        if tensors
            .iter()
            .any(|(_, meta)| meta.bytes_per_expert != slot_bytes)
        {
            // Unexpected: real DeepSeek-V4 gate/up and down slices carry equal
            // element counts. Mixed sizes still work — smaller slices simply
            // under-fill a maximum-size slot.
            eprintln!(
                "dsv4 expert pool: expert slice sizes differ; slots use the maximum {slot_bytes} bytes"
            );
        }
        let total_slices: usize = tensors.iter().map(|(_, meta)| meta.expert_count).sum();
        let slot_stride = slot_bytes.div_ceil(DSV4_POOL_SLOT_ALIGN) * DSV4_POOL_SLOT_ALIGN;
        let slot_count = (budget_bytes / slot_stride)
            .min(total_slices)
            .min(POOL_NIL as usize - 1);
        if slot_count < 2 {
            eprintln!(
                "dsv4 expert pool disabled: budget {budget_bytes} bytes holds {slot_count} slot(s) of {slot_stride} bytes (need >= 2); experts stream per call"
            );
            return Ok(());
        }
        let slots_per_chunk = (DSV4_POOL_CHUNK_TARGET_BYTES / slot_stride).clamp(1, slot_count);
        let mut chunks = Vec::with_capacity(slot_count.div_ceil(slots_per_chunk));
        let mut remaining = slot_count;
        while remaining > 0 {
            let chunk_slots = remaining.min(slots_per_chunk);
            let bytes = chunk_slots
                .checked_mul(slot_stride)
                .context("expert pool chunk byte count overflows usize")?;
            chunks.push(DeviceBuffer::alloc(bytes).context("allocating expert pool chunk")?);
            remaining -= chunk_slots;
        }
        let slots = (0..slot_count)
            .map(|idx| PoolSlot {
                key: None,
                prev: if idx == 0 { POOL_NIL } else { idx as u32 - 1 },
                next: if idx + 1 == slot_count {
                    POOL_NIL
                } else {
                    idx as u32 + 1
                },
                pinned: false,
                spec: false,
                permanent: false,
            })
            .collect();
        eprintln!(
            "dsv4 expert pool: {slot_count} slots x {:.2} MiB = {:.1} GiB ({:.0}% of {total_slices} expert slices resident at capacity)",
            slot_bytes as f64 / (1u64 << 20) as f64,
            gib((slot_count * slot_stride) as u64),
            100.0 * slot_count as f64 / total_slices as f64,
        );
        *self.expert_pool.borrow_mut() = Some(DsV4ExpertPool {
            chunks,
            slots_per_chunk,
            slot_stride,
            slots,
            lru_head: 0,
            lru_tail: slot_count as u32 - 1,
            pinned_slots: Vec::new(),
            resident: HashMap::new(),
            tensors: tensors.into_iter().collect(),
            stats: DsV4ExpertPoolStats::default(),
        });
        Ok(())
    }

    /// Every packed rank-3 MXFP4 expert tensor eligible for the pool, with its
    /// parsed (layer, projection) identity.
    fn scan_packed_expert_tensors(&self) -> Vec<(String, ExpertTensorMeta)> {
        self.gguf
            .tensors()
            .iter()
            .filter_map(|info| Some((info.name.clone(), packed_expert_meta(info)?)))
            .collect()
    }

    /// Fill the pool with experts in (layer, projection, expert) order until
    /// every slot is loaded, so early decode steps hit instead of paying the
    /// first-touch upload. Counted as `prefilled`, not misses.
    fn prefill_expert_pool(&self) -> Result<()> {
        let mut guard = self.expert_pool.borrow_mut();
        let Some(pool) = guard.as_mut() else {
            return Ok(());
        };
        let mut ordered: Vec<(String, ExpertTensorMeta)> = pool
            .tensors
            .iter()
            .map(|(name, meta)| (name.clone(), *meta))
            .collect();
        ordered.sort_by_key(|(_, meta)| (meta.layer, meta.proj));
        let capacity = pool.slots.len();
        'tensors: for (name, meta) in &ordered {
            let source = self.expert_source(name)?;
            for expert in 0..meta.expert_count {
                if pool.resident.len() == capacity {
                    break 'tensors;
                }
                let start = expert
                    .checked_mul(meta.bytes_per_expert)
                    .context("expert byte offset overflows usize")?;
                let bytes = source
                    .bytes()
                    .get(start..start + meta.bytes_per_expert)
                    .ok_or_else(|| anyhow!("tensor {name} expert slice is out of range"))?;
                let (slot, hit) = pool.acquire((meta.layer, meta.proj, expert as u32));
                if hit {
                    continue; // repeated prefill call; already resident
                }
                let (chunk, offset) = pool.slot_location(slot);
                pool.chunks[chunk].copy_from_host_at(offset, bytes)?;
                pool.stats.prefilled += 1;
                pool.stats.bytes_uploaded += meta.bytes_per_expert as u64;
            }
        }
        eprintln!(
            "dsv4 expert pool: prefilled {} of {capacity} slots ({:.1} GiB uploaded)",
            pool.resident.len(),
            gib(pool.stats.bytes_uploaded),
        );
        Ok(())
    }

    /// Pool counters; `None` when the pool is disabled.
    pub(crate) fn pool_stats(&self) -> Option<DsV4ExpertPoolStats> {
        self.expert_pool.borrow().as_ref().map(|pool| pool.stats)
    }

    fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        self.gguf
            .tensor(name)
            .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))
    }

    /// Resolve a packed expert tensor's byte source: a registered host blob
    /// (MTP shard) first, then the GGUF mmap (trunk layers).
    fn expert_source(&self, name: &str) -> Result<ExpertSource<'_>> {
        if let Some(blob) = self.host_expert_blobs.borrow().get(name) {
            return Ok(ExpertSource::Blob(blob.clone()));
        }
        Ok(ExpertSource::Gguf(self.tensor_view(name)?))
    }

    /// Make one dense matrix GPU-resident from HOST data (the MTP module's
    /// safetensors-sourced weights — there is no GGUF tensor to read). The
    /// name must be new; the engine's own residents were uploaded at load.
    pub(crate) fn register_host_dense(
        &self,
        matrix: &RawMatrix,
        data: &HostDenseData,
    ) -> Result<()> {
        let entry = ResidentEntry::Dense(self.upload_host_matrix(matrix, data, None)?);
        self.insert_resident(&matrix.name, entry)
    }

    /// [`Self::register_host_dense`] for the block-diagonal grouped output
    /// projection: one [rank, cols] buffer per group, split host-side.
    pub(crate) fn register_host_grouped(
        &self,
        matrix: &RawMatrix,
        rank: usize,
        data: &HostDenseData,
    ) -> Result<()> {
        if rank == 0 || !matrix.rows.is_multiple_of(rank) {
            bail!(
                "grouped tensor {} rows {} do not split into rank-{rank} groups",
                matrix.name,
                matrix.rows
            );
        }
        let mut groups = Vec::with_capacity(matrix.rows / rank);
        for group in 0..matrix.rows / rank {
            let rows = group * rank..(group + 1) * rank;
            groups.push(self.upload_host_matrix(matrix, data, Some(rows))?);
        }
        self.insert_resident(&matrix.name, ResidentEntry::Grouped(groups))
    }

    fn insert_resident(&self, name: &str, entry: ResidentEntry) -> Result<()> {
        let mut resident = self.resident.borrow_mut();
        if resident.contains_key(name) {
            bail!("tensor {name} is already GPU-resident");
        }
        resident.insert(name.to_string(), entry);
        Ok(())
    }

    /// Upload a host payload (or a row range of it) as a resident matrix in
    /// its natural GEMM dtype: F32 verbatim (the exact-parity fixture path),
    /// F16 bits verbatim (fp8-dequantized real-shard weights), BF16 bits
    /// verbatim (the shard's bf16 router).
    fn upload_host_matrix(
        &self,
        matrix: &RawMatrix,
        data: &HostDenseData,
        rows: Option<std::ops::Range<usize>>,
    ) -> Result<ResidentMatrix> {
        fn upload<T>(name: &str, values: &[T]) -> Result<DeviceBuffer> {
            let buffer = DeviceBuffer::alloc(std::mem::size_of_val(values))
                .with_context(|| format!("allocating resident buffer for {name}"))?;
            buffer.copy_from_host(values)?;
            Ok(buffer)
        }
        let rows = rows.unwrap_or(0..matrix.rows);
        let row_count = rows.len();
        let range = rows.start * matrix.cols..rows.end * matrix.cols;
        check_host_payload(matrix, data.len())?;
        let (dtype, buffer) = match data {
            HostDenseData::F32(values) => (GemmDType::F32, upload(&matrix.name, &values[range])?),
            HostDenseData::F16(bits) => (GemmDType::F16, upload(&matrix.name, &bits[range])?),
            HostDenseData::Bf16(bits) => (GemmDType::BF16, upload(&matrix.name, &bits[range])?),
        };
        Ok(ResidentMatrix {
            buffer,
            rows: row_count,
            cols: matrix.cols,
            dtype,
        })
    }

    /// Register a host-owned packed expert tensor so the pool and streaming
    /// expert paths serve it exactly like a GGUF tensor of the same name and
    /// layout. MXFP4 payloads with a live pool additionally enter the pool's
    /// managed-tensor set under `layer`/`proj`; `pin_resident` preloads every
    /// slice and marks its slots permanent (never evicted — the MTP drafter
    /// touches them every draft). Returns whether the slices are pool-pinned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn register_host_experts(
        &self,
        experts: &RawExperts,
        expert_count: usize,
        layer: u32,
        proj: u8,
        dtype: GgufTensorType,
        bytes: Vec<u8>,
        pin_resident: bool,
    ) -> Result<bool> {
        let per_expert = experts
            .in_dim
            .checked_mul(experts.out_dim)
            .context("expert slice element count overflows usize")?;
        let expected = usize::try_from(dtype.byte_len((per_expert * expert_count) as u64)?)
            .context("expert blob byte length does not fit usize")?;
        if bytes.len() != expected {
            bail!(
                "expert blob {} has {} bytes; expected {expected}",
                experts.name,
                bytes.len()
            );
        }
        {
            let mut blobs = self.host_expert_blobs.borrow_mut();
            if blobs.contains_key(&experts.name) || self.gguf.tensor_info(&experts.name).is_some() {
                bail!("expert tensor {} is already registered", experts.name);
            }
            blobs.insert(
                experts.name.clone(),
                std::rc::Rc::new(HostExpertBlob { dtype, bytes }),
            );
        }
        if dtype != GgufTensorType::MXFP4 || !experts.in_dim.is_multiple_of(32) {
            // Non-MXFP4 payloads (the F32 fixture) use the streaming path,
            // exactly like trunk fixtures of the same dtype.
            return Ok(false);
        }
        let bytes_per_expert = usize::try_from(dtype.byte_len(per_expert as u64)?)
            .context("expert slice byte length does not fit usize")?;
        let mut guard = self.expert_pool.borrow_mut();
        let Some(pool) = guard.as_mut() else {
            return Ok(false);
        };
        if bytes_per_expert > pool.slot_stride {
            bail!(
                "expert blob {} slice size {bytes_per_expert} exceeds the pool slot stride {}",
                experts.name,
                pool.slot_stride
            );
        }
        pool.tensors.insert(
            experts.name.clone(),
            ExpertTensorMeta {
                layer,
                proj,
                bytes_per_expert,
                expert_count,
            },
        );
        if !pin_resident {
            return Ok(false);
        }
        // Pin only when enough unpinned slots remain for the trunk's working
        // set (a decode layer touches <= 3 * top_k slices; 64 is generous).
        let already_permanent = pool.slots.iter().filter(|slot| slot.permanent).count();
        if pool.slots.len() < already_permanent + expert_count + 64 {
            eprintln!(
                "dsv4 expert pool: not pinning {} ({expert_count} slices; pool has {} slots, {already_permanent} already pinned) — slices stay LRU-managed",
                experts.name,
                pool.slots.len()
            );
            return Ok(false);
        }
        let blob_guard = self.host_expert_blobs.borrow();
        let blob = blob_guard
            .get(&experts.name)
            .expect("blob was just inserted");
        for expert in 0..expert_count {
            let start = expert * bytes_per_expert;
            let slice = &blob.bytes[start..start + bytes_per_expert];
            let (slot, hit) = pool.acquire((layer, proj, expert as u32));
            if !hit {
                let (chunk, offset) = pool.slot_location(slot);
                pool.chunks[chunk].copy_from_host_at(offset, slice)?;
                pool.stats.prefilled += 1;
                pool.stats.bytes_uploaded += bytes_per_expert as u64;
            }
            pool.slots[slot as usize].permanent = true;
        }
        Ok(true)
    }

    /// Make one dense matrix GPU-resident. Float dtypes upload verbatim;
    /// quantized dtypes dequantize once on device to f32 and narrow to f16
    /// (same recipe as gpu.rs's `GpuMatrix::into_f16`); quant dtypes without a
    /// device kernel id fall back to a host dequant and stay f32-resident.
    fn upload_matrix(&self, matrix: &RawMatrix) -> Result<ResidentMatrix> {
        let view = self.tensor_view(&matrix.name)?;
        let elements = matrix
            .rows
            .checked_mul(matrix.cols)
            .context("resident matrix element count overflows usize")?;
        let native = match view.info.dtype {
            GgufTensorType::F32 => Some(GemmDType::F32),
            GgufTensorType::F16 => Some(GemmDType::F16),
            GgufTensorType::BF16 => Some(GemmDType::BF16),
            _ => None,
        };
        if let Some(dtype) = native {
            let buffer = DeviceBuffer::alloc(view.bytes.len())
                .with_context(|| format!("allocating resident buffer for {}", matrix.name))?;
            buffer.copy_from_host(view.bytes)?;
            return Ok(ResidentMatrix {
                buffer,
                rows: matrix.rows,
                cols: matrix.cols,
                dtype,
            });
        }
        if let Some(quant_type) = quant_type_id(view.info.dtype) {
            let packed = DeviceBuffer::alloc(view.bytes.len())
                .with_context(|| format!("allocating packed upload for {}", matrix.name))?;
            packed.copy_from_host(view.bytes)?;
            let f32_scratch = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
                .with_context(|| format!("allocating f32 dequant scratch for {}", matrix.name))?;
            crate::kernels::launch_dequantize_matrix(
                &packed,
                &f32_scratch,
                elements,
                quant_type,
                &self.stream,
            )
            .with_context(|| format!("dequantizing {} on device", matrix.name))?;
            let f16 = DeviceBuffer::alloc(elements * std::mem::size_of::<u16>())
                .with_context(|| format!("allocating resident f16 buffer for {}", matrix.name))?;
            crate::kernels::launch_cast_f32_to_f16(&f32_scratch, &f16, elements, &self.stream)?;
            // Finish before packed/f32_scratch free on drop.
            self.stream.synchronize()?;
            return Ok(ResidentMatrix {
                buffer: f16,
                rows: matrix.rows,
                cols: matrix.cols,
                dtype: GemmDType::F16,
            });
        }
        // No device dequant id (exotic dtype): dequantize on host and keep the
        // f32 copy resident. Costs 2x the f16 VRAM but stays exact and only
        // triggers for dtypes the real checkpoint does not use.
        let data = dequantize_elem_range(&view, 0, elements)?;
        let buffer = DeviceBuffer::alloc(elements * std::mem::size_of::<f32>())
            .with_context(|| format!("allocating resident f32 buffer for {}", matrix.name))?;
        buffer.copy_from_host(&data)?;
        Ok(ResidentMatrix {
            buffer,
            rows: matrix.rows,
            cols: matrix.cols,
            dtype: GemmDType::F32,
        })
    }

    /// Upload a block-diagonal matrix as per-group [rank, cols] buffers:
    /// upload/dequantize the whole tensor once, then carve out each group's
    /// contiguous row range with a device-to-device copy.
    fn upload_grouped(&self, matrix: &RawMatrix, rank: usize) -> Result<Vec<ResidentMatrix>> {
        if rank == 0 || !matrix.rows.is_multiple_of(rank) {
            bail!(
                "grouped tensor {} rows {} do not split into rank-{rank} groups",
                matrix.name,
                matrix.rows
            );
        }
        let whole = self.upload_matrix(matrix)?;
        let group_bytes = rank
            .checked_mul(matrix.cols)
            .and_then(|elements| elements.checked_mul(gemm_element_size(whole.dtype)))
            .context("grouped slice byte length overflows usize")?;
        let mut groups = Vec::with_capacity(matrix.rows / rank);
        for group in 0..matrix.rows / rank {
            let buffer = DeviceBuffer::alloc(group_bytes)
                .with_context(|| format!("allocating group buffer for {}", matrix.name))?;
            buffer.copy_device_range(
                0,
                &whole.buffer,
                group * group_bytes,
                group_bytes,
                &self.stream,
            )?;
            groups.push(ResidentMatrix {
                buffer,
                rows: rank,
                cols: matrix.cols,
                dtype: whole.dtype,
            });
        }
        // Finish the carve-outs before the whole-matrix buffer frees on drop.
        self.stream.synchronize()?;
        Ok(groups)
    }

    /// y[rows] = W[rows, cols] · x[cols] against a resident matrix: upload the
    /// f32 activation, cast it to the weight dtype if needed, GEMV with f32
    /// accumulation, download. Small transient buffers recycle through the
    /// runtime's thread-local pool.
    fn resident_gemv(&self, name: &str, matrix: &ResidentMatrix, x: &[f32]) -> Result<Vec<f32>> {
        self.resident_gemm(name, matrix, x, 1)
    }

    /// Row-major Y[m, rows] = X[m, cols] · W[rows, cols]^T against a resident
    /// matrix — one cuBLAS GEMM per call, f32 accumulation. `m = 1` is the
    /// decode GEMV (identical cuBLAS invocation as before batching); the
    /// chunked prefill passes the whole chunk.
    fn resident_gemm(
        &self,
        name: &str,
        matrix: &ResidentMatrix,
        xs: &[f32],
        m: usize,
    ) -> Result<Vec<f32>> {
        if m == 0 || xs.len() != m * matrix.cols {
            bail!(
                "matmul input length {} does not match {m} rows of tensor {name} input dim {}",
                xs.len(),
                matrix.cols
            );
        }
        let input = DeviceBuffer::alloc(std::mem::size_of_val(xs))
            .with_context(|| format!("allocating GEMM input for {name}"))?;
        input.copy_from_host(xs)?;
        let output = DeviceBuffer::alloc(m * matrix.rows * std::mem::size_of::<f32>())
            .with_context(|| format!("allocating GEMM output for {name}"))?;
        match matrix.dtype {
            GemmDType::F32 => self.cublas.matmul_f32_rhs_transposed_row_major(
                &input,
                &matrix.buffer,
                &output,
                m,
                matrix.rows,
                matrix.cols,
            )?,
            dtype => {
                let cast = DeviceBuffer::alloc(xs.len() * std::mem::size_of::<u16>())
                    .with_context(|| format!("allocating GEMM input cast for {name}"))?;
                match dtype {
                    GemmDType::F16 => crate::kernels::launch_cast_f32_to_f16(
                        &input,
                        &cast,
                        xs.len(),
                        &self.stream,
                    )?,
                    GemmDType::BF16 => crate::kernels::launch_cast_f32_to_bf16(
                        &input,
                        &cast,
                        xs.len(),
                        &self.stream,
                    )?,
                    GemmDType::F32 => unreachable!("F32 GEMM returned before the cast path"),
                }
                self.cublas.matmul_mixed_rhs_transposed_row_major(
                    &cast,
                    &matrix.buffer,
                    &output,
                    m,
                    matrix.rows,
                    matrix.cols,
                    dtype,
                    dtype,
                )?;
            }
        }
        // Synchronous D2H: legacy default-stream semantics order it after the
        // enqueued GEMM (same pattern as gpu.rs's copy_to_host).
        output.copy_to_host(m * matrix.rows)
    }

    fn dense_mul_vec(&self, matrix: &RawMatrix, x: &[f32]) -> Result<Vec<f32>> {
        let resident = self.resident.borrow();
        match resident.get(&matrix.name) {
            Some(ResidentEntry::Dense(entry)) => self.resident_gemv(&matrix.name, entry, x),
            Some(ResidentEntry::Grouped(_)) => bail!(
                "tensor {} is resident as a grouped matrix but was used densely",
                matrix.name
            ),
            None => bail!(
                "tensor {} is not GPU-resident; the engine's resident_matrices list is incomplete",
                matrix.name
            ),
        }
    }

    fn grouped_mul_vec(&self, matrix: &RawMatrix, rank: usize, x: &[f32]) -> Result<Vec<f32>> {
        let resident = self.resident.borrow();
        let Some(ResidentEntry::Grouped(groups)) = resident.get(&matrix.name) else {
            bail!(
                "tensor {} is not GPU-resident as a grouped matrix",
                matrix.name
            );
        };
        let group_features = matrix.cols;
        if groups.len() * rank != matrix.rows || x.len() != groups.len() * group_features {
            bail!(
                "grouped matvec input length {} does not match {} groups of {group_features} for {}",
                x.len(),
                groups.len(),
                matrix.name
            );
        }
        // One GEMV per group over its input slice; concatenated outputs land in
        // group order, exactly matching the CPU block-diagonal loop.
        let mut out = Vec::with_capacity(matrix.rows);
        for (group, entry) in groups.iter().enumerate() {
            let x_group = &x[group * group_features..(group + 1) * group_features];
            out.extend(self.resident_gemv(&matrix.name, entry, x_group)?);
        }
        Ok(out)
    }

    /// Chunked-prefill batching for a resident dense matrix: one cuBLAS GEMM
    /// over the whole chunk.
    fn dense_mul_mat(&self, matrix: &RawMatrix, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let resident = self.resident.borrow();
        let entry = match resident.get(&matrix.name) {
            Some(ResidentEntry::Dense(entry)) => entry,
            Some(ResidentEntry::Grouped(_)) => bail!(
                "tensor {} is resident as a grouped matrix but was used densely",
                matrix.name
            ),
            None => bail!(
                "tensor {} is not GPU-resident; the engine's resident_matrices list is incomplete",
                matrix.name
            ),
        };
        let flat = flatten_activations(&matrix.name, xs, matrix.cols)?;
        let out = self.resident_gemm(&matrix.name, entry, &flat, xs.len())?;
        Ok(split_rows(out, matrix.rows))
    }

    /// Chunked-prefill batching for the block-diagonal output projection: one
    /// GEMM per group over the chunk's input slices; per-token outputs
    /// concatenate in group order, exactly matching the CPU loop.
    fn grouped_mul_mat(
        &self,
        matrix: &RawMatrix,
        rank: usize,
        xs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let resident = self.resident.borrow();
        let Some(ResidentEntry::Grouped(groups)) = resident.get(&matrix.name) else {
            bail!(
                "tensor {} is not GPU-resident as a grouped matrix",
                matrix.name
            );
        };
        let group_features = matrix.cols;
        if groups.len() * rank != matrix.rows
            || xs.iter().any(|x| x.len() != groups.len() * group_features)
        {
            bail!(
                "grouped matmul input rows do not all match {} groups of {group_features} for {}",
                groups.len(),
                matrix.name
            );
        }
        let m = xs.len();
        let mut out = vec![Vec::with_capacity(matrix.rows); m];
        let mut flat = Vec::with_capacity(m * group_features);
        for (group, entry) in groups.iter().enumerate() {
            flat.clear();
            for x in xs {
                flat.extend_from_slice(&x[group * group_features..(group + 1) * group_features]);
            }
            let group_out = self.resident_gemm(&matrix.name, entry, &flat, m)?;
            for (token_out, chunk) in out.iter_mut().zip(group_out.chunks(rank)) {
                token_out.extend_from_slice(chunk);
            }
        }
        Ok(out)
    }

    /// Expert path: slice the expert's packed bytes out of the host mmap and
    /// multiply on device. MXFP4 runs the fused GEMV against the LRU pool slot
    /// (Stage 1b), streaming through the grow-only scratch when the pool is
    /// disabled; anything else (the F32 test fixture, exotic quants)
    /// host-dequantizes the slice and runs the f32 GEMV.
    fn expert_mul_vec(&self, experts: &RawExperts, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != experts.in_dim {
            bail!(
                "expert matvec input length {} does not match tensor {} input dim {}",
                x.len(),
                experts.name,
                experts.in_dim
            );
        }
        let source = self.expert_source(&experts.name)?;
        let per_expert = experts
            .in_dim
            .checked_mul(experts.out_dim)
            .context("expert slice element count overflows usize")?;

        let input = DeviceBuffer::alloc(std::mem::size_of_val(x))
            .context("allocating expert GEMV input")?;
        input.copy_from_host(x)?;
        let output = DeviceBuffer::alloc(experts.out_dim * std::mem::size_of::<f32>())
            .context("allocating expert GEMV output")?;

        if source.dtype() == GgufTensorType::MXFP4 && experts.in_dim.is_multiple_of(32) {
            // Contiguous packed range of expert e: rank-3 [in, out, experts]
            // stores expert-major, so the byte stride is in*out/32*17.
            let dtype = source.dtype();
            let offset = dtype.byte_len((expert * per_expert) as u64)? as usize;
            let len = dtype.byte_len(per_expert as u64)? as usize;
            let bytes = source
                .bytes()
                .get(offset..offset + len)
                .ok_or_else(|| anyhow!("tensor {} expert slice is out of range", experts.name))?;
            if !self.pool_expert_gemv(experts, expert, bytes, &input, &output)? {
                // Stage-1a fallback: pool disabled or unmanaged tensor.
                self.with_expert_scratch(len, |weights| {
                    weights.copy_from_host(bytes)?;
                    crate::kernels::launch_mxfp4_gemv(
                        weights,
                        &input,
                        &output,
                        experts.out_dim,
                        experts.in_dim,
                        &self.stream,
                    )
                })?;
            }
        } else {
            let data = source.dequant_range(expert * per_expert, per_expert)?;
            self.with_expert_scratch(per_expert * std::mem::size_of::<f32>(), |weights| {
                weights.copy_from_host(&data)?;
                self.cublas.matmul_f32_rhs_transposed_row_major(
                    &input,
                    weights,
                    &output,
                    1,
                    experts.out_dim,
                    experts.in_dim,
                )
            })?;
        }
        output.copy_to_host(experts.out_dim)
    }

    /// Batched expert path for the chunked prefill: dequantize the expert's
    /// packed MXFP4 slice once on device (straight out of its pooled slot
    /// when resident — pool LRU accounting is identical to the GEMV path)
    /// into the grow-only f32 scratch, then run one GEMM over every token the
    /// chunk routed to this expert. Small batches and non-MXFP4 tensors keep
    /// the per-token fused-GEMV path, which also remains the decode path.
    fn expert_mul_mat(
        &self,
        experts: &RawExperts,
        expert: usize,
        xs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let source = self.expert_source(&experts.name)?;
        if xs.len() < DSV4_EXPERT_GEMM_MIN_TOKENS
            || source.dtype() != GgufTensorType::MXFP4
            || !experts.in_dim.is_multiple_of(32)
        {
            return xs
                .iter()
                .map(|x| self.expert_mul_vec(experts, expert, x))
                .collect();
        }
        let m = xs.len();
        let flat = flatten_activations(&experts.name, xs, experts.in_dim)?;
        let per_expert = experts
            .in_dim
            .checked_mul(experts.out_dim)
            .context("expert slice element count overflows usize")?;
        let dtype = source.dtype();
        let byte_offset = dtype.byte_len((expert * per_expert) as u64)? as usize;
        let byte_len = dtype.byte_len(per_expert as u64)? as usize;
        let bytes = source
            .bytes()
            .get(byte_offset..byte_offset + byte_len)
            .ok_or_else(|| anyhow!("tensor {} expert slice is out of range", experts.name))?;
        let quant_type = quant_type_id(dtype).expect("MXFP4 has a device dequant kernel id");

        let input = DeviceBuffer::alloc(std::mem::size_of_val(flat.as_slice()))
            .context("allocating expert GEMM input")?;
        input.copy_from_host(&flat)?;
        let output = DeviceBuffer::alloc(m * experts.out_dim * std::mem::size_of::<f32>())
            .context("allocating expert GEMM output")?;

        {
            let mut slot = self.expert_dequant_scratch.borrow_mut();
            let min_bytes = per_expert * std::mem::size_of::<f32>();
            if slot
                .as_ref()
                .is_none_or(|buffer| buffer.bytes() < min_bytes)
            {
                *slot = Some(
                    DeviceBuffer::alloc(min_bytes).context("allocating expert dequant scratch")?,
                );
            }
        }
        let scratch_guard = self.expert_dequant_scratch.borrow();
        let scratch = scratch_guard
            .as_ref()
            .expect("expert dequant scratch was just populated");

        // Dequantize out of the pooled slot when this tensor is pool-managed;
        // otherwise upload the packed bytes into the streaming scratch first.
        let mut pool_guard = self.expert_pool.borrow_mut();
        let pooled = pool_guard.as_mut().and_then(|pool| {
            let meta = pool.tensors.get(&experts.name).copied()?;
            (bytes.len() == meta.bytes_per_expert && expert < meta.expert_count)
                .then_some((pool, meta))
        });
        match pooled {
            Some((pool, meta)) => {
                let (slot, hit) = pool.acquire((meta.layer, meta.proj, expert as u32));
                let (chunk, slot_offset) = pool.slot_location(slot);
                if hit {
                    pool.stats.hits += 1;
                } else {
                    pool.chunks[chunk].copy_from_host_at(slot_offset, bytes)?;
                    pool.stats.misses += 1;
                    pool.stats.bytes_uploaded += bytes.len() as u64;
                }
                crate::kernels::launch_dequantize_matrix_at(
                    &pool.chunks[chunk],
                    slot_offset,
                    byte_len,
                    scratch,
                    per_expert,
                    quant_type,
                    &self.stream,
                )?;
                if (pool.stats.hits + pool.stats.misses).is_multiple_of(DSV4_POOL_LOG_EVERY_CALLS) {
                    eprintln!("dsv4 expert pool: {}", format_pool_stats(&pool.stats));
                }
            }
            None => {
                self.with_expert_scratch(byte_len, |packed| {
                    packed.copy_from_host(bytes)?;
                    crate::kernels::launch_dequantize_matrix_at(
                        packed,
                        0,
                        byte_len,
                        scratch,
                        per_expert,
                        quant_type,
                        &self.stream,
                    )
                })?;
            }
        }
        drop(pool_guard);

        self.cublas.matmul_f32_rhs_transposed_row_major(
            &input,
            scratch,
            &output,
            m,
            experts.out_dim,
            experts.in_dim,
        )?;
        let out = output.copy_to_host::<f32>(m * experts.out_dim)?;
        Ok(split_rows(out, experts.out_dim))
    }

    /// Serve one MXFP4 expert GEMV from the LRU pool: on a hit launch straight
    /// against the resident slot, on a miss upload the packed slice into the
    /// evicted slot first. Returns false (caller streams instead) when the
    /// pool is disabled or does not manage this tensor.
    fn pool_expert_gemv(
        &self,
        experts: &RawExperts,
        expert: usize,
        bytes: &[u8],
        input: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<bool> {
        let mut guard = self.expert_pool.borrow_mut();
        let Some(pool) = guard.as_mut() else {
            return Ok(false);
        };
        let Some(meta) = pool.tensors.get(&experts.name).copied() else {
            return Ok(false);
        };
        if bytes.len() != meta.bytes_per_expert || expert >= meta.expert_count {
            // Shape drifted from the load-time scan; stream it instead.
            return Ok(false);
        }
        let (slot, hit) = pool.acquire((meta.layer, meta.proj, expert as u32));
        let (chunk, offset) = pool.slot_location(slot);
        if hit {
            pool.stats.hits += 1;
        } else {
            pool.chunks[chunk].copy_from_host_at(offset, bytes)?;
            pool.stats.misses += 1;
            pool.stats.bytes_uploaded += bytes.len() as u64;
        }
        crate::kernels::launch_mxfp4_gemv_at(
            &pool.chunks[chunk],
            offset,
            input,
            output,
            experts.out_dim,
            experts.in_dim,
            &self.stream,
        )?;
        if (pool.stats.hits + pool.stats.misses).is_multiple_of(DSV4_POOL_LOG_EVERY_CALLS) {
            eprintln!("dsv4 expert pool: {}", format_pool_stats(&pool.stats));
        }
        Ok(true)
    }

    /// Run `f` against the grow-only expert weight scratch, (re)allocating when
    /// the requested size exceeds the held capacity. Excess capacity is
    /// harmless: uploads and GEMV reads are bounded by explicit lengths.
    fn with_expert_scratch<T>(
        &self,
        min_bytes: usize,
        f: impl FnOnce(&DeviceBuffer) -> Result<T>,
    ) -> Result<T> {
        let mut slot = self.expert_scratch.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|buffer| buffer.bytes() < min_bytes)
        {
            *slot =
                Some(DeviceBuffer::alloc(min_bytes).context("allocating expert weight scratch")?);
        }
        f(slot.as_ref().expect("expert scratch was just populated"))
    }

    /// Copy-stream prefetch of a MoE layer's expert slices (roadmap item 5),
    /// invoked by the engine after routing is known and before the layer's
    /// expert matmuls. Issues every missing gate/up/down slice for the routed
    /// experts on the dedicated copy stream (staged through pinned host memory)
    /// so the demand GEMVs hit, gated into the engine stream by a copy-done
    /// event; with `HI_DSV4_SPEC_PREFETCH=1` it additionally warms the next
    /// layer's same-expert-id slices speculatively (never displacing a slice the
    /// current batch still needs). A no-op when the copy stream is disabled or
    /// the pool is absent — the synchronous demand path then serves everything,
    /// so the produced logits are identical either way.
    fn prefetch_experts_impl(
        &self,
        tensors: DsV4ExpertTensors<'_>,
        expert_ids: &[usize],
    ) -> Result<()> {
        let Some(copy_cell) = self.copy.as_ref() else {
            return Ok(());
        };
        if expert_ids.is_empty() || !self.prefetch_enabled.get() {
            return Ok(());
        }
        let mut pool_guard = self.expert_pool.borrow_mut();
        let Some(pool) = pool_guard.as_mut() else {
            return Ok(());
        };
        let mut copy = copy_cell.borrow_mut();

        // New batch: release the previous layer's pins, ensure staging, and
        // select/gate this batch's staging generation.
        pool.clear_pins();
        copy.ensure_staging(pool.slot_stride)?;
        copy.begin_batch()?;
        // Fork: prefetch copies wait for any expert reads already enqueued on the
        // engine stream, so a copy never clobbers a slot still being read.
        copy.fork.record(&self.stream)?;
        copy.stream.wait_event(&copy.fork)?;

        // (a) Within-layer: this layer's gate/up/down slices for each routed
        // expert, in the exact order the expert loop consumes them, up to the
        // pinned-staging capacity. Anything skipped falls to the demand path.
        let capacity = copy.staging_slots;
        let mut region = 0usize;
        let mut full = false;
        let this = [
            (tensors.gate.name.as_str(), 0u8),
            (tensors.up.name.as_str(), 1u8),
            (tensors.down.name.as_str(), 2u8),
        ];
        'within: for &expert in expert_ids {
            for (name, proj) in this {
                if region >= capacity {
                    full = true;
                    break 'within;
                }
                match self.prefetch_one(pool, &copy, name, proj, expert, region, false)? {
                    PrefetchStep::Copied => region += 1,
                    PrefetchStep::Skipped => {}
                    PrefetchStep::Full => {
                        full = true;
                        break 'within;
                    }
                }
            }
        }
        // Join: the layer's expert GEMVs (issued next on the engine stream) wait
        // for the within-layer copies. `done` captures ALL prior copy-stream work,
        // so it also covers any speculative slices this layer now consumes.
        copy.done.record(&copy.stream)?;
        self.stream.wait_event(&copy.done)?;

        // (b) Speculative next-layer warm: best-effort, never gated into the
        // engine here (the next layer's own join covers it), so it overlaps this
        // layer's experts and the next layer's attention.
        if self.spec_prefetch.get() && !full {
            if let Some(layer) = pool
                .tensors
                .get(tensors.gate.name.as_str())
                .map(|m| m.layer)
            {
                let next = layer + 1;
                let next_projs = [
                    (format!("blk.{next}.ffn_gate_exps.weight"), 0u8),
                    (format!("blk.{next}.ffn_up_exps.weight"), 1u8),
                    (format!("blk.{next}.ffn_down_exps.weight"), 2u8),
                ];
                'spec: for &expert in expert_ids {
                    for (name, proj) in &next_projs {
                        if region >= capacity {
                            break 'spec;
                        }
                        match self.prefetch_one(pool, &copy, name, *proj, expert, region, true)? {
                            PrefetchStep::Copied => region += 1,
                            PrefetchStep::Skipped => {}
                            PrefetchStep::Full => break 'spec,
                        }
                    }
                }
            }
        }

        // Guard this batch's staging generation for its next reuse.
        copy.end_batch()?;
        Ok(())
    }

    /// Prefetch a single expert slice into its pool slot on the copy stream.
    /// `spec` marks a speculative next-layer load. Returns whether a staging
    /// region was consumed (see [`PrefetchStep`]).
    fn prefetch_one(
        &self,
        pool: &mut DsV4ExpertPool,
        copy: &CopyEngine,
        name: &str,
        proj: u8,
        expert: usize,
        region: usize,
        spec: bool,
    ) -> Result<PrefetchStep> {
        let Some(meta) = pool.tensors.get(name).copied() else {
            return Ok(PrefetchStep::Skipped);
        };
        if expert >= meta.expert_count {
            return Ok(PrefetchStep::Skipped);
        }
        let slot = match pool.acquire_prefetch((meta.layer, proj, expert as u32)) {
            PrefetchAcquire::Resident(_) => return Ok(PrefetchStep::Skipped),
            PrefetchAcquire::Full => return Ok(PrefetchStep::Full),
            PrefetchAcquire::Loaded(slot) => slot,
        };
        let source = self.expert_source(name)?;
        let start = expert
            .checked_mul(meta.bytes_per_expert)
            .context("expert byte offset overflows usize")?;
        let bytes = source
            .bytes()
            .get(start..start + meta.bytes_per_expert)
            .ok_or_else(|| anyhow!("tensor {name} expert slice is out of range"))?;
        let (chunk, offset) = pool.slot_location(slot);
        copy.stage_upload(region, &pool.chunks[chunk], offset, bytes)?;
        pool.stats.bytes_uploaded += bytes.len() as u64;
        if spec {
            pool.slots[slot as usize].spec = true;
            pool.stats.spec_uploads += 1;
        } else {
            pool.stats.prefetch_uploads += 1;
        }
        Ok(PrefetchStep::Copied)
    }

    /// Can the device MoE block serve this layer? Requires the MXFP4 expert
    /// pool to manage all three expert tensors, fused-GEMV-compatible dims
    /// (both matmul widths on the 32-element block grid), and the router +
    /// shared-expert matrices resident as plain dense entries. Anything else
    /// falls back to the host path (the F32 fixture, exotic quants).
    fn device_moe_supported(&self, ctx: &DsV4MoeBlockCtx<'_>) -> bool {
        let ff = ctx.gate.out_dim;
        if ctx.embed == 0
            || ff == 0
            || !ctx.embed.is_multiple_of(32)
            || !ff.is_multiple_of(32)
            || ctx.top_k == 0
            || ctx.top_k > ctx.experts
            || ctx.gate.in_dim != ctx.embed
            || ctx.up.in_dim != ctx.embed
            || ctx.up.out_dim != ff
            || ctx.down.in_dim != ff
            || ctx.down.out_dim != ctx.embed
        {
            return false;
        }
        if let Some(table) = ctx.tid2eid
            && (table.stride != ctx.top_k
                || table.tokens == 0
                || table.values.len() < table.tokens * table.stride)
        {
            return false;
        }
        if let Some(bias) = ctx.probs_bias
            && bias.len() < ctx.experts
        {
            return false;
        }
        {
            let pool_guard = self.expert_pool.borrow();
            let Some(pool) = pool_guard.as_ref() else {
                return false;
            };
            for tensor in [ctx.gate, ctx.up, ctx.down] {
                match pool.tensors.get(&tensor.name) {
                    Some(meta) if meta.expert_count >= ctx.experts => {}
                    _ => return false,
                }
            }
        }
        let resident = self.resident.borrow();
        let dense_ok = |name: &str, rows: usize, cols: usize| {
            matches!(resident.get(name),
                Some(ResidentEntry::Dense(m)) if m.rows == rows && m.cols == cols)
        };
        if !dense_ok(&ctx.router.name, ctx.experts, ctx.embed) {
            return false;
        }
        if let Some(shared) = &ctx.shared {
            let sff = shared.gate.rows;
            if sff == 0
                || !dense_ok(&shared.gate.name, sff, ctx.embed)
                || !dense_ok(&shared.up.name, sff, ctx.embed)
                || !dense_ok(&shared.down.name, ctx.embed, sff)
            {
                return false;
            }
        }
        true
    }

    /// One M=1 GEMV against a resident dense matrix with the activation and
    /// output at byte offsets into flat batch buffers — the IDENTICAL cuBLAS
    /// call shape `resident_gemv` makes (same dims, same dtypes, same
    /// 256-byte-aligned pointers), so per-token results are bit-identical to
    /// the host path's. `input` must already be in the matrix dtype.
    #[allow(clippy::too_many_arguments)]
    fn resident_gemv_at(
        &self,
        matrix: &ResidentMatrix,
        input: &DeviceBuffer,
        input_byte_offset: usize,
        output: &DeviceBuffer,
        output_byte_offset: usize,
    ) -> Result<()> {
        match matrix.dtype {
            GemmDType::F32 => self.cublas.matmul_f32_rhs_transposed_row_major_at(
                input,
                input_byte_offset,
                &matrix.buffer,
                output,
                output_byte_offset,
                1,
                matrix.rows,
                matrix.cols,
            ),
            dtype => self.cublas.matmul_mixed_rhs_transposed_row_major_at(
                input,
                input_byte_offset,
                &matrix.buffer,
                output,
                output_byte_offset,
                1,
                matrix.rows,
                matrix.cols,
                dtype,
                dtype,
            ),
        }
    }

    /// The Wave-2 Stage 2a device MoE block: ONE activation upload, then
    /// [`Self::device_moe_core`], then ONE ys download. Every numeric op
    /// mirrors the host path exactly (per-token M=1 GEMVs, bit-exact glibc
    /// expf/logf ports, explicit rounding in the elementwise kernels), so the
    /// produced activations are bit-identical to `HI_DSV4_NO_DEVICE_MOE=1` —
    /// the Stage-2a parity gate.
    fn device_moe_block(
        &self,
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
        let embed = ctx.embed;
        let f32s = std::mem::size_of::<f32>();
        let mut launches = 0u64;
        let mut syncs = 0u64;

        let mut scratch = self.moe_scratch.borrow_mut();
        let scratch = &mut *scratch;
        scratch.xs_f16_valid = false;
        scratch.xs_bf16_valid = false;

        // ---- ONE H2D upload of the batch activations (async, engine stream).
        let flat = flatten_activations(&ctx.router.name, xs, embed)?;
        ensure_scratch(&mut scratch.xs_f32, b * embed * f32s, "moe xs")?;
        let xs_f32 = scratch
            .xs_f32
            .take()
            .expect("moe xs scratch was just ensured");
        xs_f32.copy_from_host_async(&flat, &self.stream)?;
        launches += 1;

        let resident = self.resident.borrow();
        let outcome = self.device_moe_core(ctx, &xs_f32, b, tokens, scratch, &resident);
        drop(resident);
        scratch.xs_f32 = Some(xs_f32);
        match outcome? {
            MoeCoreOutcome::Done {
                launches: core_launches,
                syncs: core_syncs,
            } => {
                launches += core_launches;
                syncs += core_syncs;
            }
            MoeCoreOutcome::PoolTooSmall {
                launches: core_launches,
                syncs: core_syncs,
            } => {
                let mut stats = self.moe_stats.borrow_mut();
                stats.host_blocks += 1;
                stats.launches += launches + core_launches;
                stats.syncs += syncs + core_syncs;
                drop(stats);
                return host_moe_block(self, ctx, xs, tokens);
            }
        }

        // ---- ONE D2H download of the block outputs.
        let ys = scratch.ys.as_ref().expect("moe core populated ys");
        let out: Vec<f32> = ys.copy_to_host(b * embed)?;
        syncs += 1;

        let mut stats = self.moe_stats.borrow_mut();
        stats.device_blocks += 1;
        stats.launches += launches;
        stats.syncs += syncs;
        Ok(split_rows(out, embed))
    }

    /// The device MoE block's interior (shared by the Stage-2a trait path,
    /// which uploads host activations first, and the Stage-2b device step,
    /// whose activations are already resident): per-token router GEMVs (M=1,
    /// the host path's exact cuBLAS calls), the scoring+selection kernel, ONE
    /// topk_ids readback (the interior sync servicing the expert-pool LRU),
    /// back-to-back fused MXFP4 expert GEMVs chained device-side through the
    /// SwiGLU-clamp and weighted-accumulate kernels, and the shared expert.
    /// Writes the block outputs into `scratch.ys` ([b, embed] f32, device).
    fn device_moe_core(
        &self,
        ctx: &DsV4MoeBlockCtx<'_>,
        xs_f32: &DeviceBuffer,
        b: usize,
        tokens: &[u32],
        scratch: &mut DeviceMoeScratch,
        resident: &HashMap<String, ResidentEntry>,
    ) -> Result<MoeCoreOutcome> {
        let k = ctx.top_k;
        let embed = ctx.embed;
        let experts = ctx.experts;
        let ff = ctx.gate.out_dim;
        let f32s = std::mem::size_of::<f32>();
        let mut launches = 0u64;
        let mut syncs = 0u64;

        let Some(ResidentEntry::Dense(router)) = resident.get(&ctx.router.name) else {
            bail!(
                "tensor {} is not GPU-resident as a dense matrix",
                ctx.router.name
            );
        };

        // ---- Per-token router GEMVs (M=1 each — the host path's exact
        // cuBLAS calls, so logits are bit-identical to the sequential path).
        let logits_buf = ensure_scratch(
            &mut scratch.router_logits,
            b * experts * f32s,
            "moe router logits",
        )?;
        let router_input: &DeviceBuffer = match router.dtype {
            GemmDType::F32 => xs_f32,
            GemmDType::F16 => {
                let cast = ensure_scratch(&mut scratch.xs_f16, b * embed * 2, "moe xs f16")?;
                crate::kernels::launch_cast_f32_to_f16(xs_f32, cast, b * embed, &self.stream)?;
                launches += 1;
                scratch.xs_f16_valid = true;
                cast
            }
            GemmDType::BF16 => {
                let cast = ensure_scratch(&mut scratch.xs_bf16, b * embed * 2, "moe xs bf16")?;
                crate::kernels::launch_cast_f32_to_bf16(xs_f32, cast, b * embed, &self.stream)?;
                launches += 1;
                scratch.xs_bf16_valid = true;
                cast
            }
        };
        let router_elem = gemm_element_size(router.dtype);
        for t in 0..b {
            self.resident_gemv_at(
                router,
                router_input,
                t * embed * router_elem,
                logits_buf,
                t * experts * f32s,
            )?;
        }
        launches += b as u64;

        // ---- Device scoring + selection (hash layers gather tid2eid rows).
        // Bias and hash tables upload once per layer, on first touch.
        if let Some(bias) = ctx.probs_bias
            && !self.moe_bias_tables.borrow().contains_key(&ctx.router.name)
        {
            let buffer = DeviceBuffer::alloc(experts * f32s)
                .context("allocating device exp_probs_b bias")?;
            buffer.copy_from_host(&bias[..experts])?;
            syncs += 1;
            self.moe_bias_tables
                .borrow_mut()
                .insert(ctx.router.name.clone(), buffer);
        }
        if let Some(table) = ctx.tid2eid
            && !self.moe_tid2eid_tables.borrow().contains_key(&table.name)
        {
            let buffer = DeviceBuffer::alloc(table.values.len() * 4)
                .context("allocating device tid2eid table")?;
            buffer.copy_from_host(&table.values)?;
            syncs += 1;
            self.moe_tid2eid_tables
                .borrow_mut()
                .insert(table.name.clone(), buffer);
        }
        let bias_tables = self.moe_bias_tables.borrow();
        let bias_buf = ctx.probs_bias.map(|_| {
            bias_tables
                .get(&ctx.router.name)
                .expect("bias just uploaded")
        });
        let tid2eid_tables = self.moe_tid2eid_tables.borrow();
        let table_buf = ctx.tid2eid.map(|table| {
            (
                tid2eid_tables
                    .get(&table.name)
                    .expect("table just uploaded"),
                table.tokens,
            )
        });
        let token_ids_buf = if ctx.tid2eid.is_some() {
            let ids: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
            let buffer = ensure_scratch(&mut scratch.token_ids, b * 4, "moe token ids")?;
            buffer.copy_from_host_async(&ids, &self.stream)?;
            launches += 1;
            Some(&*buffer)
        } else {
            None
        };
        let ids_buf = ensure_scratch(&mut scratch.topk_ids, b * k * 4, "moe topk ids")?;
        let weights_buf =
            ensure_scratch(&mut scratch.topk_weights, b * k * f32s, "moe topk weights")?;
        crate::kernels::launch_dsv4_moe_select(
            logits_buf,
            bias_buf,
            table_buf,
            token_ids_buf,
            ids_buf,
            weights_buf,
            b,
            experts,
            k,
            ctx.weights_norm,
            ctx.weights_scale,
            &self.stream,
        )?;
        launches += 1;

        // ---- THE interior sync: 4*top_k bytes/token of selected expert ids,
        // needed on host to service the pool LRU (miss uploads).
        let ids_host: Vec<i32> = ids_buf.copy_to_host(b * k)?;
        syncs += 1;
        let mut unique: Vec<usize> = Vec::new();
        for &id in &ids_host {
            let expert = usize::try_from(id)
                .ok()
                .filter(|&expert| expert < experts)
                .ok_or_else(|| anyhow!("device MoE selection produced invalid expert id {id}"))?;
            unique.push(expert);
        }
        unique.sort_unstable();
        unique.dedup();

        // The batched resolve-then-launch below requires every routed slice
        // resident SIMULTANEOUSLY (the host path acquires and launches one
        // GEMV at a time, so it tolerates same-batch eviction; this path does
        // not). A pool smaller than the batch's slice set — pathological
        // budgets only; the real model needs <= 768 of ~16k slots — makes the
        // caller fall back to the host path for this call.
        {
            let pool_guard = self.expert_pool.borrow();
            let pool = pool_guard
                .as_ref()
                .ok_or_else(|| anyhow!("device MoE block requires the expert pool"))?;
            if pool.slots.len() < 3 * unique.len() {
                return Ok(MoeCoreOutcome::PoolTooSmall { launches, syncs });
            }
        }

        // Optional copy-stream prefetch (roadmap item 5) warms the slices the
        // acquires below will use; without it misses upload synchronously,
        // exactly like the host path's demand GEMVs.
        self.prefetch_experts_impl(
            DsV4ExpertTensors {
                gate: ctx.gate,
                up: ctx.up,
                down: ctx.down,
            },
            &unique,
        )?;

        let mut pool_guard = self.expert_pool.borrow_mut();
        let pool = pool_guard
            .as_mut()
            .ok_or_else(|| anyhow!("device MoE block requires the expert pool"))?;
        let served_before = pool.stats.hits + pool.stats.misses;
        let mut slots: HashMap<(u8, usize), (usize, usize)> = HashMap::new();
        for (proj, tensor) in [(0u8, ctx.gate), (1u8, ctx.up), (2u8, ctx.down)] {
            let meta = *pool
                .tensors
                .get(&tensor.name)
                .ok_or_else(|| anyhow!("tensor {} left the expert pool", tensor.name))?;
            let source = self.expert_source(&tensor.name)?;
            for &expert in &unique {
                let (slot, hit) = pool.acquire((meta.layer, meta.proj, expert as u32));
                let location = pool.slot_location(slot);
                if hit {
                    pool.stats.hits += 1;
                } else {
                    let start = expert
                        .checked_mul(meta.bytes_per_expert)
                        .context("expert byte offset overflows usize")?;
                    let bytes = source
                        .bytes()
                        .get(start..start + meta.bytes_per_expert)
                        .ok_or_else(|| {
                            anyhow!("tensor {} expert slice is out of range", tensor.name)
                        })?;
                    pool.chunks[location.0].copy_from_host_at(location.1, bytes)?;
                    pool.stats.misses += 1;
                    pool.stats.bytes_uploaded += meta.bytes_per_expert as u64;
                    syncs += 1;
                }
                slots.insert((proj, expert), location);
            }
        }
        let served_after = pool.stats.hits + pool.stats.misses;
        if served_before / DSV4_POOL_LOG_EVERY_CALLS != served_after / DSV4_POOL_LOG_EVERY_CALLS {
            eprintln!("dsv4 expert pool: {}", format_pool_stats(&pool.stats));
        }

        // ---- Routed experts: per-(token, rank) fused MXFP4 GEMVs, launched
        // back-to-back with NO intervening syncs, chained device-side through
        // the SwiGLU-clamp kernel. Slot (t, j) lives at flat index t*k+j.
        let gate_buf = ensure_scratch(&mut scratch.gate, b * k * ff * f32s, "moe gate")?;
        let up_buf = ensure_scratch(&mut scratch.up, b * k * ff * f32s, "moe up")?;
        let expert_out =
            ensure_scratch(&mut scratch.expert_out, b * k * embed * f32s, "moe outputs")?;
        for t in 0..b {
            for j in 0..k {
                let slot = t * k + j;
                let expert = ids_host[slot] as usize;
                let (chunk, offset) = slots[&(0, expert)];
                crate::kernels::launch_mxfp4_gemv_slice(
                    &pool.chunks[chunk],
                    offset,
                    xs_f32,
                    t * embed * f32s,
                    gate_buf,
                    slot * ff * f32s,
                    ff,
                    embed,
                    &self.stream,
                )?;
                let (chunk, offset) = slots[&(1, expert)];
                crate::kernels::launch_mxfp4_gemv_slice(
                    &pool.chunks[chunk],
                    offset,
                    xs_f32,
                    t * embed * f32s,
                    up_buf,
                    slot * ff * f32s,
                    ff,
                    embed,
                    &self.stream,
                )?;
            }
        }
        launches += 2 * (b * k) as u64;
        crate::kernels::launch_dsv4_swiglu_clamp(
            gate_buf,
            up_buf,
            gate_buf,
            b * k * ff,
            ctx.swiglu_clamp,
            &self.stream,
        )?;
        launches += 1;
        for t in 0..b {
            for j in 0..k {
                let slot = t * k + j;
                let expert = ids_host[slot] as usize;
                let (chunk, offset) = slots[&(2, expert)];
                crate::kernels::launch_mxfp4_gemv_slice(
                    &pool.chunks[chunk],
                    offset,
                    gate_buf,
                    slot * ff * f32s,
                    expert_out,
                    slot * embed * f32s,
                    embed,
                    ff,
                    &self.stream,
                )?;
            }
        }
        launches += (b * k) as u64;
        drop(pool_guard);

        // ---- Shared expert (resident dense, plain SwiGLU), still no syncs.
        let shared_out: Option<&DeviceBuffer> = if let Some(shared) = &ctx.shared {
            let Some(ResidentEntry::Dense(gate_mat)) = resident.get(&shared.gate.name) else {
                bail!("tensor {} is not GPU-resident", shared.gate.name);
            };
            let Some(ResidentEntry::Dense(up_mat)) = resident.get(&shared.up.name) else {
                bail!("tensor {} is not GPU-resident", shared.up.name);
            };
            let Some(ResidentEntry::Dense(down_mat)) = resident.get(&shared.down.name) else {
                bail!("tensor {} is not GPU-resident", shared.down.name);
            };
            let sff = gate_mat.rows;
            let sh_gate =
                ensure_scratch(&mut scratch.shared_gate, b * sff * f32s, "moe shexp gate")?;
            let sh_up = ensure_scratch(&mut scratch.shared_up, b * sff * f32s, "moe shexp up")?;
            let sh_out =
                ensure_scratch(&mut scratch.shared_out, b * embed * f32s, "moe shexp out")?;
            for (matrix, out) in [(gate_mat, sh_gate), (up_mat, sh_up)] {
                let input: &DeviceBuffer = match matrix.dtype {
                    GemmDType::F32 => xs_f32,
                    GemmDType::F16 => {
                        let cast =
                            ensure_scratch(&mut scratch.xs_f16, b * embed * 2, "moe xs f16")?;
                        if !scratch.xs_f16_valid {
                            crate::kernels::launch_cast_f32_to_f16(
                                xs_f32,
                                cast,
                                b * embed,
                                &self.stream,
                            )?;
                            launches += 1;
                            scratch.xs_f16_valid = true;
                        }
                        cast
                    }
                    GemmDType::BF16 => {
                        let cast =
                            ensure_scratch(&mut scratch.xs_bf16, b * embed * 2, "moe xs bf16")?;
                        if !scratch.xs_bf16_valid {
                            crate::kernels::launch_cast_f32_to_bf16(
                                xs_f32,
                                cast,
                                b * embed,
                                &self.stream,
                            )?;
                            launches += 1;
                            scratch.xs_bf16_valid = true;
                        }
                        cast
                    }
                };
                let elem = gemm_element_size(matrix.dtype);
                for t in 0..b {
                    self.resident_gemv_at(matrix, input, t * embed * elem, out, t * sff * f32s)?;
                }
                launches += b as u64;
            }
            // Plain SwiGLU (clamp 0 disables the clamps), in place into gate.
            crate::kernels::launch_dsv4_swiglu_clamp(
                sh_gate,
                sh_up,
                sh_gate,
                b * sff,
                0.0,
                &self.stream,
            )?;
            launches += 1;
            let down_input: &DeviceBuffer = match down_mat.dtype {
                GemmDType::F32 => sh_gate,
                dtype => {
                    let cast =
                        ensure_scratch(&mut scratch.shared_cast, b * sff * 2, "moe shexp cast")?;
                    match dtype {
                        GemmDType::F16 => crate::kernels::launch_cast_f32_to_f16(
                            sh_gate,
                            cast,
                            b * sff,
                            &self.stream,
                        )?,
                        GemmDType::BF16 => crate::kernels::launch_cast_f32_to_bf16(
                            sh_gate,
                            cast,
                            b * sff,
                            &self.stream,
                        )?,
                        GemmDType::F32 => unreachable!("F32 handled above"),
                    }
                    launches += 1;
                    cast
                }
            };
            let down_elem = gemm_element_size(down_mat.dtype);
            for t in 0..b {
                self.resident_gemv_at(
                    down_mat,
                    down_input,
                    t * sff * down_elem,
                    sh_out,
                    t * embed * f32s,
                )?;
            }
            launches += b as u64;
            Some(sh_out)
        } else {
            None
        };

        // ---- Weighted accumulate (selection order, host rounding) + shared.
        let ys = ensure_scratch(&mut scratch.ys, b * embed * f32s, "moe ys")?;
        crate::kernels::launch_dsv4_moe_accum(
            expert_out,
            weights_buf,
            shared_out,
            ys,
            b,
            k,
            embed,
            &self.stream,
        )?;
        launches += 1;

        Ok(MoeCoreOutcome::Done { launches, syncs })
    }
}

// ---- Wave-2 Stage 2b: device decode step orchestration --------------------
impl DsV4GpuLinearInner {
    /// The [`DsV4Linear::try_device_step`] implementation: decline (host step)
    /// when killed via `HI_DSV4_HOST_STEP=1`, when the position would overflow
    /// the context (the host step owns that error), or when the model shape is
    /// unsupported; otherwise run the full device step.
    fn try_device_step_impl(
        &self,
        engine: &DsV4Engine<DsV4GpuLinear>,
        state: &mut DsV4State,
        token: u32,
    ) -> Option<Result<Vec<f32>>> {
        if self.host_step_forced.get() {
            self.step_stats.borrow_mut().host_steps += 1;
            return None;
        }
        if state.pos >= engine.geometry().context {
            return None;
        }
        {
            let mut guard = self.device_step.borrow_mut();
            if guard.is_none() {
                *guard = Some(match self.build_device_step(engine) {
                    Ok(build) => build,
                    Err(err) => {
                        eprintln!(
                            "dsv4 device step disabled (resource build failed): {err:#}; decoding stays on the host step"
                        );
                        DeviceStepBuild::Unsupported
                    }
                });
            }
            if matches!(guard.as_ref(), Some(DeviceStepBuild::Unsupported)) {
                self.step_stats.borrow_mut().host_steps += 1;
                return None;
            }
        }
        Some(self.device_decode_step(engine, state, token))
    }

    /// Build every device-step resource once: uploaded small weights, fixed
    /// activation buffers, the packed embedding copy, and the arena layout.
    /// `Unsupported` (reason logged) declines cleanly to the host step.
    fn build_device_step(&self, engine: &DsV4Engine<DsV4GpuLinear>) -> Result<DeviceStepBuild> {
        let unsupported = |reason: &'static str| {
            eprintln!("dsv4 device step unavailable ({reason}); decoding stays on the host step");
            Ok(DeviceStepBuild::Unsupported)
        };
        let g = engine.geometry();
        let Some(window) = g.window else {
            return unsupported("no sliding window");
        };
        if g.hc == 0 || 32 + (g.hc * g.hc + 2 * g.hc) > 256 {
            return unsupported("hyper-connection width exceeds the kernel block");
        }
        if g.rope_dims > g.head_dim || !g.rope_dims.is_multiple_of(2) {
            return unsupported("rope tail shape");
        }
        if g.o_groups == 0 || !(g.heads * g.head_dim).is_multiple_of(g.o_groups) {
            return unsupported("grouped output projection shape");
        }
        let hc_rows = g.hc * g.hc + 2 * g.hc;
        let group_features = (g.heads * g.head_dim) / g.o_groups;
        let resident = self.resident.borrow();
        let dense_dtype =
            |matrix: &RawMatrix| -> Result<GemmDType> { Ok(dense_entry(&resident, matrix)?.dtype) };

        let hyper_weights = engine.hyper_head_weights();
        if hyper_weights.func.shape() != (g.hc, g.hc * g.embed) {
            return unsupported("hyper head mixer shape");
        }
        let mut rope_bases: Vec<f32> = Vec::new();
        let mut layers = Vec::with_capacity(engine.layers().len());
        let mut max_comp_rows = 0usize; // ratio * out_stride over all compressors
        for layer in engine.layers() {
            if layer.hc_attn.func.shape() != (hc_rows, g.hc * g.embed)
                || layer.hc_ffn.func.shape() != (hc_rows, g.hc * g.embed)
                || layer.hc_attn.scale.len() < 3
                || layer.hc_ffn.scale.len() < 3
            {
                return unsupported("hyper-connection mixer shape");
            }
            let x_dtype = dense_dtype(&layer.q_a)?;
            let mut x_consumers = vec![dense_dtype(&layer.kv)?];
            if let Some(comp) = &layer.compressor {
                x_consumers.push(dense_dtype(&comp.gate)?);
                x_consumers.push(dense_dtype(&comp.kv)?);
            }
            if let Some(indexer) = &layer.indexer {
                x_consumers.push(dense_dtype(&indexer.proj)?);
                x_consumers.push(dense_dtype(&indexer.compressor.gate)?);
                x_consumers.push(dense_dtype(&indexer.compressor.kv)?);
            }
            if x_consumers.iter().any(|dtype| *dtype != x_dtype) {
                return unsupported("mixed activation-consumer dtypes");
            }
            let qr_dtype = dense_dtype(&layer.q_b)?;
            if let Some(indexer) = &layer.indexer
                && dense_dtype(&indexer.q_b)? != qr_dtype
            {
                return unsupported("mixed q-latent-consumer dtypes");
            }
            let wo_groups = grouped_entry(&resident, &layer.out_a)?;
            if wo_groups.len() != g.o_groups {
                return unsupported("grouped output projection group count");
            }
            let wo_a_dtype = wo_groups[0].dtype;
            let wo_b_dtype = dense_dtype(&layer.out_b)?;
            let a_elem = gemm_element_size(wo_a_dtype);
            let wo_direct =
                (group_features * a_elem).is_multiple_of(256) && (g.o_rank * 4).is_multiple_of(256);

            let base_idx = match rope_bases
                .iter()
                .position(|base| base.to_bits() == layer.rope_base.to_bits())
            {
                Some(idx) => idx,
                None => {
                    rope_bases.push(layer.rope_base);
                    rope_bases.len() - 1
                }
            };
            let build_comp = |weights: &CompressorWeights| -> Result<DevCompConsts> {
                let out_stride = align256(weights.width * 4) / 4;
                Ok(DevCompConsts {
                    ape: upload_f32(&weights.ape, "dsv4 compressor ape")?,
                    norm: upload_f32(&weights.norm, "dsv4 compressor norm")?,
                    ratio: weights.ratio,
                    dim: weights.dim,
                    width: weights.width,
                    out_stride,
                    pending_row_bytes: align256(g.embed * gemm_element_size(x_dtype)),
                })
            };
            let comp = layer.compressor.as_ref().map(&build_comp).transpose()?;
            let idx = layer
                .indexer
                .as_ref()
                .map(|indexer| build_comp(&indexer.compressor))
                .transpose()?;
            for consts in [comp.as_ref(), idx.as_ref()].into_iter().flatten() {
                max_comp_rows = max_comp_rows.max(consts.ratio * consts.out_stride);
            }
            let upload_hc = |hc: &crate::dsv4_cpu::HcWeights| -> Result<DevHc> {
                Ok(DevHc {
                    func: upload_f32(hc.func.data(), "dsv4 hc mixer")?,
                    base: upload_f32(&hc.base, "dsv4 hc base")?,
                    scale: upload_f32(&hc.scale, "dsv4 hc scale")?,
                    rows: hc.func.shape().0,
                })
            };
            layers.push(DevLayerConsts {
                attn_norm: upload_f32(&layer.attn_norm, "dsv4 attn norm")?,
                ffn_norm: upload_f32(&layer.ffn_norm, "dsv4 ffn norm")?,
                q_a_norm: upload_f32(&layer.q_a_norm, "dsv4 q_a norm")?,
                kv_norm: upload_f32(&layer.kv_norm, "dsv4 kv norm")?,
                sinks: layer
                    .sinks
                    .as_ref()
                    .map(|sinks| upload_f32(sinks, "dsv4 sinks"))
                    .transpose()?,
                hc_attn: upload_hc(&layer.hc_attn)?,
                hc_ffn: upload_hc(&layer.hc_ffn)?,
                base_idx,
                x_dtype,
                qr_dtype,
                wo_a_dtype,
                wo_b_dtype,
                wo_direct,
                comp,
                idx,
            });
        }
        let head_dtype = dense_dtype(engine.output_head_matrix())?;
        drop(resident);

        // Packed embedding copy for the device gather; exotic dtypes fall
        // back to a host dequant + upload per step (still one device step).
        let embed_matrix = engine.token_embd_matrix();
        let embed_view = self
            .gguf
            .tensor(&embed_matrix.name)
            .ok_or_else(|| anyhow!("GGUF tensor {} is missing", embed_matrix.name))?;
        let embed_code = match embed_view.info.dtype {
            GgufTensorType::F32 => Some(0u32),
            GgufTensorType::F16 => Some(1),
            GgufTensorType::BF16 => Some(2),
            GgufTensorType::Q8_0 if g.embed.is_multiple_of(32) => Some(3),
            _ => None,
        };
        let embed_src = match embed_code {
            Some(code) => match DeviceBuffer::alloc(embed_view.bytes.len().max(4)) {
                Ok(buffer) => {
                    buffer.copy_from_host(embed_view.bytes)?;
                    EmbedSrc::Packed { buffer, code }
                }
                Err(err) => {
                    eprintln!(
                        "dsv4 device step: packed embedding upload failed ({err:#}); gathering embedding rows on host"
                    );
                    EmbedSrc::Host {
                        staging: alloc_dev(g.embed * 4, "dsv4 embed staging")?,
                    }
                }
            },
            None => EmbedSrc::Host {
                staging: alloc_dev(g.embed * 4, "dsv4 embed staging")?,
            },
        };

        // Fixed arena layout: logits, then per-layer mirror slots.
        let n_layers = layers.len();
        let mut next = align_elems(g.vocab);
        let mut kv_off = Vec::with_capacity(n_layers);
        let mut x_off = Vec::with_capacity(n_layers);
        let mut comp_off = Vec::with_capacity(n_layers);
        let mut idx_off = Vec::with_capacity(n_layers);
        for lc in &layers {
            kv_off.push(next);
            next += align_elems(g.head_dim);
            x_off.push(next);
            next += align_elems(g.embed);
            comp_off.push(lc.comp.as_ref().map(|consts| {
                let k = next;
                next += align_elems(consts.dim);
                let v = next;
                next += align_elems(consts.dim);
                (k, v)
            }));
            idx_off.push(lc.idx.as_ref().map(|consts| {
                let k = next;
                next += align_elems(consts.dim);
                let v = next;
                next += align_elems(consts.dim);
                (k, v)
            }));
        }
        let arena_layout = ArenaLayout {
            kv_off,
            x_off,
            comp_off,
            idx_off,
            total: next,
        };

        let half = g.rope_dims / 2;
        let rope_base_count = rope_bases.len().max(1);
        let wo_pad_in_stride = align256(group_features * 4);
        let proj_pad_stride = align256(g.o_rank * 4);
        let res = DeviceStepRes {
            embed: g.embed,
            heads: g.heads,
            head_dim: g.head_dim,
            rope_dims: g.rope_dims,
            q_lora: g.q_lora,
            o_groups: g.o_groups,
            o_rank: g.o_rank,
            group_features,
            window,
            hc: g.hc,
            sinkhorn: g.sinkhorn_iterations,
            idx_heads: g.idx_heads,
            idx_key: g.idx_key,
            idx_top_k: g.idx_top_k,
            vocab: g.vocab,
            rms_eps: engine.rms_eps(),
            hc_eps: engine.hc_eps(),
            attn_scale: (g.head_dim as f32).powf(-0.5),
            idx_head_scale: if g.idx_heads > 0 {
                (g.idx_heads as f32).powf(-0.5)
            } else {
                0.0
            },
            idx_key_scale: if g.idx_key > 0 {
                (g.idx_key as f32).powf(-0.5)
            } else {
                0.0
            },
            head_dtype,
            rope_bases,
            layers,
            hyper: DevHc {
                func: upload_f32(hyper_weights.func.data(), "dsv4 hyper mixer")?,
                base: upload_f32(&hyper_weights.base, "dsv4 hyper base")?,
                scale: upload_f32(&hyper_weights.scale, "dsv4 hyper scale")?,
                rows: g.hc,
            },
            hyper_scale0: hyper_weights.scale[0],
            output_norm: upload_f32(engine.output_norm_weights(), "dsv4 output norm")?,
            embed_src,
            arena: alloc_dev(arena_layout.total * 4, "dsv4 step arena")?,
            arena_layout,
            streams_a: alloc_dev(g.hc * g.embed * 4, "dsv4 streams A")?,
            streams_b: alloc_dev(g.hc * g.embed * 4, "dsv4 streams B")?,
            post: alloc_dev(g.hc * 4, "dsv4 hc post gates")?,
            comb: alloc_dev(g.hc * g.hc * 4, "dsv4 hc comb")?,
            x_cast: alloc_dev(g.embed * 2, "dsv4 x cast")?,
            qr: alloc_dev(g.q_lora * 4, "dsv4 q latent")?,
            qr_cast: alloc_dev(g.q_lora * 2, "dsv4 q latent cast")?,
            q: alloc_dev(g.heads * g.head_dim * 4, "dsv4 q")?,
            kv_tmp: alloc_dev(g.head_dim * 4, "dsv4 kv scratch")?,
            attn_out: alloc_dev(g.heads * g.head_dim * 4, "dsv4 attention out")?,
            attn_out_cast: alloc_dev(g.heads * g.head_dim * 2, "dsv4 attention out cast")?,
            attn_final: alloc_dev(g.embed * 4, "dsv4 attention block out")?,
            wo_pad_in: alloc_dev(g.o_groups * wo_pad_in_stride, "dsv4 wo padded input")?,
            wo_pad_in_stride,
            proj_pad: alloc_dev(g.o_groups * proj_pad_stride, "dsv4 wo padded output")?,
            proj_pad_stride,
            proj_flat: alloc_dev(g.o_groups * g.o_rank * 4, "dsv4 wo projected")?,
            proj_cast: alloc_dev(g.o_groups * g.o_rank * 2, "dsv4 wo projected cast")?,
            comp_gates: alloc_dev(max_comp_rows * 4, "dsv4 compressor gates")?,
            comp_kvs: alloc_dev(max_comp_rows * 4, "dsv4 compressor kvs")?,
            ffn_y: alloc_dev(g.embed * 4, "dsv4 ffn input")?,
            ffn_out: alloc_dev(g.embed * 4, "dsv4 ffn fallback output")?,
            idx_qi: alloc_dev(g.idx_heads * g.idx_key * 4, "dsv4 indexer queries")?,
            idx_w: alloc_dev(g.idx_heads * 4, "dsv4 indexer head weights")?,
            idx_sel: alloc_dev(g.idx_top_k * 4, "dsv4 indexer selection")?,
            idx_scores: None,
            idx_marks: None,
            attn_w: None,
            attn_wn: None,
            attn_max_keys: 0,
            hyper_hidden: alloc_dev(g.embed * 4, "dsv4 hyper hidden")?,
            hyper_cast: alloc_dev(g.embed * 2, "dsv4 hyper hidden cast")?,
            // One [fwd cos|sin, inv cos|sin] block of rope_dims/2 pairs per
            // distinct rope base, refilled per step at the step's position.
            rope_buf: alloc_dev(rope_base_count * 4 * half.max(1) * 4, "dsv4 rope table")?,
            restore_staging: None,
            state: None,
            tag_counter: 0,
            retired: Vec::new(),
        };
        Ok(DeviceStepBuild::Ready(Box::new(res)))
    }

    /// Cast `elems` f32 values at a byte offset into `dst` (offset 0) in the
    /// matrix dtype. Callers skip F32 (no cast needed).
    fn cast_slice(
        &self,
        dtype: GemmDType,
        src: &DeviceBuffer,
        src_byte_offset: usize,
        dst: &DeviceBuffer,
        dst_byte_offset: usize,
        elems: usize,
        launches: &mut u64,
    ) -> Result<()> {
        match dtype {
            GemmDType::F32 => bail!("cast_slice called for F32"),
            GemmDType::F16 => crate::kernels::launch_cast_f32_to_f16_slice(
                src,
                src_byte_offset,
                dst,
                dst_byte_offset,
                elems,
                &self.stream,
            )?,
            GemmDType::BF16 => crate::kernels::launch_cast_f32_to_bf16_slice(
                src,
                src_byte_offset,
                dst,
                dst_byte_offset,
                elems,
                &self.stream,
            )?,
        }
        *launches += 1;
        Ok(())
    }

    /// Full host→device state restore: upload the raw rings (trailing window
    /// only — slack-retained entries are never attention-visible), compressed
    /// K/V blocks, and pending activations (re-cast to each layer's GEMV
    /// dtype exactly as the in-step path casts them). Reuses the device-state
    /// buffers across restores; returns (blocking copies, async launches).
    fn restore_device_state(
        &self,
        res: &mut DeviceStepRes,
        state: &DsV4State,
    ) -> Result<(u64, u64)> {
        let mut copies = 0u64;
        let mut launches = 0u64;
        if state.layers.len() != res.layers.len() {
            bail!("state layer count does not match the model");
        }
        // Size the pending staging ONCE up front: a restore begins with an
        // idle stream (the previous step ended in a full sync), which is the
        // only point a grow-free is provably safe — later in the restore,
        // queued async pending casts may still be reading the buffer.
        let max_pending_bytes = state
            .layers
            .iter()
            .flat_map(|layer| [&layer.compressor, &layer.indexer])
            .flatten()
            .map(|comp| comp.pending.len() * res.embed * 4)
            .max()
            .unwrap_or(0);
        if max_pending_bytes > 0 {
            ensure_scratch(
                &mut res.restore_staging,
                max_pending_bytes,
                "dsv4 restore staging",
            )?;
        }
        if res.state.is_none() {
            let mut dev_layers = Vec::with_capacity(res.layers.len());
            for lc in &res.layers {
                let alloc_comp = |consts: &DevCompConsts| -> Result<DevCompState> {
                    let cap = 64usize;
                    Ok(DevCompState {
                        keys: alloc_dev(cap * consts.dim * 4, "dsv4 compressed keys")?,
                        values: alloc_dev(cap * consts.dim * 4, "dsv4 compressed values")?,
                        cap_blocks: cap,
                        pending: alloc_dev(
                            consts.ratio * consts.pending_row_bytes,
                            "dsv4 pending rows",
                        )?,
                    })
                };
                dev_layers.push(DevLayerState {
                    ring: alloc_dev(res.window * res.head_dim * 4, "dsv4 device ring")?,
                    comp: lc.comp.as_ref().map(&alloc_comp).transpose()?,
                    idx: lc.idx.as_ref().map(&alloc_comp).transpose()?,
                });
            }
            res.state = Some(DevState {
                tag: 0,
                pos: 0,
                layers: dev_layers,
            });
        }
        let dev = res.state.as_mut().expect("device state was just built");
        for (li, host_layer) in state.layers.iter().enumerate() {
            let lc = &res.layers[li];
            let dl = &mut dev.layers[li];
            // Raw ring: trailing min(len, window) entries into circular slots.
            let len = host_layer.ring.len();
            let take = len.min(res.window);
            if take > 0 {
                let start_pos = state.pos - take;
                let s0 = start_pos % res.window;
                let first_run = take.min(res.window - s0);
                let mut flat = Vec::with_capacity(first_run * res.head_dim);
                for entry in host_layer.ring.iter().skip(len - take).take(first_run) {
                    if entry.len() != res.head_dim {
                        bail!("ring latent width {} does not match head dim", entry.len());
                    }
                    flat.extend_from_slice(entry);
                }
                dl.ring.copy_from_host_at(s0 * res.head_dim * 4, &flat)?;
                copies += 1;
                if first_run < take {
                    let mut flat = Vec::with_capacity((take - first_run) * res.head_dim);
                    for entry in host_layer.ring.iter().skip(len - take + first_run) {
                        flat.extend_from_slice(entry);
                    }
                    dl.ring.copy_from_host_at(0, &flat)?;
                    copies += 1;
                }
            }
            for (consts, host_comp, dev_comp) in [
                (lc.comp.as_ref(), &host_layer.compressor, &mut dl.comp),
                (lc.idx.as_ref(), &host_layer.indexer, &mut dl.idx),
            ] {
                let (Some(consts), Some(host_comp), Some(dev_comp)) =
                    (consts, host_comp.as_ref(), dev_comp.as_mut())
                else {
                    continue;
                };
                let blocks = host_comp.keys.len();
                if blocks > dev_comp.cap_blocks {
                    let cap = blocks.next_power_of_two().max(64);
                    dev_comp.keys = alloc_dev(cap * consts.dim * 4, "dsv4 compressed keys")?;
                    dev_comp.values = alloc_dev(cap * consts.dim * 4, "dsv4 compressed values")?;
                    dev_comp.cap_blocks = cap;
                }
                if blocks > 0 {
                    let mut flat = Vec::with_capacity(blocks * consts.dim);
                    for key in &host_comp.keys {
                        flat.extend_from_slice(key);
                    }
                    dev_comp.keys.copy_from_host(&flat)?;
                    flat.clear();
                    for value in &host_comp.values {
                        flat.extend_from_slice(value);
                    }
                    dev_comp.values.copy_from_host(&flat)?;
                    copies += 2;
                }
                let n_pending = host_comp.pending.len();
                if n_pending > 0 {
                    let mut flat = Vec::with_capacity(n_pending * res.embed);
                    for row in &host_comp.pending {
                        flat.extend_from_slice(row);
                    }
                    let staging = res
                        .restore_staging
                        .as_ref()
                        .expect("restore staging pre-sized at restore start");
                    staging.copy_from_host(&flat)?;
                    copies += 1;
                    for row in 0..n_pending {
                        if lc.x_dtype == GemmDType::F32 {
                            dev_comp.pending.copy_device_range(
                                row * consts.pending_row_bytes,
                                staging,
                                row * res.embed * 4,
                                res.embed * 4,
                                &self.stream,
                            )?;
                            launches += 1;
                        } else {
                            self.cast_slice(
                                lc.x_dtype,
                                staging,
                                row * res.embed * 4,
                                &dev_comp.pending,
                                row * consts.pending_row_bytes,
                                res.embed,
                                &mut launches,
                            )?;
                        }
                    }
                }
            }
        }
        dev.pos = state.pos;
        Ok((copies, launches))
    }

    /// One APE compressor's per-step work: stage the activation into its
    /// pending row, and on block completion run the per-row gate/kv GEMVs
    /// (the host path's exact M=1 calls) + the emit kernel into the
    /// compressed cache and arena mirror slots.
    #[allow(clippy::too_many_arguments)]
    fn step_compressor(
        &self,
        weights: &CompressorWeights,
        consts: &DevCompConsts,
        dev: &mut DevCompState,
        resident: &HashMap<String, ResidentEntry>,
        x_in: (&DeviceBuffer, usize),
        x_elem_bytes: usize,
        embed: usize,
        pos: usize,
        rms_eps: f32,
        gates: &DeviceBuffer,
        kvs: &DeviceBuffer,
        arena: &DeviceBuffer,
        arena_pair: (usize, usize),
        retired: &mut Vec<DeviceBuffer>,
        launches: &mut u64,
    ) -> Result<()> {
        // Pending push: the host stores x and casts it per GEMV; the device
        // stores the already-cast row (identical rounding).
        let row = pos % consts.ratio;
        dev.pending.copy_device_range(
            row * consts.pending_row_bytes,
            x_in.0,
            x_in.1,
            embed * x_elem_bytes,
            &self.stream,
        )?;
        *launches += 1;
        if !(pos + 1).is_multiple_of(consts.ratio) {
            return Ok(());
        }
        let block = (pos + 1) / consts.ratio - 1;
        if block + 1 > dev.cap_blocks {
            let cap = (dev.cap_blocks * 2).max(block + 1).max(64);
            let keys = alloc_dev(cap * consts.dim * 4, "dsv4 compressed keys")?;
            let values = alloc_dev(cap * consts.dim * 4, "dsv4 compressed values")?;
            keys.copy_device_range(0, &dev.keys, 0, block * consts.dim * 4, &self.stream)?;
            values.copy_device_range(0, &dev.values, 0, block * consts.dim * 4, &self.stream)?;
            *launches += 2;
            retired.push(std::mem::replace(&mut dev.keys, keys));
            retired.push(std::mem::replace(&mut dev.values, values));
            dev.cap_blocks = cap;
        }
        let gate_m = dense_entry(resident, &weights.gate)?;
        let kv_m = dense_entry(resident, &weights.kv)?;
        for r in 0..consts.ratio {
            self.resident_gemv_at(
                gate_m,
                &dev.pending,
                r * consts.pending_row_bytes,
                gates,
                r * consts.out_stride * 4,
            )?;
            self.resident_gemv_at(
                kv_m,
                &dev.pending,
                r * consts.pending_row_bytes,
                kvs,
                r * consts.out_stride * 4,
            )?;
        }
        *launches += 2 * consts.ratio as u64;
        crate::kernels::launch_dsv4_compressor_emit(
            gates,
            kvs,
            consts.out_stride,
            &consts.ape,
            &consts.norm,
            consts.ratio,
            consts.dim,
            consts.width,
            rms_eps,
            &dev.keys,
            block * consts.dim * 4,
            &dev.values,
            block * consts.dim * 4,
            arena,
            arena_pair.0 * 4,
            arena_pair.1 * 4,
            &self.stream,
        )?;
        *launches += 1;
        Ok(())
    }

    /// Run one decode step device-side. Any error invalidates the device
    /// state (partially-written ring slots / pending rows), forcing the next
    /// step to restore from the untouched host mirror.
    fn device_decode_step(
        &self,
        engine: &DsV4Engine<DsV4GpuLinear>,
        state: &mut DsV4State,
        token: u32,
    ) -> Result<Vec<f32>> {
        let result = self.device_decode_step_inner(engine, state, token);
        if result.is_err() {
            // A failed step may have partially mutated device buffers (ring
            // slot, pending rows): drain the stream so nothing still reads
            // them, then drop the device state so the next step restores from
            // the untouched host mirror.
            let _ = self.stream.synchronize();
            if let Some(DeviceStepBuild::Ready(res)) = self.device_step.borrow_mut().as_mut() {
                res.state = None;
                res.retired.clear();
            }
        }
        result
    }

    fn device_decode_step_inner(
        &self,
        engine: &DsV4Engine<DsV4GpuLinear>,
        state: &mut DsV4State,
        token: u32,
    ) -> Result<Vec<f32>> {
        let mut build_guard = self.device_step.borrow_mut();
        let Some(DeviceStepBuild::Ready(res)) = build_guard.as_mut() else {
            bail!("device step resources are unavailable");
        };
        let res: &mut DeviceStepRes = res;
        let stream = &self.stream;
        let pos = state.pos;
        if (token as usize) >= res.vocab {
            bail!("token id {token} is outside vocab size {}", res.vocab);
        }
        let mut launches = 0u64;
        let mut syncs = 0u64;
        let mut restore_syncs = 0u64;
        let mut restored = false;

        // ---- Device-state currency: (tag, pos) must match the host state.
        let current = matches!(
            &res.state,
            Some(dev) if state.device_tag != 0 && dev.tag == state.device_tag && dev.pos == pos
        );
        if !current {
            let (copies, restore_launches) = self.restore_device_state(res, state)?;
            restore_syncs += copies;
            launches += restore_launches;
            res.tag_counter += 1;
            let dev = res.state.as_mut().expect("restore built the device state");
            dev.tag = res.tag_counter;
            dev.pos = pos;
            state.device_tag = res.tag_counter;
            restored = true;
        }

        // ---- Step-start scratch growth (the stream is idle here: the
        // previous step ended in a full sync; restore copies are blocking).
        let layer_defs = engine.layers();
        let n_ring = res.window.min(pos + 1);
        let mut max_keys = n_ring;
        let mut max_blocks = 0usize;
        for (layer, lc) in layer_defs.iter().zip(&res.layers) {
            if let Some(consts) = &lc.comp {
                let blocks_after = (pos + 1) / consts.ratio;
                let n_comp = if layer.indexer.is_some() && blocks_after > res.idx_top_k {
                    res.idx_top_k
                } else {
                    blocks_after
                };
                max_keys = max_keys.max(n_comp + n_ring);
            }
            if let Some(consts) = &lc.idx {
                max_blocks = max_blocks.max((pos + 1) / consts.ratio);
            }
        }
        if max_keys > res.attn_max_keys {
            let cap = max_keys.next_power_of_two().max(256);
            ensure_scratch(
                &mut res.attn_w,
                res.heads * cap * 4,
                "dsv4 attention weights",
            )?;
            ensure_scratch(
                &mut res.attn_wn,
                res.heads * cap * 4,
                "dsv4 attention normalized weights",
            )?;
            res.attn_max_keys = cap;
        }
        if max_blocks > 0 {
            ensure_scratch(&mut res.idx_scores, max_blocks * 4, "dsv4 indexer scores")?;
            ensure_scratch(&mut res.idx_marks, max_blocks, "dsv4 indexer marks")?;
        }

        // ---- Per-step uploads (async on the engine stream; tiny).
        let half = res.rope_dims / 2;
        if res.rope_dims > 0 {
            let mut rope_host = Vec::with_capacity(res.rope_bases.len() * 4 * half);
            for &base in &res.rope_bases {
                for inverse in [false, true] {
                    let table = v4_rope_sincos(res.rope_dims, pos, base, inverse);
                    rope_host.extend(table.iter().map(|&(_, cos)| cos));
                    rope_host.extend(table.iter().map(|&(sin, _)| sin));
                }
            }
            res.rope_buf.copy_from_host_async(&rope_host, stream)?;
            launches += 1;
        }

        // ---- Embedding gather + broadcast into the hc streams.
        match &res.embed_src {
            EmbedSrc::Packed { buffer, code } => {
                crate::kernels::launch_dsv4_embed_broadcast(
                    buffer,
                    *code,
                    token as usize * res.embed,
                    res.embed,
                    res.hc,
                    &res.streams_a,
                    stream,
                )?;
            }
            EmbedSrc::Host { staging } => {
                let row = engine.embed_row(token)?;
                staging.copy_from_host_async(&row, stream)?;
                launches += 1;
                crate::kernels::launch_dsv4_embed_broadcast(
                    staging,
                    0,
                    0,
                    res.embed,
                    res.hc,
                    &res.streams_a,
                    stream,
                )?;
            }
        }
        launches += 1;

        // ---- Layers. Stream ping-pong: attn half A->B, ffn half B->A.
        let resident = self.resident.borrow();
        let dev_state = res.state.as_mut().expect("device state is current");
        for (li, layer) in layer_defs.iter().enumerate() {
            let lc = &res.layers[li];
            let lstate = &mut dev_state.layers[li];
            let rope_fwd = lc.base_idx * 4 * half * 4;
            let rope_inv = rope_fwd + 2 * half * 4;
            let x_off_bytes = res.arena_layout.x_off[li] * 4;

            // hc_attn.pre + attn_norm, x into the arena mirror slot.
            crate::kernels::launch_dsv4_hc_pre(
                &res.streams_a,
                &lc.hc_attn.func,
                &lc.hc_attn.base,
                &lc.hc_attn.scale,
                Some(&lc.attn_norm),
                res.hc,
                res.embed,
                lc.hc_attn.rows,
                res.sinkhorn,
                res.rms_eps,
                res.hc_eps,
                &res.arena,
                x_off_bytes,
                &res.post,
                &res.comb,
                stream,
            )?;
            launches += 1;

            let x_elem = gemm_element_size(lc.x_dtype);
            let x_in: (&DeviceBuffer, usize) = if lc.x_dtype == GemmDType::F32 {
                (&res.arena, x_off_bytes)
            } else {
                self.cast_slice(
                    lc.x_dtype,
                    &res.arena,
                    x_off_bytes,
                    &res.x_cast,
                    0,
                    res.embed,
                    &mut launches,
                )?;
                (&res.x_cast, 0)
            };

            // q latent -> per-head q (+ unweighted RMS + rope).
            self.resident_gemv_at(
                dense_entry(&resident, &layer.q_a)?,
                x_in.0,
                x_in.1,
                &res.qr,
                0,
            )?;
            launches += 1;
            crate::kernels::launch_dsv4_rms_exact(
                &res.qr,
                &lc.q_a_norm,
                res.q_lora,
                res.rms_eps,
                stream,
            )?;
            launches += 1;
            let qr_in: (&DeviceBuffer, usize) = if lc.qr_dtype == GemmDType::F32 {
                (&res.qr, 0)
            } else {
                self.cast_slice(
                    lc.qr_dtype,
                    &res.qr,
                    0,
                    &res.qr_cast,
                    0,
                    res.q_lora,
                    &mut launches,
                )?;
                (&res.qr_cast, 0)
            };
            self.resident_gemv_at(
                dense_entry(&resident, &layer.q_b)?,
                qr_in.0,
                qr_in.1,
                &res.q,
                0,
            )?;
            launches += 1;
            crate::kernels::launch_dsv4_q_prep(
                &res.q,
                res.heads,
                res.head_dim,
                res.rope_dims,
                &res.rope_buf,
                rope_fwd,
                res.rms_eps,
                stream,
            )?;
            launches += 1;

            // Shared KV latent -> ring slot + arena mirror slot.
            self.resident_gemv_at(
                dense_entry(&resident, &layer.kv)?,
                x_in.0,
                x_in.1,
                &res.kv_tmp,
                0,
            )?;
            launches += 1;
            crate::kernels::launch_dsv4_kv_prep(
                &res.kv_tmp,
                &lc.kv_norm,
                res.head_dim,
                res.rope_dims,
                &res.rope_buf,
                rope_fwd,
                res.rms_eps,
                &lstate.ring,
                (pos % res.window) * res.head_dim * 4,
                &res.arena,
                res.arena_layout.kv_off[li] * 4,
                stream,
            )?;
            launches += 1;

            // Compressor + indexer-compressor pending/completion.
            if let (Some(weights), Some(consts), Some(dev_comp)) =
                (&layer.compressor, &lc.comp, &mut lstate.comp)
            {
                let pair = res.arena_layout.comp_off[li].expect("compressor arena slot");
                self.step_compressor(
                    weights,
                    consts,
                    dev_comp,
                    &resident,
                    x_in,
                    x_elem,
                    res.embed,
                    pos,
                    res.rms_eps,
                    &res.comp_gates,
                    &res.comp_kvs,
                    &res.arena,
                    pair,
                    &mut res.retired,
                    &mut launches,
                )?;
            }
            if let (Some(indexer), Some(consts), Some(dev_idx)) =
                (&layer.indexer, &lc.idx, &mut lstate.idx)
            {
                let pair = res.arena_layout.idx_off[li].expect("indexer arena slot");
                self.step_compressor(
                    &indexer.compressor,
                    consts,
                    dev_idx,
                    &resident,
                    x_in,
                    x_elem,
                    res.embed,
                    pos,
                    res.rms_eps,
                    &res.comp_gates,
                    &res.comp_kvs,
                    &res.arena,
                    pair,
                    &mut res.retired,
                    &mut launches,
                )?;
            }

            // Indexer top-k narrowing (blocks visible AFTER this token).
            let mut n_comp = 0usize;
            if let Some(consts) = &lc.comp {
                n_comp = (pos + 1) / consts.ratio;
            }
            let mut sel_used = false;
            if let (Some(indexer), Some(consts), Some(dev_idx)) =
                (&layer.indexer, &lc.idx, &lstate.idx)
            {
                let blocks_after = (pos + 1) / consts.ratio;
                if blocks_after > res.idx_top_k {
                    self.resident_gemv_at(
                        dense_entry(&resident, &indexer.q_b)?,
                        qr_in.0,
                        qr_in.1,
                        &res.idx_qi,
                        0,
                    )?;
                    self.resident_gemv_at(
                        dense_entry(&resident, &indexer.proj)?,
                        x_in.0,
                        x_in.1,
                        &res.idx_w,
                        0,
                    )?;
                    launches += 2;
                    let scores = res.idx_scores.as_ref().expect("indexer scores scratch");
                    let marks = res.idx_marks.as_ref().expect("indexer marks scratch");
                    crate::kernels::launch_dsv4_indexer_score(
                        &res.idx_qi,
                        &res.idx_w,
                        &dev_idx.keys,
                        consts.dim,
                        blocks_after,
                        res.idx_heads,
                        res.idx_key,
                        res.idx_head_scale,
                        res.idx_key_scale,
                        scores,
                        stream,
                    )?;
                    crate::kernels::launch_dsv4_indexer_select(
                        scores,
                        blocks_after,
                        res.idx_top_k,
                        marks,
                        &res.idx_sel,
                        stream,
                    )?;
                    launches += 2;
                    sel_used = true;
                    n_comp = res.idx_top_k;
                }
            }

            // Attention proper over [compressed ‖ ring window].
            let comp_arg = match (&lstate.comp, &lc.comp) {
                (Some(dev_comp), Some(consts)) => {
                    Some((&dev_comp.keys, &dev_comp.values, consts.dim))
                }
                _ => None,
            };
            crate::kernels::launch_dsv4_attention_decode(
                &res.q,
                comp_arg,
                sel_used.then_some(&res.idx_sel),
                n_comp,
                &lstate.ring,
                res.window,
                pos + 1 - n_ring,
                n_ring,
                lc.sinks.as_ref(),
                res.attn_scale,
                res.heads,
                res.head_dim,
                res.rope_dims,
                &res.rope_buf,
                rope_inv,
                &res.attn_out,
                res.attn_w.as_ref().expect("attention weight scratch"),
                res.attn_wn.as_ref().expect("attention weight scratch"),
                res.attn_max_keys,
                stream,
            )?;
            launches += 1;

            // Grouped low-rank output projection.
            let a_elem = gemm_element_size(lc.wo_a_dtype);
            let wo_src: (&DeviceBuffer, usize) = if lc.wo_a_dtype == GemmDType::F32 {
                (&res.attn_out, 4)
            } else {
                self.cast_slice(
                    lc.wo_a_dtype,
                    &res.attn_out,
                    0,
                    &res.attn_out_cast,
                    0,
                    res.heads * res.head_dim,
                    &mut launches,
                )?;
                (&res.attn_out_cast, a_elem)
            };
            let groups = grouped_entry(&resident, &layer.out_a)?;
            for (gidx, entry) in groups.iter().enumerate() {
                let (input, in_off) = if lc.wo_direct {
                    (wo_src.0, gidx * res.group_features * wo_src.1)
                } else {
                    res.wo_pad_in.copy_device_range(
                        gidx * res.wo_pad_in_stride,
                        wo_src.0,
                        gidx * res.group_features * wo_src.1,
                        res.group_features * wo_src.1,
                        stream,
                    )?;
                    launches += 1;
                    (&res.wo_pad_in, gidx * res.wo_pad_in_stride)
                };
                let (output, out_off) = if lc.wo_direct {
                    (&res.proj_flat, gidx * res.o_rank * 4)
                } else {
                    (&res.proj_pad, gidx * res.proj_pad_stride)
                };
                self.resident_gemv_at(entry, input, in_off, output, out_off)?;
            }
            launches += res.o_groups as u64;
            if !lc.wo_direct {
                for gidx in 0..res.o_groups {
                    res.proj_flat.copy_device_range(
                        gidx * res.o_rank * 4,
                        &res.proj_pad,
                        gidx * res.proj_pad_stride,
                        res.o_rank * 4,
                        stream,
                    )?;
                }
                launches += res.o_groups as u64;
            }
            let wo_b_in: (&DeviceBuffer, usize) = if lc.wo_b_dtype == GemmDType::F32 {
                (&res.proj_flat, 0)
            } else {
                self.cast_slice(
                    lc.wo_b_dtype,
                    &res.proj_flat,
                    0,
                    &res.proj_cast,
                    0,
                    res.o_groups * res.o_rank,
                    &mut launches,
                )?;
                (&res.proj_cast, 0)
            };
            self.resident_gemv_at(
                dense_entry(&resident, &layer.out_b)?,
                wo_b_in.0,
                wo_b_in.1,
                &res.attn_final,
                0,
            )?;
            launches += 1;

            // hc_attn.post: streams_a (residual) + attention -> streams_b.
            crate::kernels::launch_dsv4_hc_post(
                &res.attn_final,
                &res.streams_a,
                &res.post,
                &res.comb,
                res.hc,
                res.embed,
                &res.streams_b,
                stream,
            )?;
            launches += 1;

            // hc_ffn.pre + ffn_norm -> ffn_y.
            crate::kernels::launch_dsv4_hc_pre(
                &res.streams_b,
                &lc.hc_ffn.func,
                &lc.hc_ffn.base,
                &lc.hc_ffn.scale,
                Some(&lc.ffn_norm),
                res.hc,
                res.embed,
                lc.hc_ffn.rows,
                res.sinkhorn,
                res.rms_eps,
                res.hc_eps,
                &res.ffn_y,
                0,
                &res.post,
                &res.comb,
                stream,
            )?;
            launches += 1;

            // MoE block: inlined Stage-2a device core (per-layer ids readback
            // stays — it services the expert-pool LRU); unsupported layers
            // (F32 fixture experts, no pool) or a too-small pool fall back to
            // the exact host block on the downloaded activation.
            let ctx = engine.moe_ctx(layer);
            let mut device_moe_done = false;
            if !self.device_moe_disabled.get() && self.device_moe_supported(&ctx) {
                let mut scratch_guard = self.moe_scratch.borrow_mut();
                let scratch = &mut *scratch_guard;
                scratch.xs_f16_valid = false;
                scratch.xs_bf16_valid = false;
                match self.device_moe_core(
                    &ctx,
                    &res.ffn_y,
                    1,
                    std::slice::from_ref(&token),
                    scratch,
                    &resident,
                )? {
                    MoeCoreOutcome::Done {
                        launches: core_launches,
                        syncs: core_syncs,
                    } => {
                        launches += core_launches;
                        syncs += core_syncs;
                        {
                            let mut stats = self.moe_stats.borrow_mut();
                            stats.device_blocks += 1;
                            stats.launches += core_launches;
                            stats.syncs += core_syncs;
                        }
                        let ys = scratch.ys.as_ref().expect("moe core populated ys");
                        crate::kernels::launch_dsv4_hc_post(
                            ys,
                            &res.streams_b,
                            &res.post,
                            &res.comb,
                            res.hc,
                            res.embed,
                            &res.streams_a,
                            stream,
                        )?;
                        launches += 1;
                        device_moe_done = true;
                    }
                    MoeCoreOutcome::PoolTooSmall {
                        launches: core_launches,
                        syncs: core_syncs,
                    } => {
                        launches += core_launches;
                        syncs += core_syncs;
                    }
                }
            }
            if !device_moe_done {
                // Exact host MoE block on the downloaded activation (the host
                // step's own path given identical inputs).
                let y_host: Vec<f32> = res.ffn_y.copy_to_host(res.embed)?;
                syncs += 1;
                let ys_rows = host_moe_block(self, &ctx, std::slice::from_ref(&y_host), &[token])?;
                self.moe_stats.borrow_mut().host_blocks += 1;
                res.ffn_out.copy_from_host_async(&ys_rows[0], stream)?;
                launches += 1;
                crate::kernels::launch_dsv4_hc_post(
                    &res.ffn_out,
                    &res.streams_b,
                    &res.post,
                    &res.comb,
                    res.hc,
                    res.embed,
                    &res.streams_a,
                    stream,
                )?;
                launches += 1;
            }
        }

        // ---- Hyper head + final norm + lm head into the arena's logits row.
        crate::kernels::launch_dsv4_hyper_head(
            &res.streams_a,
            &res.hyper.func,
            &res.hyper.base,
            res.hyper_scale0,
            res.hc,
            res.embed,
            res.rms_eps,
            res.hc_eps,
            &res.hyper_hidden,
            stream,
        )?;
        launches += 1;
        crate::kernels::launch_dsv4_rms_exact(
            &res.hyper_hidden,
            &res.output_norm,
            res.embed,
            res.rms_eps,
            stream,
        )?;
        launches += 1;
        let head_in: (&DeviceBuffer, usize) = if res.head_dtype == GemmDType::F32 {
            (&res.hyper_hidden, 0)
        } else {
            self.cast_slice(
                res.head_dtype,
                &res.hyper_hidden,
                0,
                &res.hyper_cast,
                0,
                res.embed,
                &mut launches,
            )?;
            (&res.hyper_cast, 0)
        };
        self.resident_gemv_at(
            dense_entry(&resident, engine.output_head_matrix())?,
            head_in.0,
            head_in.1,
            &res.arena,
            0,
        )?;
        launches += 1;
        drop(resident);

        // ---- THE end-of-step sync: one download of logits + state delta.
        let arena_host: Vec<f32> = res.arena.copy_to_host(res.arena_layout.total)?;
        syncs += 1;
        res.retired.clear();

        // ---- Replay the delta into the host mirror (stays authoritative).
        let al = &res.arena_layout;
        let mut mirror = DsV4StepMirror {
            kv: Vec::with_capacity(layer_defs.len()),
            x: Vec::with_capacity(layer_defs.len()),
            comp_block: Vec::with_capacity(layer_defs.len()),
            idx_block: Vec::with_capacity(layer_defs.len()),
        };
        for (li, lc) in res.layers.iter().enumerate() {
            mirror
                .kv
                .push(&arena_host[al.kv_off[li]..al.kv_off[li] + res.head_dim]);
            mirror
                .x
                .push(Some(&arena_host[al.x_off[li]..al.x_off[li] + res.embed]));
            let completed = |consts: &DevCompConsts, pair: Option<(usize, usize)>| {
                if (pos + 1).is_multiple_of(consts.ratio) {
                    pair.map(|(k, v)| {
                        (
                            &arena_host[k..k + consts.dim],
                            &arena_host[v..v + consts.dim],
                        )
                    })
                } else {
                    None
                }
            };
            mirror.comp_block.push(
                lc.comp
                    .as_ref()
                    .and_then(|consts| completed(consts, al.comp_off[li])),
            );
            mirror.idx_block.push(
                lc.idx
                    .as_ref()
                    .and_then(|consts| completed(consts, al.idx_off[li])),
            );
        }
        engine.apply_device_step_mirror(state, &mirror)?;
        dev_state.pos = pos + 1;

        let logits = arena_host[..res.vocab].to_vec();
        let mut stats = self.step_stats.borrow_mut();
        stats.device_steps += 1;
        stats.launches += launches;
        stats.syncs += syncs;
        stats.restore_syncs += restore_syncs;
        if restored {
            stats.restores += 1;
        }
        Ok(logits)
    }
}

impl Drop for DsV4GpuLinearInner {
    fn drop(&mut self) {
        // Final pool summary (the periodic line only fires every
        // DSV4_POOL_LOG_EVERY_CALLS pooled GEMVs).
        if let Some(pool) = self.expert_pool.get_mut() {
            let stats = pool.stats;
            if stats.hits + stats.misses + stats.prefilled > 0 {
                eprintln!("dsv4 expert pool (final): {}", format_pool_stats(&stats));
            }
        }
    }
}

impl DsV4Linear for DsV4GpuLinearInner {
    fn mul_vec(&self, key: TensorKey<'_>, x: &[f32]) -> Result<Vec<f32>> {
        match key {
            TensorKey::Dense(matrix) => self.dense_mul_vec(matrix, x),
            TensorKey::Grouped { matrix, rank } => self.grouped_mul_vec(matrix, rank, x),
            TensorKey::Expert { experts, expert } => self.expert_mul_vec(experts, expert, x),
        }
    }

    fn prefetch_experts(&self, tensors: DsV4ExpertTensors<'_>, expert_ids: &[usize]) -> Result<()> {
        self.prefetch_experts_impl(tensors, expert_ids)
    }

    /// Speculative-verify pin: force `mul_mat` onto its bit-exact per-token
    /// loops regardless of `HI_DSV4_PREFILL_GEMM` (see the field docs).
    fn set_exact_batching(&self, exact: bool) {
        self.exact_batching.set(exact);
    }

    // NOTE: `try_device_step` deliberately keeps its default (`None`) here —
    // the engine is parameterized by the [`DsV4GpuLinear`] handle, whose
    // trait impl forwards to [`Self::try_device_step_impl`]. Nothing drives
    // an engine over the inner type directly.

    /// Wave-2 Stage 2a: the whole MoE block device-side (see
    /// [`Self::device_moe_block`]). `HI_DSV4_NO_DEVICE_MOE=1` or an
    /// unsupported layer (F32 fixture experts, no pool, off-grid dims) falls
    /// back to the exact host path.
    fn moe_block(
        &self,
        ctx: &DsV4MoeBlockCtx<'_>,
        xs: &[Vec<f32>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        if self.device_moe_disabled.get() || !self.device_moe_supported(ctx) {
            self.moe_stats.borrow_mut().host_blocks += 1;
            return host_moe_block(self, ctx, xs, tokens);
        }
        self.device_moe_block(ctx, xs, tokens)
    }

    /// True batching for the chunked prefill: resident matrices run one cuBLAS
    /// GEMM per call (per group for the block-diagonal projection), and MXFP4
    /// expert slices dequantize once per call before a single GEMM over the
    /// chunk's tokens. The M=1 decode path is untouched (`resident_gemm` with
    /// m=1 issues the exact GEMV-shaped call `mul_vec` always made).
    fn mul_mat(&self, key: TensorKey<'_>, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        // Default (bit-exact) mode: serve the batch as per-token mul_vec
        // calls — identical invocations to the sequential path, so chunked
        // prefill reproduces it exactly. `HI_DSV4_PREFILL_GEMM=1` enables the
        // GEMM batching below, except while the engine pins exact batching
        // (speculative verify chunks — see `set_exact_batching`).
        if !self.gemm_batching.get() || self.exact_batching.get() {
            return xs.iter().map(|x| self.mul_vec(key, x)).collect();
        }
        // Numerics-debug escape hatch inside GEMM mode:
        // `HI_DSV4_MULMAT_LOOP=dense,grouped,expert` (or `all`) falls back to
        // per-token mul_vec for the listed key kinds, making that kind
        // bit-identical to the sequential path. Used to bisect
        // chunked-vs-sequential drift to a batched operator; read once.
        match mulmat_loop_kinds() {
            kinds if !kinds.is_empty() => {
                let force = |kind: MulMatKind| kinds.contains(&kind);
                match key {
                    TensorKey::Dense(matrix) if force(MulMatKind::Dense) => {
                        return xs.iter().map(|x| self.dense_mul_vec(matrix, x)).collect();
                    }
                    TensorKey::Grouped { matrix, rank } if force(MulMatKind::Grouped) => {
                        return xs
                            .iter()
                            .map(|x| self.grouped_mul_vec(matrix, rank, x))
                            .collect();
                    }
                    TensorKey::Expert { experts, expert } if force(MulMatKind::Expert) => {
                        return xs
                            .iter()
                            .map(|x| self.expert_mul_vec(experts, expert, x))
                            .collect();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        match key {
            TensorKey::Dense(matrix) => self.dense_mul_mat(matrix, xs),
            TensorKey::Grouped { matrix, rank } => self.grouped_mul_mat(matrix, rank, xs),
            TensorKey::Expert { experts, expert } => self.expert_mul_mat(experts, expert, xs),
        }
    }
}

/// `HI_DSV4_PREFILL_GEMM=1` opts the chunked prefill into true GEMM batching
/// (see the module docs for the bit-exactness trade-off).
fn prefill_gemm_from_env() -> bool {
    std::env::var("HI_DSV4_PREFILL_GEMM").ok().as_deref() == Some("1")
}

/// `HI_DSV4_NO_DEVICE_MOE=1` kill switch: route every MoE block through the
/// exact host path (Stage-2a bisection / A-B baseline).
fn device_moe_disabled_from_env() -> bool {
    std::env::var("HI_DSV4_NO_DEVICE_MOE").ok().as_deref() == Some("1")
}

/// `HI_DSV4_NO_COPY_STREAM=1` force-disables copy-stream expert prefetch,
/// reverting to the original synchronous per-miss H2D on the engine stream.
fn copy_stream_disabled() -> bool {
    std::env::var("HI_DSV4_NO_COPY_STREAM").ok().as_deref() == Some("1")
}

/// Whether to build the copy-stream prefetch machinery. Copy-stream prefetch is
/// OPT-IN (`HI_DSV4_COPY_STREAM=1`, or implied by `HI_DSV4_SPEC_PREFETCH=1`),
/// deviating from a default-on design because the real-model measurement showed
/// it throughput-neutral at the default pool size and slightly negative under
/// heavy eviction: decode is bottlenecked on the host-synchronous GEMV
/// orchestration (roadmap items 1-2), not on expert-weight H2D copies, and the
/// pinned-staging memcpy that async DMA requires costs about as much host
/// bandwidth as the DMA it overlaps. The mechanism is kept fully wired and
/// tested so it pays off once the GEMV bottleneck lands. `HI_DSV4_NO_COPY_STREAM=1`
/// overrides everything (hard off, for bisection).
fn copy_stream_enabled() -> bool {
    if copy_stream_disabled() {
        return false;
    }
    let requested = std::env::var("HI_DSV4_COPY_STREAM").ok().as_deref() == Some("1");
    requested || spec_prefetch_from_env()
}

/// `HI_DSV4_SPEC_PREFETCH=1` enables speculative next-layer prefetch (5b).
fn spec_prefetch_from_env() -> bool {
    std::env::var("HI_DSV4_SPEC_PREFETCH").ok().as_deref() == Some("1")
}

/// Pinned staging capacity in slots (`HI_DSV4_PREFETCH_STAGING_SLOTS`, default
/// [`DSV4_PREFETCH_STAGING_SLOTS`]); clamped to at least 1.
fn prefetch_staging_slots() -> usize {
    std::env::var("HI_DSV4_PREFETCH_STAGING_SLOTS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&slots| slots > 0)
        .unwrap_or(DSV4_PREFETCH_STAGING_SLOTS)
}

/// Operator kinds for the `HI_DSV4_MULMAT_LOOP` numerics-debug escape hatch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MulMatKind {
    Dense,
    Grouped,
    Expert,
}

/// Parse `HI_DSV4_MULMAT_LOOP` once (comma list of `dense`/`grouped`/`expert`
/// or `all`); empty means the normal batched path everywhere.
fn mulmat_loop_kinds() -> &'static [MulMatKind] {
    use std::sync::OnceLock;
    static KINDS: OnceLock<Vec<MulMatKind>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let raw = std::env::var("HI_DSV4_MULMAT_LOOP").unwrap_or_default();
        let mut kinds = Vec::new();
        for entry in raw.split(',').map(str::trim) {
            match entry {
                "all" => {
                    return vec![MulMatKind::Dense, MulMatKind::Grouped, MulMatKind::Expert];
                }
                "dense" => kinds.push(MulMatKind::Dense),
                "grouped" => kinds.push(MulMatKind::Grouped),
                "expert" => kinds.push(MulMatKind::Expert),
                _ => {}
            }
        }
        kinds
    })
}

/// Flatten a chunk of equal-length activation rows for a single GEMM upload.
fn flatten_activations(name: &str, xs: &[Vec<f32>], cols: usize) -> Result<Vec<f32>> {
    let mut flat = Vec::with_capacity(xs.len() * cols);
    for x in xs {
        if x.len() != cols {
            bail!(
                "matmul input length {} does not match tensor {name} input dim {cols}",
                x.len()
            );
        }
        flat.extend_from_slice(x);
    }
    Ok(flat)
}

/// Split a downloaded row-major [m, rows] result into per-token vectors.
fn split_rows(flat: Vec<f32>, rows: usize) -> Vec<Vec<f32>> {
    flat.chunks(rows).map(<[f32]>::to_vec).collect()
}

fn gemm_element_size(dtype: GemmDType) -> usize {
    match dtype {
        GemmDType::F32 => 4,
        GemmDType::F16 | GemmDType::BF16 => 2,
    }
}

/// GGML type id for the generic device dequant kernel; mirrors
/// `GpuMatrix::quant_type_id` in gpu.rs (kept in sync by hand — that method is
/// private to the Qwen GPU model). `None` routes to the host-dequant fallback.
fn quant_type_id(dtype: GgufTensorType) -> Option<i32> {
    match dtype {
        GgufTensorType::MXFP4 => Some(39),
        GgufTensorType::NVFP4 => Some(40),
        GgufTensorType::Q4_0 => Some(2),
        GgufTensorType::Q4_0_4_4 => Some(31),
        GgufTensorType::Q4_0_4_8 => Some(32),
        GgufTensorType::Q4_0_8_8 => Some(33),
        GgufTensorType::Q4_1 => Some(3),
        GgufTensorType::Q1_0 => Some(41),
        GgufTensorType::Q5_0 => Some(6),
        GgufTensorType::Q5_1 => Some(7),
        GgufTensorType::Q8_0 => Some(8),
        GgufTensorType::Q8_1 => Some(9),
        GgufTensorType::IQ2_XXS => Some(16),
        GgufTensorType::IQ2_XS => Some(17),
        GgufTensorType::IQ3_XXS => Some(18),
        GgufTensorType::IQ1_S => Some(19),
        GgufTensorType::IQ2_S => Some(22),
        GgufTensorType::IQ3_S => Some(21),
        GgufTensorType::IQ4_NL => Some(20),
        GgufTensorType::IQ4_NL_4_4 => Some(36),
        GgufTensorType::IQ4_NL_4_8 => Some(37),
        GgufTensorType::IQ4_NL_8_8 => Some(38),
        GgufTensorType::IQ4_XS => Some(23),
        GgufTensorType::IQ1_M => Some(29),
        GgufTensorType::Q2_K => Some(10),
        GgufTensorType::Q3_K => Some(11),
        GgufTensorType::Q4_K => Some(12),
        GgufTensorType::Q5_K => Some(13),
        GgufTensorType::Q6_K => Some(14),
        GgufTensorType::Q8_K => Some(15),
        GgufTensorType::TQ1_0 => Some(34),
        GgufTensorType::TQ2_0 => Some(35),
        _ => None,
    }
}

// pub(crate): dsv4_backend's ignored real-model test reuses the checkpoint
// path + long-prompt helpers.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dsv4_cpu::DeepSeekV4CpuReference;
    use crate::dsv4_cpu::fixture::{tempfile_path, write_deepseek4_gguf};

    /// The Stage-1 acceptance gate: identical greedy tokens and near-identical
    /// logits between the CPU oracle and the GPU provider on the tiny fixture.
    /// The fixture is F32, so the GPU path is f32 end-to-end and only cuBLAS
    /// reduction order separates the two.
    #[test]
    fn dsv4_gpu_parity_with_cpu_reference() {
        let path = tempfile_path("gpu-parity");
        write_deepseek4_gguf(&path);

        let cpu = DeepSeekV4CpuReference::load(&path).unwrap();
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        // The fixture's experts are F32, so the pool (MXFP4-only) must stay
        // out of the way and the streaming fallback must serve everything.
        assert!(gpu.pool_stats().is_none());

        // Same 3-token prompt + 3-token greedy generation on both providers.
        let options = QwenCpuRunOptions {
            max_tokens: 3,
            top_k: 3,
            include_logits: true,
            ..QwenCpuRunOptions::default()
        };
        let cpu_out = cpu.run_tokens(&[0, 1, 2], options.clone()).unwrap();
        let gpu_out = gpu.run_tokens(&[0, 1, 2], options).unwrap();

        assert_eq!(cpu_out.backend, "cpu-reference");
        assert_eq!(gpu_out.backend, "cuda-dsv4");
        assert_eq!(cpu_out.next_token, gpu_out.next_token);
        assert_eq!(cpu_out.generated_tokens, gpu_out.generated_tokens);
        assert_max_abs_diff(
            cpu_out.logits.as_ref().unwrap(),
            gpu_out.logits.as_ref().unwrap(),
            "prompt logits",
        );

        // Final logits of the full greedy continuation (prompt + generated).
        let mut sequence = vec![0u32, 1, 2];
        sequence.extend(&cpu_out.generated_tokens);
        assert_max_abs_diff(
            &cpu.last_logits(&sequence).unwrap(),
            &gpu.last_logits(&sequence).unwrap(),
            "post-generation logits",
        );

        // Long sequence: ring eviction (window 4), both compressor forms
        // (split ratio-4 and shared-K=V ratio-2), and indexer top-k gathering
        // all engage by token 14.
        let long: Vec<u32> = (0..14).map(|idx| idx % 3).collect();
        assert_max_abs_diff(
            &cpu.last_logits(&long).unwrap(),
            &gpu.last_logits(&long).unwrap(),
            "long-sequence logits",
        );
    }

    fn assert_max_abs_diff(cpu: &[f32], gpu: &[f32], what: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{what}: length mismatch");
        for (idx, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (c - g).abs() < 1.0e-3,
                "{what}[{idx}]: cpu {c} vs gpu {g} exceeds 1e-3"
            );
        }
    }

    // ---- Stage-1b expert pool tests (packed MXFP4 slices) -----------------
    //
    // The engine fixture is F32, so the pool path never engages there. These
    // tests fabricate a minimal GGUF holding only packed rank-3 MXFP4 expert
    // tensors and drive the provider's expert path directly through mul_vec,
    // checking every result against a host dequant + f32 matmul reference.

    /// Gate/up [in=32, out=64] and down [in=64, out=32] with `experts` slices
    /// each: equal element counts per slice (2048 -> 1088 packed bytes),
    /// mirroring the real model's uniform-slot property. Random nibbles with
    /// every block scale pinned to 0.5 keep values in ±6.
    fn write_mxfp4_experts_gguf(path: &std::path::Path, experts: u64) {
        write_mxfp4_experts_gguf_layers(path, experts, 1);
    }

    /// [`write_mxfp4_experts_gguf`] with `layers` MoE layers (`blk.0..layers`),
    /// for the speculative next-layer prefetch test.
    fn write_mxfp4_experts_gguf_layers(path: &std::path::Path, experts: u64, layers: u64) {
        use rand::RngCore;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(0xd5f4);
        let mut specs: Vec<(String, u64, u64)> = Vec::new();
        for layer in 0..layers {
            specs.push((format!("blk.{layer}.ffn_gate_exps.weight"), 32, 64));
            specs.push((format!("blk.{layer}.ffn_up_exps.weight"), 32, 64));
            specs.push((format!("blk.{layer}.ffn_down_exps.weight"), 64, 32));
        }
        let mut infos = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, in_dim, out_dim) in &specs {
            let blocks = (in_dim * out_dim * experts / 32) as usize;
            let mut bytes = vec![0u8; blocks * 17];
            rng.fill_bytes(&mut bytes);
            for block in 0..blocks {
                bytes[block * 17] = 127; // e8m0 scale byte for 0.5
            }
            while !data.len().is_multiple_of(32) {
                data.push(0);
            }
            infos.push((
                name.clone(),
                [*in_dim, *out_dim, experts],
                data.len() as u64,
            ));
            data.extend_from_slice(&bytes);
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(infos.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        // Single metadata entry: general.alignment = 32 (value type 4 = u32).
        let key = "general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        for (name, dims, offset) in &infos {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&3u32.to_le_bytes());
            for dim in dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&39u32.to_le_bytes()); // MXFP4
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    /// Packed bytes of one expert slice of the test tensors: 32*64 elements
    /// = 64 MXFP4 blocks of 17 bytes.
    const TEST_EXPERT_SLICE_BYTES: u64 = 32 * 64 / 32 * 17;

    fn pool_test_linear(path: &std::path::Path) -> (Arc<GgufFile>, DsV4GpuLinear) {
        CudaRuntime::probe().unwrap();
        let gguf = Arc::new(GgufFile::open(path).unwrap());
        let linear = DsV4GpuLinear::new(gguf.clone()).unwrap();
        (gguf, linear)
    }

    fn raw_experts(name: &str, in_dim: usize, out_dim: usize) -> RawExperts {
        RawExperts {
            name: name.to_string(),
            in_dim,
            out_dim,
        }
    }

    /// Deterministic activation in roughly [-1, 1].
    fn test_activation(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let mixed = seed
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(idx as u32)
                    .wrapping_mul(0x85eb_ca6b);
                ((mixed >> 9) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// Host oracle: dequantize the expert slice and run the row-major f32
    /// matvec the GEMV must match.
    fn host_expert_reference(
        gguf: &GgufFile,
        name: &str,
        expert: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let view = gguf.tensor(name).unwrap();
        let per_expert = in_dim * out_dim;
        let weights = dequantize_elem_range(&view, expert * per_expert, per_expert).unwrap();
        (0..out_dim)
            .map(|row| {
                weights[row * in_dim..(row + 1) * in_dim]
                    .iter()
                    .zip(x)
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect()
    }

    #[track_caller]
    fn assert_close_relative(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (idx, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1.0e-2 * w.abs().max(0.5);
            assert!(
                (g - w).abs() <= tol,
                "{what}[{idx}]: gpu {g} vs host {w} exceeds {tol}"
            );
        }
    }

    /// Runs one pooled expert matvec and checks it against the host oracle.
    fn check_expert_matvec(
        gguf: &GgufFile,
        linear: &DsV4GpuLinear,
        name: &str,
        in_dim: usize,
        out_dim: usize,
        expert: usize,
        seed: u32,
    ) {
        let experts = raw_experts(name, in_dim, out_dim);
        let x = test_activation(in_dim, seed);
        let got = linear
            .mul_vec(
                TensorKey::Expert {
                    experts: &experts,
                    expert,
                },
                &x,
            )
            .unwrap();
        let want = host_expert_reference(gguf, name, expert, in_dim, out_dim, &x);
        assert_close_relative(&got, &want, &format!("{name}[{expert}]"));
    }

    /// Pool path: cold sweep misses once per slice, warm sweep hits every
    /// slice, and every GEMV (gate/up/down projections alike) matches the
    /// host dequant + matmul reference.
    #[test]
    fn dsv4_gpu_expert_pool_hit_miss_accounting_and_parity() {
        let path = tempfile_path("mxfp4-pool");
        write_mxfp4_experts_gguf(&path, 3);
        let (gguf, linear) = pool_test_linear(&path);
        // 9 slices total; a 64 KiB budget caps at the 9 existing slices.
        linear.init_expert_pool_with_budget(64 << 10).unwrap();
        let stats = linear.pool_stats().expect("pool must be enabled");
        assert_eq!((stats.hits, stats.misses), (0, 0));

        let tensors: [(&str, usize, usize); 3] = [
            ("blk.0.ffn_gate_exps.weight", 32, 64),
            ("blk.0.ffn_up_exps.weight", 32, 64),
            ("blk.0.ffn_down_exps.weight", 64, 32),
        ];
        for _sweep in 0..2 {
            for (name, in_dim, out_dim) in tensors {
                for expert in 0..3usize {
                    // Same seed both sweeps: identical inputs, so a stale or
                    // misindexed slot in the warm sweep would be caught.
                    let seed = expert as u32 + 7;
                    check_expert_matvec(&gguf, &linear, name, in_dim, out_dim, expert, seed);
                }
            }
        }
        let stats = linear.pool_stats().unwrap();
        assert_eq!(
            (stats.hits, stats.misses, stats.evictions, stats.prefilled),
            (9, 9, 0, 0),
            "cold sweep must miss all 9 slices, warm sweep must hit all 9"
        );
        assert_eq!(stats.bytes_uploaded, 9 * TEST_EXPERT_SLICE_BYTES);
    }

    /// `HI_DSV4_EXPERT_PREFILL_POOL`-style warmup (driven directly, not via
    /// env): every slot loads at init and the first demand touch already hits.
    #[test]
    fn dsv4_gpu_expert_pool_prefill_hits_first_touch() {
        let path = tempfile_path("mxfp4-prefill");
        write_mxfp4_experts_gguf(&path, 3);
        let (gguf, linear) = pool_test_linear(&path);
        linear.init_expert_pool_with_budget(64 << 10).unwrap();
        linear.prefill_expert_pool().unwrap();

        let stats = linear.pool_stats().unwrap();
        assert_eq!((stats.prefilled, stats.hits, stats.misses), (9, 0, 0));
        assert_eq!(stats.bytes_uploaded, 9 * TEST_EXPERT_SLICE_BYTES);

        check_expert_matvec(&gguf, &linear, "blk.0.ffn_down_exps.weight", 64, 32, 2, 11);
        let stats = linear.pool_stats().unwrap();
        assert_eq!((stats.hits, stats.misses), (1, 0));
        assert_eq!(stats.bytes_uploaded, 9 * TEST_EXPERT_SLICE_BYTES);
    }

    /// Craft one packed MXFP4 host blob shaped like the test tensors (random
    /// nibbles, every block scale 0.5), plus its expert count.
    fn host_expert_blob(in_dim: usize, out_dim: usize, experts: usize, seed: u64) -> Vec<u8> {
        use rand::RngCore;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let blocks = in_dim * out_dim * experts / 32;
        let mut bytes = vec![0u8; blocks * 17];
        rng.fill_bytes(&mut bytes);
        for block in 0..blocks {
            bytes[block * 17] = 127;
        }
        bytes
    }

    /// Host oracle over a raw packed blob (the registered MTP source).
    fn host_blob_reference(
        bytes: &[u8],
        expert: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let per_expert = in_dim * out_dim;
        let slice_bytes = per_expert / 32 * 17;
        let weights = hi_gguf::dequantize_tensor_as_f32(
            &bytes[expert * slice_bytes..(expert + 1) * slice_bytes],
            GgufTensorType::MXFP4,
            per_expert,
        )
        .unwrap();
        (0..out_dim)
            .map(|row| {
                weights[row * in_dim..(row + 1) * in_dim]
                    .iter()
                    .zip(x)
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect()
    }

    /// Stage-B pool extension: host-blob expert tensors register alongside
    /// the GGUF-mmap trunk tensors (here as layer 9, the MTP pattern), their
    /// pinned slots survive arbitrary trunk-driven eviction pressure, and
    /// every GEMV against them matches the host dequant reference.
    #[test]
    fn dsv4_gpu_expert_pool_host_blobs_register_and_stay_pinned() {
        let path = tempfile_path("mxfp4-pinned");
        // 120 trunk slices vs 80 slots (12 of them pinned): heavy eviction.
        write_mxfp4_experts_gguf(&path, 40);
        let (gguf, linear) = pool_test_linear(&path);
        let slot_stride = 1280; // 1088-byte slices aligned up to 256
        linear
            .init_expert_pool_with_budget(80 * slot_stride)
            .unwrap();

        let blobs: [(&str, usize, usize, u8); 3] = [
            ("blk.9.ffn_gate_exps.weight", 32, 64, 0),
            ("blk.9.ffn_up_exps.weight", 32, 64, 1),
            ("blk.9.ffn_down_exps.weight", 64, 32, 2),
        ];
        let mut payloads = Vec::new();
        for (name, in_dim, out_dim, proj) in blobs {
            let bytes = host_expert_blob(in_dim, out_dim, 4, 0x517 + proj as u64);
            let pinned = linear
                .register_host_experts(
                    &raw_experts(name, in_dim, out_dim),
                    4,
                    9,
                    proj,
                    GgufTensorType::MXFP4,
                    bytes.clone(),
                    true,
                )
                .unwrap();
            assert!(pinned, "{name} must pin (pool has ample headroom)");
            payloads.push(bytes);
        }
        let stats = linear.pool_stats().unwrap();
        assert_eq!(stats.prefilled, 12, "registration preloads every slice");

        // Duplicate registration must fail loudly.
        let err = linear
            .register_host_experts(
                &raw_experts("blk.9.ffn_gate_exps.weight", 32, 64),
                4,
                9,
                0,
                GgufTensorType::MXFP4,
                payloads[0].clone(),
                true,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("already registered"), "{err}");

        // Blob GEMVs match the host reference and hit their pinned slots.
        let before = linear.pool_stats().unwrap();
        for ((name, in_dim, out_dim, _), bytes) in blobs.iter().zip(&payloads) {
            for expert in 0..4usize {
                let x = test_activation(*in_dim, 31 + expert as u32);
                let got = linear
                    .mul_vec(
                        TensorKey::Expert {
                            experts: &raw_experts(name, *in_dim, *out_dim),
                            expert,
                        },
                        &x,
                    )
                    .unwrap();
                let want = host_blob_reference(bytes, expert, *in_dim, *out_dim, &x);
                assert_close_relative(&got, &want, &format!("{name}[{expert}]"));
            }
        }
        let after = linear.pool_stats().unwrap();
        assert_eq!(after.hits - before.hits, 12, "pinned slices must hit");
        assert_eq!(after.misses, before.misses);

        // Trunk-driven eviction pressure: 120 slices through 68 free slots,
        // twice. The pinned layer-9 slots must never be victims.
        for sweep in 0..2 {
            for (name, in_dim, out_dim) in [
                ("blk.0.ffn_gate_exps.weight", 32usize, 64usize),
                ("blk.0.ffn_up_exps.weight", 32, 64),
                ("blk.0.ffn_down_exps.weight", 64, 32),
            ] {
                for expert in 0..40usize {
                    let x = test_activation(in_dim, 100 + sweep + expert as u32);
                    let experts = raw_experts(name, in_dim, out_dim);
                    linear
                        .mul_vec(
                            TensorKey::Expert {
                                experts: &experts,
                                expert,
                            },
                            &x,
                        )
                        .unwrap();
                }
            }
        }
        let pressured = linear.pool_stats().unwrap();
        assert!(
            pressured.evictions > 0,
            "the sweep must overflow the unpinned slots"
        );

        // After the storm: every blob slice is STILL resident (hits, no new
        // misses) and still bit-faithful to the host reference.
        let before = linear.pool_stats().unwrap();
        for ((name, in_dim, out_dim, _), bytes) in blobs.iter().zip(&payloads) {
            for expert in 0..4usize {
                let x = test_activation(*in_dim, 31 + expert as u32);
                let got = linear
                    .mul_vec(
                        TensorKey::Expert {
                            experts: &raw_experts(name, *in_dim, *out_dim),
                            expert,
                        },
                        &x,
                    )
                    .unwrap();
                let want = host_blob_reference(bytes, expert, *in_dim, *out_dim, &x);
                assert_close_relative(&got, &want, &format!("{name}[{expert}] post-pressure"));
            }
        }
        let after = linear.pool_stats().unwrap();
        assert_eq!(
            after.hits - before.hits,
            12,
            "pinned slices must survive eviction pressure"
        );
        assert_eq!(after.misses, before.misses, "no pinned slice may reload");

        // A pool without pinning headroom declines to pin but keeps serving.
        let tiny_path = tempfile_path("mxfp4-pinned-tiny");
        write_mxfp4_experts_gguf(&tiny_path, 3);
        let (_tiny_gguf, tiny_linear) = pool_test_linear(&tiny_path);
        tiny_linear
            .init_expert_pool_with_budget(8 * slot_stride)
            .unwrap();
        let bytes = host_expert_blob(32, 64, 4, 0x99);
        let pinned = tiny_linear
            .register_host_experts(
                &raw_experts("blk.9.ffn_gate_exps.weight", 32, 64),
                4,
                9,
                0,
                GgufTensorType::MXFP4,
                bytes.clone(),
                true,
            )
            .unwrap();
        assert!(!pinned, "an 8-slot pool must refuse to pin 4 slices");
        let x = test_activation(32, 5);
        let got = tiny_linear
            .mul_vec(
                TensorKey::Expert {
                    experts: &raw_experts("blk.9.ffn_gate_exps.weight", 32, 64),
                    expert: 1,
                },
                &x,
            )
            .unwrap();
        let want = host_blob_reference(&bytes, 1, 32, 64, &x);
        assert_close_relative(&got, &want, "unpinned blob GEMV");
    }

    /// Parity gate (a) on the GPU provider: chunked prefill (B=4 and a whole-
    /// prompt chunk) must match the sequential path within 1e-4 on the final
    /// logits and produce the identical greedy continuation — in the default
    /// bit-exact mode (per-token loops: identical invocations) AND in the
    /// opt-in GEMM mode (only cuBLAS GEMM-vs-GEMV reduction order separates
    /// the two on the F32 fixture).
    #[test]
    fn dsv4_gpu_chunked_prefill_matches_sequential() {
        let path = tempfile_path("gpu-chunked");
        write_deepseek4_gguf(&path);
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();

        // 14 tokens engage ring eviction, both compressor forms, and the
        // indexer's top-k narrowing on the tiny fixture.
        let tokens: Vec<u32> = (0..14).map(|idx| idx % 3).collect();
        let mut sequential_state = engine.new_state();
        let sequential = engine
            .prefill_with_chunk(&mut sequential_state, &tokens, 1)
            .unwrap();

        for gemm_batching in [false, true] {
            engine.linear().set_gemm_batching(gemm_batching);
            for chunk in [4, 64] {
                let mut state = engine.new_state();
                let chunked = engine
                    .prefill_with_chunk(&mut state, &tokens, chunk)
                    .unwrap();
                for (idx, (seq, chk)) in sequential.iter().zip(&chunked).enumerate() {
                    assert!(
                        (seq - chk).abs() <= 1.0e-4,
                        "gemm {gemm_batching} chunk {chunk} logit[{idx}]: sequential {seq} vs chunked {chk}"
                    );
                }
                let mut seq_state = sequential_state.clone();
                let mut seq_logits = sequential.clone();
                let mut chk_logits = chunked;
                for step in 0..4 {
                    let seq_next = crate::qwen_cpu::argmax(&seq_logits).unwrap();
                    let chk_next = crate::qwen_cpu::argmax(&chk_logits).unwrap();
                    assert_eq!(
                        seq_next, chk_next,
                        "gemm {gemm_batching} chunk {chunk} greedy step {step}"
                    );
                    seq_logits = engine.step(&mut seq_state, seq_next).unwrap();
                    chk_logits = engine.step(&mut state, chk_next).unwrap();
                }
            }
        }
        engine.linear().set_gemm_batching(false);
    }

    /// Batched expert path: a chunk's worth of tokens routed to one expert
    /// dequantizes the pooled slice once (one acquire per call) and GEMMs;
    /// every row matches the host oracle, small batches fall back to the
    /// fused GEMV path, and the pool-less provider streams the packed bytes.
    #[test]
    fn dsv4_gpu_expert_batched_mul_mat_parity_and_accounting() {
        let path = tempfile_path("mxfp4-mulmat");
        write_mxfp4_experts_gguf(&path, 3);
        let (gguf, linear) = pool_test_linear(&path);
        linear.set_gemm_batching(true);
        linear.init_expert_pool_with_budget(64 << 10).unwrap();

        let name = "blk.0.ffn_gate_exps.weight";
        let experts = raw_experts(name, 32, 64);
        let key = TensorKey::Expert {
            experts: &experts,
            expert: 1,
        };
        let xs: Vec<Vec<f32>> = (0..9).map(|row| test_activation(32, 40 + row)).collect();
        for sweep in 0..2 {
            let got = linear.mul_mat(key, &xs).unwrap();
            for (row, (out, x)) in got.iter().zip(&xs).enumerate() {
                let want = host_expert_reference(&gguf, name, 1, 32, 64, x);
                assert_close_relative(out, &want, &format!("gemm sweep {sweep} row {row}"));
            }
        }
        let stats = linear.pool_stats().unwrap();
        assert_eq!(
            (stats.hits, stats.misses, stats.evictions),
            (1, 1, 0),
            "one acquire per batched call: cold miss then warm hit"
        );
        assert_eq!(stats.bytes_uploaded, TEST_EXPERT_SLICE_BYTES);

        // Below the GEMM threshold the per-token fused GEMV path serves the
        // call (and empty input short-circuits).
        assert!(linear.mul_mat(key, &[]).unwrap().is_empty());
        let small_key = TensorKey::Expert {
            experts: &experts,
            expert: 2,
        };
        let small: Vec<Vec<f32>> = (0..2).map(|row| test_activation(32, 90 + row)).collect();
        let got = linear.mul_mat(small_key, &small).unwrap();
        for (row, (out, x)) in got.iter().zip(&small).enumerate() {
            let want = host_expert_reference(&gguf, name, 2, 32, 64, x);
            assert_close_relative(out, &want, &format!("gemv fallback row {row}"));
        }
        let stats = linear.pool_stats().unwrap();
        assert_eq!((stats.hits, stats.misses), (2, 2));

        // Pool disabled: the batched path streams the packed slice through the
        // expert scratch before dequantizing — results must be identical.
        let (gguf2, streaming) = pool_test_linear(&path);
        streaming.set_gemm_batching(true);
        assert!(streaming.pool_stats().is_none());
        let got = streaming.mul_mat(key, &xs).unwrap();
        for (row, (out, x)) in got.iter().zip(&xs).enumerate() {
            let want = host_expert_reference(&gguf2, name, 1, 32, 64, x);
            assert_close_relative(out, &want, &format!("streaming gemm row {row}"));
        }
    }

    /// Two slots, three experts round-robin: every touch after the first two
    /// evicts the LRU slice, each re-upload must still compute correctly, and
    /// the most recently used slice stays resident.
    #[test]
    fn dsv4_gpu_expert_pool_eviction_round_robin_stays_correct() {
        let path = tempfile_path("mxfp4-evict");
        write_mxfp4_experts_gguf(&path, 3);
        let (gguf, linear) = pool_test_linear(&path);
        // Slot stride is the 1088-byte slice rounded up to 256-byte alignment.
        let slot_stride = (TEST_EXPERT_SLICE_BYTES as usize).div_ceil(256) * 256;
        linear
            .init_expert_pool_with_budget(2 * slot_stride)
            .unwrap();

        let name = "blk.0.ffn_gate_exps.weight";
        for round in 0..2u32 {
            for expert in 0..3usize {
                let seed = 31 + round * 3 + expert as u32;
                check_expert_matvec(&gguf, &linear, name, 32, 64, expert, seed);
            }
        }
        let stats = linear.pool_stats().unwrap();
        assert_eq!(
            (stats.hits, stats.misses, stats.evictions),
            (0, 6, 4),
            "2-slot pool over e0,e1,e2 round-robin thrashes: first two misses fill, the rest evict"
        );
        assert_eq!(stats.bytes_uploaded, 6 * TEST_EXPERT_SLICE_BYTES);

        // Expert 2 was touched last; it must still be resident.
        check_expert_matvec(&gguf, &linear, name, 32, 64, 2, 99);
        let stats = linear.pool_stats().unwrap();
        assert_eq!((stats.hits, stats.misses, stats.evictions), (1, 6, 4));
    }

    // ---- Copy-stream expert prefetch tests (roadmap item 5) ----------------

    /// The fabricated fixture's three layer-0 expert-tensor handles.
    fn mxfp4_layer0_tensors() -> (RawExperts, RawExperts, RawExperts) {
        (
            raw_experts("blk.0.ffn_gate_exps.weight", 32, 64),
            raw_experts("blk.0.ffn_up_exps.weight", 32, 64),
            raw_experts("blk.0.ffn_down_exps.weight", 64, 32),
        )
    }

    /// (i) Cross-stream ordering correctness: a layer prefetched on the copy
    /// stream then consumed by demand GEMVs yields results identical to the
    /// synchronous (`HI_DSV4_NO_COPY_STREAM`) path and to the host oracle, cold
    /// (all prefetch misses) and warm (all prefetch hits). This is the core
    /// guarantee — the copy-done event must gate every expert GEMV behind its
    /// slice's H2D copy.
    #[test]
    fn dsv4_gpu_expert_prefetch_matches_sync_and_host() {
        let path = tempfile_path("mxfp4-prefetch");
        write_mxfp4_experts_gguf(&path, 4);

        let (gguf, mut async_lin) = pool_test_linear(&path);
        async_lin.set_copy_stream_enabled(true).unwrap();
        async_lin.init_expert_pool_with_budget(1 << 20).unwrap(); // fits all 12 slices

        let (_g2, mut sync_lin) = pool_test_linear(&path);
        sync_lin.set_copy_stream_enabled(false).unwrap();
        sync_lin.init_expert_pool_with_budget(1 << 20).unwrap();

        let (gate, up, down) = mxfp4_layer0_tensors();
        let ids = [0usize, 1, 2, 3];
        let projs: [(&RawExperts, usize, usize); 3] =
            [(&gate, 32, 64), (&up, 32, 64), (&down, 64, 32)];

        for sweep in 0..2u32 {
            async_lin
                .prefetch_experts(
                    DsV4ExpertTensors {
                        gate: &gate,
                        up: &up,
                        down: &down,
                    },
                    &ids,
                )
                .unwrap();
            for &(experts, in_dim, out_dim) in &projs {
                for &e in &ids {
                    let x = test_activation(in_dim, e as u32 + sweep * 13 + 5);
                    let key = TensorKey::Expert { experts, expert: e };
                    let got_async = async_lin.mul_vec(key, &x).unwrap();
                    let got_sync = sync_lin.mul_vec(key, &x).unwrap();
                    let want = host_expert_reference(&gguf, &experts.name, e, in_dim, out_dim, &x);
                    assert_close_relative(
                        &got_async,
                        &want,
                        &format!("async {} e{e} sweep {sweep}", experts.name),
                    );
                    assert_eq!(
                        got_async, got_sync,
                        "async and sync paths must be identical for {} e{e} sweep {sweep}",
                        experts.name
                    );
                }
            }
        }
        let stats = async_lin.pool_stats().unwrap();
        assert_eq!(
            stats.prefetch_uploads, 12,
            "12 slices prefetched once; the warm sweep re-hits them"
        );
        assert_eq!(stats.misses, 0, "prefetch must absorb every demand miss");
        assert!(stats.hits >= 24, "both sweeps' demand GEMVs hit the pool");
    }

    /// (ii) LRU protection of in-flight slices, at the pool level: a slice being
    /// copied is pinned, so a later same-batch prefetch cannot evict it; only a
    /// demand consume (or the next batch) releases the pin.
    #[test]
    fn dsv4_expert_pool_prefetch_pins_block_eviction() {
        let path = tempfile_path("mxfp4-pin");
        write_mxfp4_experts_gguf(&path, 3);
        let (_g, mut lin) = pool_test_linear(&path);
        lin.set_copy_stream_enabled(true).unwrap();
        let slot_stride = (TEST_EXPERT_SLICE_BYTES as usize).div_ceil(256) * 256;
        lin.init_expert_pool_with_budget(2 * slot_stride).unwrap(); // 2 slots

        let mut guard = lin.expert_pool.borrow_mut();
        let pool = guard.as_mut().expect("pool enabled");

        // Two distinct slices prefetch-acquire into the two slots, both pinned.
        assert!(matches!(
            pool.acquire_prefetch((0, 0, 0)),
            PrefetchAcquire::Loaded(_)
        ));
        assert!(matches!(
            pool.acquire_prefetch((0, 0, 1)),
            PrefetchAcquire::Loaded(_)
        ));
        // A third slice cannot be prefetched: both slots are pinned in-flight.
        assert!(matches!(
            pool.acquire_prefetch((0, 0, 2)),
            PrefetchAcquire::Full
        ));
        assert!(pool.find_victim().is_none(), "every slot is pinned");

        // A demand GEMV consumes slice 0, unpinning its slot.
        let (_slot, hit) = pool.acquire((0, 0, 0));
        assert!(hit);
        assert!(
            pool.find_victim().is_some(),
            "consumed slot is now evictable"
        );
        // The third slice now fits (evicting the freed slot, not the pinned one).
        assert!(matches!(
            pool.acquire_prefetch((0, 0, 2)),
            PrefetchAcquire::Loaded(_)
        ));

        // The next batch releases all pins.
        pool.clear_pins();
        assert!(pool.find_victim().is_some());
    }

    /// (ii, end-to-end) A layer prefetched into a pool too small to hold it: the
    /// pins protect the two in-flight slices, the overflow returns from the
    /// prefetch as `Full` and demand-loads instead, and every GEMV still matches
    /// the host oracle.
    #[test]
    fn dsv4_gpu_expert_prefetch_tiny_pool_stays_correct() {
        let path = tempfile_path("mxfp4-prefetch-tiny");
        write_mxfp4_experts_gguf(&path, 3);
        let (gguf, mut lin) = pool_test_linear(&path);
        lin.set_copy_stream_enabled(true).unwrap();
        let slot_stride = (TEST_EXPERT_SLICE_BYTES as usize).div_ceil(256) * 256;
        lin.init_expert_pool_with_budget(2 * slot_stride).unwrap(); // 2 slots

        let (gate, up, down) = mxfp4_layer0_tensors();
        let ids = [0usize, 1, 2];
        lin.prefetch_experts(
            DsV4ExpertTensors {
                gate: &gate,
                up: &up,
                down: &down,
            },
            &ids,
        )
        .unwrap();
        assert_eq!(
            lin.pool_stats().unwrap().prefetch_uploads,
            2,
            "only two of the nine slices fit; the rest fall to demand"
        );

        // Demand every slice in the same order the expert loop would; the pinned
        // in-flight slices survived, so all results match the host oracle.
        let projs: [(&RawExperts, usize, usize); 3] =
            [(&gate, 32, 64), (&up, 32, 64), (&down, 64, 32)];
        for &e in &ids {
            for &(experts, in_dim, out_dim) in &projs {
                let x = test_activation(in_dim, e as u32 + 70);
                let got = lin
                    .mul_vec(TensorKey::Expert { experts, expert: e }, &x)
                    .unwrap();
                let want = host_expert_reference(&gguf, &experts.name, e, in_dim, out_dim, &x);
                assert_close_relative(&got, &want, &format!("{} e{e}", experts.name));
            }
        }
    }

    /// (b) Speculative next-layer prefetch: routing layer 0 warms the same
    /// expert ids for layer 1; when layer 1 actually routes to them the slices
    /// are already resident, retired as speculative hits with no new upload.
    #[test]
    fn dsv4_gpu_expert_prefetch_speculative_warms_next_layer() {
        let path = tempfile_path("mxfp4-spec");
        write_mxfp4_experts_gguf_layers(&path, 4, 2);
        let (_g, mut lin) = pool_test_linear(&path);
        lin.set_copy_stream_enabled(true).unwrap();
        lin.set_spec_prefetch(true);
        lin.init_expert_pool_with_budget(1 << 20).unwrap(); // fits both layers

        let l0 = mxfp4_layer0_tensors();
        let l1 = (
            raw_experts("blk.1.ffn_gate_exps.weight", 32, 64),
            raw_experts("blk.1.ffn_up_exps.weight", 32, 64),
            raw_experts("blk.1.ffn_down_exps.weight", 64, 32),
        );
        let ids = [0usize, 1];

        // Layer 0: within-layer prefetch of blk.0 (6 slices) + speculative warm
        // of blk.1 for the same ids (6 slices).
        lin.prefetch_experts(
            DsV4ExpertTensors {
                gate: &l0.0,
                up: &l0.1,
                down: &l0.2,
            },
            &ids,
        )
        .unwrap();
        let s = lin.pool_stats().unwrap();
        assert_eq!(s.prefetch_uploads, 6, "blk.0: 2 experts x 3 projections");
        assert_eq!(s.spec_uploads, 6, "blk.1 warmed speculatively");
        assert_eq!(s.spec_hits, 0);

        // Layer 1 routes to the SAME ids: every speculative slice is resident.
        lin.prefetch_experts(
            DsV4ExpertTensors {
                gate: &l1.0,
                up: &l1.1,
                down: &l1.2,
            },
            &ids,
        )
        .unwrap();
        let s = lin.pool_stats().unwrap();
        assert_eq!(
            s.spec_hits, 6,
            "every speculative slice was reused next layer"
        );
        assert_eq!(
            s.prefetch_uploads, 6,
            "blk.1's within-layer prefetch hit the speculative slices — no new upload"
        );
    }

    /// Kernel-level gate for the indexer top-k selection: across block counts
    /// spanning the decode range (just past top_k up to 16k-token contexts)
    /// and score distributions WITH duplicates, the device selection must
    /// equal the host `indexer_select_math` ranking semantics exactly
    /// (descending total_cmp, LOWER index on ties, ascending output), and one
    /// selection must stay microseconds-scale — the serving regression gate
    /// (the original serial kernel cost ~10ms per layer per token at 4.6k
    /// context, which stacked to seconds per decoded token).
    #[test]
    fn dsv4_gpu_indexer_select_matches_host_and_stays_fast() {
        CudaRuntime::probe().unwrap();
        let stream = Stream::create().unwrap();

        // Host reference: the exact ranking dsv4_cpu::indexer_select_math
        // applies to its scores (sort by total_cmp desc, index asc; take k;
        // ascending output).
        let host_select = |scores: &[f32], k: usize| -> Vec<i32> {
            let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            ranked.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let mut selected: Vec<i32> = ranked
                .into_iter()
                .take(k)
                .map(|(block, _)| block as i32)
                .collect();
            selected.sort_unstable();
            selected
        };

        for &(n, k) in &[
            (3usize, 1usize),
            (513, 512),
            (700, 512),
            (1162, 512), // the 4.6k-token production shape
            (4096, 512),
        ] {
            // Deterministic scores with heavy duplication (quantized to 64
            // levels) so the tie-break path is exercised hard, plus zeros.
            let scores_host: Vec<f32> = (0..n)
                .map(|idx| {
                    let mixed = (idx as u32)
                        .wrapping_mul(0x9e37_79b9)
                        .wrapping_add(0x85eb_ca6b);
                    ((mixed >> 8) % 64) as f32 / 64.0
                })
                .collect();
            let scores = DeviceBuffer::alloc(n * 4).unwrap();
            scores.copy_from_host(&scores_host).unwrap();
            let marks = DeviceBuffer::alloc(n).unwrap();
            let sel = DeviceBuffer::alloc(k.min(n) * 4).unwrap();
            crate::kernels::launch_dsv4_indexer_select(&scores, n, k, &marks, &sel, &stream)
                .unwrap();
            let got: Vec<i32> = sel.copy_to_host(k.min(n)).unwrap();
            let want = host_select(&scores_host, k);
            assert_eq!(got, want, "selection mismatch at n={n} k={k}");

            // Timing: 21 launches = one decode step's worth of indexer layers.
            stream.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..21 {
                crate::kernels::launch_dsv4_indexer_select(&scores, n, k, &marks, &sel, &stream)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let elapsed = started.elapsed();
            eprintln!("dsv4 indexer select n={n} k={k}: 21 layers in {elapsed:?}");
            if n == 1162 {
                assert!(
                    elapsed.as_millis() < 50,
                    "one decode step's indexer selections took {elapsed:?} (regression: must be far below the ~280ms serial baseline)"
                );
            }
        }
    }

    // ---- Wave-2 Stage 2a: device MoE block ---------------------------------

    /// The device MoE block's parity rests on the kernels' glibc expf/logf
    /// ports being bit-identical to the host libm (selection ordering, mixture
    /// weights, and the SwiGLU all run through them). Sweep every port across
    /// strided samples of the full f32 bit space in its live range (plus the
    /// special-case boundaries) and require BIT equality with the host
    /// functions the engine actually calls.
    #[test]
    fn dsv4_gpu_exact_math_matches_host() {
        CudaRuntime::probe().unwrap();
        let stream = Stream::create().unwrap();

        // Strided walk of the positive bit space up to `hi`, mirrored to
        // negatives, plus the special-case boundary values.
        let sweep = |hi_bits: u32, stride: u32, negatives: bool| -> Vec<f32> {
            let mut out = Vec::new();
            let mut bits = 1u32;
            while bits <= hi_bits {
                out.push(f32::from_bits(bits));
                if negatives {
                    out.push(f32::from_bits(bits | 0x8000_0000));
                }
                bits += stride;
            }
            out.extend([
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                1.0,
                -1.0,
                20.0,
                -20.0,
                f32::from_bits(0x42b17217), // expf overflow boundary
                f32::from_bits(0x42b17218),
                -f32::from_bits(0x42cff1b4), // expf underflow boundary
                -f32::from_bits(0x42cff1b5),
                -f32::from_bits(0x42ce8ecf), // expf may-underflow boundary
                -f32::from_bits(0x42ce8ed0),
            ]);
            out
        };
        let check = |op: u32, inputs: &[f32], host: &dyn Fn(f32) -> f32, label: &str| {
            let input = DeviceBuffer::alloc(std::mem::size_of_val(inputs)).unwrap();
            input.copy_from_host(inputs).unwrap();
            let output = DeviceBuffer::alloc(std::mem::size_of_val(inputs)).unwrap();
            crate::kernels::launch_dsv4_exact_math(&input, &output, inputs.len(), op, &stream)
                .unwrap();
            let got: Vec<f32> = output.copy_to_host(inputs.len()).unwrap();
            let mut mismatches = 0usize;
            let mut first = None;
            for (&x, &g) in inputs.iter().zip(&got) {
                let want = host(x);
                if g.to_bits() != want.to_bits() && !(g.is_nan() && want.is_nan()) {
                    mismatches += 1;
                    first.get_or_insert((x, g, want));
                }
            }
            assert_eq!(
                mismatches,
                0,
                "{label}: {mismatches} of {} results differ from host bits; first: x={:?} device={:?} host={:?}",
                inputs.len(),
                first.map(|f| f.0),
                first.map(|f| f.1),
                first.map(|f| f.2),
            );
        };

        // exp/silu live on (-inf, ~88.7]; 90.0f = 0x42b40000 caps the sweep.
        let exp_range = sweep(0x42b4_0000, 331, true);
        check(0, &exp_range, &|x| x.exp(), "expf");
        check(3, &exp_range, &crate::qwen_cpu::silu, "silu");
        check(2, &exp_range, &crate::qwen_cpu::softplus, "softplus");
        // log sweeps every positive exponent (subnormals included) + specials.
        let log_range = sweep(0x7f7f_ffff, 331, true);
        check(1, &log_range, &|x| x.ln(), "logf");
    }

    /// Fabricated-MXFP4 fixture for the full device MoE block: packed expert
    /// tensors (gate/up [32 -> 64], down [64 -> 32], 4 experts) plus an F32
    /// router [32 -> 4] and F32 shared-expert mats [32 -> 64 -> 32], all with
    /// deterministic payloads.
    fn write_mxfp4_moe_gguf(path: &std::path::Path) {
        use rand::RngCore;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(0xd5f4_20a);
        struct Spec {
            name: &'static str,
            dims: Vec<u64>,
            dtype: u32,
            bytes: Vec<u8>,
        }
        let mut mxfp4 = |name: &'static str, in_dim: u64, out_dim: u64| -> Spec {
            let blocks = (in_dim * out_dim * 4 / 32) as usize;
            let mut bytes = vec![0u8; blocks * 17];
            rng.fill_bytes(&mut bytes);
            for block in 0..blocks {
                bytes[block * 17] = 127; // e8m0 scale 0.5 keeps values in ±6
            }
            Spec {
                name,
                dims: vec![in_dim, out_dim, 4],
                dtype: 39,
                bytes,
            }
        };
        let f32_tensor = |name: &'static str, dims: Vec<u64>, seed: u32| -> Spec {
            let count = dims.iter().product::<u64>() as usize;
            let values = test_activation(count, seed);
            Spec {
                name,
                dims,
                dtype: 0,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }
        };
        let specs = vec![
            mxfp4("blk.0.ffn_gate_exps.weight", 32, 64),
            mxfp4("blk.0.ffn_up_exps.weight", 32, 64),
            mxfp4("blk.0.ffn_down_exps.weight", 64, 32),
            f32_tensor("blk.0.ffn_gate_inp.weight", vec![32, 4], 501),
            f32_tensor("blk.0.ffn_gate_shexp.weight", vec![32, 64], 502),
            f32_tensor("blk.0.ffn_up_shexp.weight", vec![32, 64], 503),
            f32_tensor("blk.0.ffn_down_shexp.weight", vec![64, 32], 504),
        ];

        let mut data: Vec<u8> = Vec::new();
        let mut infos = Vec::new();
        for spec in &specs {
            while !data.len().is_multiple_of(32) {
                data.push(0);
            }
            infos.push((spec.name, spec.dims.clone(), spec.dtype, data.len() as u64));
            data.extend_from_slice(&spec.bytes);
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(infos.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        for (name, dims, dtype, offset) in &infos {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&dtype.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    /// Stage-2a fixture gate: the full device MoE block (device routing +
    /// selection, pooled fused GEMVs, SwiGLU clamp, weighted accumulate,
    /// shared expert) against (a) the host path on the SAME provider — the
    /// `HI_DSV4_NO_DEVICE_MOE=1` A/B, expected bit-identical, gated at 1e-4 —
    /// and (b) a pure-CPU host_moe_block oracle over the same GGUF. Covers
    /// learned routing with selection bias AND hash-table routing, plus the
    /// pool accounting and stats counters.
    #[test]
    fn dsv4_gpu_device_moe_block_matches_host_and_oracle() {
        use crate::dsv4_cpu::{DsV4CpuLinear, DsV4MoeShared, Tid2Eid};

        let path = tempfile_path("mxfp4-moe-block");
        write_mxfp4_moe_gguf(&path);
        let (gguf, linear) = pool_test_linear(&path);
        linear.init_expert_pool_with_budget(1 << 20).unwrap(); // all 12 slices fit

        let router = RawMatrix {
            name: "blk.0.ffn_gate_inp.weight".to_string(),
            rows: 4,
            cols: 32,
        };
        let sh_gate = RawMatrix {
            name: "blk.0.ffn_gate_shexp.weight".to_string(),
            rows: 64,
            cols: 32,
        };
        let sh_up = RawMatrix {
            name: "blk.0.ffn_up_shexp.weight".to_string(),
            rows: 64,
            cols: 32,
        };
        let sh_down = RawMatrix {
            name: "blk.0.ffn_down_shexp.weight".to_string(),
            rows: 32,
            cols: 64,
        };
        linear
            .upload_resident(&[
                (router.clone(), None),
                (sh_gate.clone(), None),
                (sh_up.clone(), None),
                (sh_down.clone(), None),
            ])
            .unwrap();

        let gate = raw_experts("blk.0.ffn_gate_exps.weight", 32, 64);
        let up = raw_experts("blk.0.ffn_up_exps.weight", 32, 64);
        let down = raw_experts("blk.0.ffn_down_exps.weight", 64, 32);
        let shared = DsV4MoeShared {
            gate: &sh_gate,
            up: &sh_up,
            down: &sh_down,
        };
        let bias: Vec<f32> = test_activation(4, 505);
        let learned_ctx = DsV4MoeBlockCtx {
            router: &router,
            probs_bias: Some(&bias),
            tid2eid: None,
            gate: &gate,
            up: &up,
            down: &down,
            shared: Some(shared),
            swiglu_clamp: 10.0,
            experts: 4,
            top_k: 2,
            weights_norm: true,
            weights_scale: 1.5,
            embed: 32,
        };
        // Duplicate expert 3 in one row exercises the host's compute-once /
        // device's compute-per-rank equivalence.
        let table = Tid2Eid {
            name: "test.tid2eid".to_string(),
            stride: 2,
            tokens: 3,
            values: vec![0, 1, 1, 2, 3, 3],
        };
        let hash_ctx = DsV4MoeBlockCtx {
            probs_bias: None,
            tid2eid: Some(&table),
            ..learned_ctx
        };

        // 5 tokens: batch > 1 exercises the flat-buffer offsets; token id 4
        // exceeds the 3-row hash table and must clamp.
        let xs: Vec<Vec<f32>> = (0..5).map(|row| test_activation(32, 300 + row)).collect();
        let tokens: Vec<u32> = (0..5).collect();
        let cpu = DsV4CpuLinear::new_for_tests(gguf.clone());

        for (label, ctx) in [("learned", &learned_ctx), ("hash", &hash_ctx)] {
            let before = linear.moe_block_stats();
            let device = linear.moe_block(ctx, &xs, &tokens).unwrap();
            let after = linear.moe_block_stats();
            assert_eq!(
                after.device_blocks,
                before.device_blocks + 1,
                "{label}: expected the device path"
            );
            assert!(after.syncs - before.syncs >= 2, "{label}: >= 2 syncs");

            // (a) Host path on the same provider (the kill-switch A/B).
            linear.set_device_moe_enabled(false);
            let host = linear.moe_block(ctx, &xs, &tokens).unwrap();
            linear.set_device_moe_enabled(true);
            assert_eq!(
                linear.moe_block_stats().host_blocks,
                after.host_blocks + 1,
                "{label}: expected the host fallback"
            );
            let mut max_diff = 0.0f32;
            for (row, (dev, hst)) in device.iter().zip(&host).enumerate() {
                assert_eq!(dev.len(), 32);
                for (idx, (d, h)) in dev.iter().zip(hst).enumerate() {
                    let diff = (d - h).abs();
                    max_diff = max_diff.max(diff);
                    assert!(
                        diff <= 1.0e-4,
                        "{label} row {row} [{idx}]: device {d} vs host {h}"
                    );
                }
            }
            eprintln!("dsv4 moe_block {label}: device-vs-host max |diff| = {max_diff:e}");

            // (b) Pure-CPU oracle over the same GGUF (mmap dequant + f32 dot).
            let oracle = crate::dsv4_cpu::host_moe_block(&cpu, ctx, &xs, &tokens).unwrap();
            for (row, (dev, want)) in device.iter().zip(&oracle).enumerate() {
                assert_close_relative(dev, want, &format!("{label} oracle row {row}"));
            }
        }

        // Warm reruns hit the pool (no new uploads) and stay identical.
        let stats = linear.pool_stats().unwrap();
        assert_eq!(stats.misses, 12, "cold pass misses each slice once");
        let again = linear.moe_block(&learned_ctx, &xs, &tokens).unwrap();
        let host_again = {
            linear.set_device_moe_enabled(false);
            let out = linear.moe_block(&learned_ctx, &xs, &tokens).unwrap();
            linear.set_device_moe_enabled(true);
            out
        };
        assert_eq!(
            linear.pool_stats().unwrap().misses,
            12,
            "warm pass all hits"
        );
        for (dev, hst) in again.iter().zip(&host_again) {
            for (d, h) in dev.iter().zip(hst) {
                assert!((d - h).abs() <= 1.0e-4);
            }
        }
    }

    /// A pool too small to hold one batch's slice set simultaneously: the
    /// device path detects it after routing and falls back to the
    /// (eviction-safe) host path, still producing correct outputs.
    #[test]
    fn dsv4_gpu_device_moe_block_tiny_pool_falls_back() {
        use crate::dsv4_cpu::DsV4MoeShared;

        let path = tempfile_path("mxfp4-moe-tiny");
        write_mxfp4_moe_gguf(&path);
        let (_gguf, linear) = pool_test_linear(&path);
        // Two slots cannot hold the >= 6 slices a top-2 batch needs.
        let slot_stride = (TEST_EXPERT_SLICE_BYTES as usize).div_ceil(256) * 256;
        linear
            .init_expert_pool_with_budget(2 * slot_stride)
            .unwrap();

        let router = RawMatrix {
            name: "blk.0.ffn_gate_inp.weight".to_string(),
            rows: 4,
            cols: 32,
        };
        let sh_gate = RawMatrix {
            name: "blk.0.ffn_gate_shexp.weight".to_string(),
            rows: 64,
            cols: 32,
        };
        let sh_up = RawMatrix {
            name: "blk.0.ffn_up_shexp.weight".to_string(),
            rows: 64,
            cols: 32,
        };
        let sh_down = RawMatrix {
            name: "blk.0.ffn_down_shexp.weight".to_string(),
            rows: 32,
            cols: 64,
        };
        linear
            .upload_resident(&[
                (router.clone(), None),
                (sh_gate.clone(), None),
                (sh_up.clone(), None),
                (sh_down.clone(), None),
            ])
            .unwrap();
        let gate = raw_experts("blk.0.ffn_gate_exps.weight", 32, 64);
        let up = raw_experts("blk.0.ffn_up_exps.weight", 32, 64);
        let down = raw_experts("blk.0.ffn_down_exps.weight", 64, 32);
        let ctx = DsV4MoeBlockCtx {
            router: &router,
            probs_bias: None,
            tid2eid: None,
            gate: &gate,
            up: &up,
            down: &down,
            shared: Some(DsV4MoeShared {
                gate: &sh_gate,
                up: &sh_up,
                down: &sh_down,
            }),
            swiglu_clamp: 10.0,
            experts: 4,
            top_k: 2,
            weights_norm: true,
            weights_scale: 1.5,
            embed: 32,
        };
        let xs: Vec<Vec<f32>> = (0..3).map(|row| test_activation(32, 700 + row)).collect();
        let tokens: Vec<u32> = (0..3).collect();

        let device_attempt = linear.moe_block(&ctx, &xs, &tokens).unwrap();
        let stats = linear.moe_block_stats();
        assert_eq!(
            (stats.device_blocks, stats.host_blocks),
            (0, 1),
            "a 2-slot pool must fall back to the host path"
        );
        linear.set_device_moe_enabled(false);
        let host = linear.moe_block(&ctx, &xs, &tokens).unwrap();
        linear.set_device_moe_enabled(true);
        for (dev, hst) in device_attempt.iter().zip(&host) {
            for (d, h) in dev.iter().zip(hst) {
                assert!((d - h).abs() <= 1.0e-4, "fallback output mismatch");
            }
        }
    }

    // ---- Wave-2 Stage 2b: device decode step --------------------------------

    /// THE Stage-2b fixture parity gate: sequential decode with the device
    /// step vs the host step (`HI_DSV4_HOST_STEP=1` semantics via the test
    /// toggle) over a 14-token sequence exercising ring eviction (window 4),
    /// BOTH compressor variants (split ratio-4 and shared-K=V ratio-2), and
    /// the indexer's top-k narrowing. Gates: logits within 1e-4 at EVERY
    /// step, identical greedy continuations, and an equal host state mirror
    /// (ring + compressed blocks + pending) — the snapshot-correctness
    /// invariant.
    #[test]
    fn dsv4_gpu_device_step_matches_host_step() {
        let path = tempfile_path("device-step");
        write_deepseek4_gguf(&path);
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let lin = engine.linear();
        let tokens: Vec<u32> = (0..14).map(|idx| idx % 3).collect();

        // Host-step reference.
        lin.set_device_step_enabled(false);
        let mut host_state = engine.new_state();
        let mut host_logits = Vec::new();
        for &token in &tokens {
            host_logits.push(engine.step(&mut host_state, token).unwrap());
        }

        // Device steps over the same sequence.
        lin.set_device_step_enabled(true);
        let mut dev_state = engine.new_state();
        let mut dev_logits = Vec::new();
        for &token in &tokens {
            dev_logits.push(engine.step(&mut dev_state, token).unwrap());
        }
        let stats = gpu.step_stats();
        assert!(
            stats.device_steps >= tokens.len() as u64,
            "device path must serve every step: {stats:?}"
        );
        assert_eq!(stats.restores, 1, "one restore for the fresh state");

        let mut max_diff = 0.0f32;
        for (step, (host, dev)) in host_logits.iter().zip(&dev_logits).enumerate() {
            for (idx, (h, d)) in host.iter().zip(dev).enumerate() {
                let diff = (h - d).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff <= 1.0e-4,
                    "step {step} logit[{idx}]: host {h} vs device {d}"
                );
            }
        }
        eprintln!("dsv4 device-step fixture: max |logit diff| over 14 steps = {max_diff:e}");

        // The host mirror advanced by device steps must EQUAL the host-stepped
        // state (bit-level: both sides run identical GEMVs + exact kernels).
        assert_eq!(host_state.pos(), dev_state.pos());
        for (li, (h, d)) in host_state.layers.iter().zip(&dev_state.layers).enumerate() {
            assert_eq!(h.ring.len(), d.ring.len(), "layer {li} ring length");
            for (a, b) in h.ring.iter().zip(&d.ring) {
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() <= 1.0e-5, "layer {li} ring drift {x} vs {y}");
                }
            }
            for (hc, dc) in [
                (h.compressor.as_ref(), d.compressor.as_ref()),
                (h.indexer.as_ref(), d.indexer.as_ref()),
            ] {
                let (Some(hc), Some(dc)) = (hc, dc) else {
                    assert!(hc.is_none() && dc.is_none());
                    continue;
                };
                assert_eq!(hc.keys.len(), dc.keys.len(), "layer {li} block count");
                assert_eq!(hc.pending.len(), dc.pending.len(), "layer {li} pending");
                for (group_h, group_d) in [(&hc.keys, &dc.keys), (&hc.values, &dc.values)] {
                    for (a, b) in group_h.iter().zip(group_d) {
                        for (x, y) in a.iter().zip(b) {
                            assert!(
                                (x - y).abs() <= 1.0e-5,
                                "layer {li} compressed drift {x} vs {y}"
                            );
                        }
                    }
                }
            }
        }

        // Greedy continuation token-for-token.
        let mut host_l = host_logits.last().unwrap().clone();
        let mut dev_l = dev_logits.last().unwrap().clone();
        for step in 0..4 {
            let host_next = crate::qwen_cpu::argmax(&host_l).unwrap();
            let dev_next = crate::qwen_cpu::argmax(&dev_l).unwrap();
            assert_eq!(host_next, dev_next, "greedy step {step}");
            lin.set_device_step_enabled(false);
            host_l = engine.step(&mut host_state, host_next).unwrap();
            lin.set_device_step_enabled(true);
            dev_l = engine.step(&mut dev_state, dev_next).unwrap();
        }
    }

    /// Snapshot/restore + truncation on device-stepped states: a state cloned
    /// mid-decode (the prefix-cache snapshot pattern) must continue exactly
    /// like the original did after the original advanced further (forcing a
    /// host→device restore for the clone), and a slack-retaining state must
    /// still truncate and resume identically to a fresh host prefill.
    #[test]
    fn dsv4_gpu_device_step_snapshot_restore_and_truncate() {
        let path = tempfile_path("device-step-snap");
        write_deepseek4_gguf(&path);
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let lin = engine.linear();
        let tokens: Vec<u32> = (0..12).map(|idx| idx % 3).collect();

        // Chunked HOST prefill (the serving path), then device decode.
        let mut state = engine.new_state_with_ring_slack(8);
        engine.prefill_with_chunk(&mut state, &tokens, 4).unwrap();
        let l13 = engine.step(&mut state, 1).unwrap(); // pos 12 -> 13 (device)
        let snapshot = state.clone(); // prefix-cache snapshot mid-decode
        let l14_a = engine.step(&mut state, 2).unwrap(); // original advances
        let _l15_a = engine.step(&mut state, 0).unwrap();

        // The clone resumes at pos 13; the device state is ahead (pos 15), so
        // this forces a restore — and must reproduce the original's step.
        let restores_before = gpu.step_stats().restores;
        let mut resumed = snapshot.clone();
        let l14_b = engine.step(&mut resumed, 2).unwrap();
        assert!(
            gpu.step_stats().restores > restores_before,
            "resuming the snapshot must restore the device state"
        );
        assert_eq!(l14_a, l14_b, "snapshot resume diverged from the original");

        // Host-step cross-check of the resumed continuation.
        lin.set_device_step_enabled(false);
        let mut host_resumed = snapshot.clone();
        let l14_h = engine.step(&mut host_resumed, 2).unwrap();
        lin.set_device_step_enabled(true);
        for (idx, (d, h)) in l14_b.iter().zip(&l14_h).enumerate() {
            assert!(
                (d - h).abs() <= 1.0e-4,
                "resumed logit[{idx}]: device {d} vs host {h}"
            );
        }
        let _ = l13;

        // The serving pattern: per-boundary snapshot clones during decode must
        // be pure host-mirror clones — NO device round-trip. Switching back to
        // the original conversation costs exactly ONE restore (the resumed
        // clone owned the single device slot); after that, decode with a clone
        // after EVERY step (a stricter cadence than the backend's per-boundary
        // on_feed) must add zero further restores.
        engine.step(&mut state, 0).unwrap(); // absorb the conversation switch
        let stats_before = gpu.step_stats();
        let mut boundary_snapshots = Vec::new();
        for tok in [1u32, 0, 2, 1] {
            engine.step(&mut state, tok).unwrap();
            boundary_snapshots.push(state.clone());
        }
        let stats_after = gpu.step_stats();
        assert_eq!(
            stats_after.restores, stats_before.restores,
            "boundary clones must not force device restores"
        );
        assert_eq!(
            stats_after.device_steps - stats_before.device_steps,
            4,
            "every decode step stays device-side across boundary clones"
        );
        drop(boundary_snapshots);

        // Truncation of a device-stepped state: rewind to pos 12, then a
        // device step must match a fresh host prefill of the same prefix.
        let mut truncated = state.clone();
        assert_eq!(
            engine.truncate_state_to_at_most(&mut truncated, 13),
            Some(12)
        );
        let from_truncated = engine.step(&mut truncated, 2).unwrap();
        lin.set_device_step_enabled(false);
        let mut fresh = engine.new_state_with_ring_slack(8);
        engine.prefill_with_chunk(&mut fresh, &tokens, 1).unwrap();
        let from_fresh = engine.step(&mut fresh, 2).unwrap();
        lin.set_device_step_enabled(true);
        for (idx, (t, f)) in from_truncated.iter().zip(&from_fresh).enumerate() {
            assert!(
                (t - f).abs() <= 1.0e-4,
                "truncated logit[{idx}]: device {t} vs fresh host {f}"
            );
        }
    }

    /// Stage-A verify on the GPU provider: per-position logits are bit-exact
    /// with the sequential HOST step path (the stated contract — the chunk
    /// machinery issues the identical cuBLAS GEMVs), greedy-identical with
    /// the production device-step path at every position, and unmoved by
    /// `HI_DSV4_PREFILL_GEMM=1` (verify pins the exact batching mode, so the
    /// GEMM opt-in stays prefill-only).
    #[test]
    fn dsv4_gpu_verify_tokens_matches_sequential_and_pins_exact_batching() {
        let path = tempfile_path("gpu-verify");
        write_deepseek4_gguf(&path);
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let lin = engine.linear();

        let prompt: Vec<u32> = (0..6).map(|idx| idx % 3).collect();
        let continuation: Vec<u32> = (0..8).map(|idx| (idx * 2) % 3).collect();

        // Host-step sequential reference.
        lin.set_device_step_enabled(false);
        let mut host_state = engine.new_state();
        engine
            .prefill_with_chunk(&mut host_state, &prompt, 4)
            .unwrap();
        let verify_base = host_state.clone();
        let host_rows: Vec<Vec<f32>> = continuation
            .iter()
            .map(|&token| engine.step(&mut host_state, token).unwrap())
            .collect();
        lin.set_device_step_enabled(true);

        let mut verify_state = verify_base.clone();
        let verify_rows = engine
            .verify_tokens(&mut verify_state, &continuation)
            .unwrap();
        assert_eq!(
            verify_rows, host_rows,
            "verify must be bit-exact with sequential host steps"
        );

        // Production decode path (device-resident steps): identical greedy
        // choice at every position, logits within the device-step tolerance.
        let mut dev_state = verify_base.clone();
        for (idx, &token) in continuation.iter().enumerate() {
            let dev = engine.step(&mut dev_state, token).unwrap();
            assert_eq!(
                crate::qwen_cpu::argmax(&verify_rows[idx]).unwrap(),
                crate::qwen_cpu::argmax(&dev).unwrap(),
                "greedy choice at position {idx}"
            );
            for (v, d) in verify_rows[idx].iter().zip(&dev) {
                assert!((v - d).abs() <= 1.0e-4, "position {idx}: {v} vs {d}");
            }
        }

        // GEMM prefill mode must not leak into verify: with batching forced
        // on, verify logits stay bit-exact with the host-step reference.
        lin.set_gemm_batching(true);
        let mut gemm_state = verify_base.clone();
        let gemm_rows = engine
            .verify_tokens(&mut gemm_state, &continuation)
            .unwrap();
        lin.set_gemm_batching(false);
        assert_eq!(
            gemm_rows, host_rows,
            "verify under HI_DSV4_PREFILL_GEMM must pin the exact batching mode"
        );
    }

    /// Stage-A rollback on the GPU provider across verify boundaries, with
    /// the production device-step decode continuing from the rewound state:
    /// the continuation matches a state that only ever processed the accepted
    /// tokens, and the (tag, pos) mirror protocol survives the rewind (the
    /// next device step restores from the authoritative host mirror).
    #[test]
    fn dsv4_gpu_verify_rewind_round_trip_with_device_steps() {
        let path = tempfile_path("gpu-rewind");
        write_deepseek4_gguf(&path);
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();

        let prompt: Vec<u32> = (0..6).map(|idx| idx % 3).collect();
        let continuation: Vec<u32> = vec![1, 2, 0, 1, 2];
        let mut history = prompt.clone();
        history.extend(&continuation);
        let slack = 4 + continuation.len() + 1;

        for keep in [1usize, 3, 5] {
            let target = prompt.len() + keep;
            let mut state = engine.new_state_with_ring_slack(slack);
            engine.prefill_with_chunk(&mut state, &prompt, 4).unwrap();
            engine.verify_tokens(&mut state, &continuation).unwrap();
            engine
                .rewind_state_to(&mut state, &history[..target], target, None)
                .unwrap();
            assert_eq!(state.pos(), target);

            let mut reference = engine.new_state_with_ring_slack(slack);
            engine
                .prefill_with_chunk(&mut reference, &prompt, 4)
                .unwrap();
            for &token in &continuation[..keep] {
                engine.step(&mut reference, token).unwrap();
            }
            let restores_before = gpu.step_stats().restores;
            for &token in &[2u32, 0, 1] {
                // Device steps on both sides (the production decode path).
                let a = engine.step(&mut state, token).unwrap();
                let b = engine.step(&mut reference, token).unwrap();
                assert_eq!(
                    crate::qwen_cpu::argmax(&a).unwrap(),
                    crate::qwen_cpu::argmax(&b).unwrap(),
                    "keep {keep}: greedy diverged after rewind"
                );
                for (x, y) in a.iter().zip(&b) {
                    assert!((x - y).abs() <= 1.0e-4, "keep {keep}: {x} vs {y}");
                }
            }
            // The rewind is a host-side mutation, so it severed the device
            // link; the continuation above must have restored the mirror.
            assert!(
                gpu.step_stats().restores > restores_before,
                "keep {keep}: rewound state must force a device restore"
            );
        }
    }

    /// Taps on the GPU provider: verify-chunk capture equals a tapped
    /// (host-step) sequential run bit for bit, and capture never perturbs the
    /// verify logits.
    #[test]
    fn dsv4_gpu_taps_match_host_step_capture() {
        use crate::dsv4_cpu::DsV4TapConfig;

        let path = tempfile_path("gpu-taps");
        write_deepseek4_gguf(&path);
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();

        let prompt: Vec<u32> = (0..6).map(|idx| idx % 3).collect();
        let continuation: Vec<u32> = vec![2, 1, 0, 2];
        let mut base = engine.new_state();
        engine.prefill_with_chunk(&mut base, &prompt, 4).unwrap();
        let config = DsV4TapConfig {
            pre_hc_head: true,
            aux_layers: vec![1],
        };

        let mut tapped_state = base.clone();
        let mut taps = engine.new_taps(config.clone()).unwrap();
        let tapped_rows = engine
            .verify_tokens_with_taps(&mut tapped_state, &continuation, Some(&mut taps))
            .unwrap();
        let mut plain_state = base.clone();
        let plain_rows = engine
            .verify_tokens(&mut plain_state, &continuation)
            .unwrap();
        assert_eq!(tapped_rows, plain_rows, "capture must not perturb verify");

        // Sequential capture oracle (step_with_taps forces the host step).
        let mut seq_taps = engine.new_taps(config).unwrap();
        let mut seq_state = base.clone();
        for &token in &continuation {
            engine
                .step_with_taps(&mut seq_state, token, Some(&mut seq_taps))
                .unwrap();
        }
        assert_eq!(taps.positions(), continuation.len());
        for position in 0..continuation.len() {
            assert_eq!(
                taps.pre_hc_head(position).unwrap(),
                seq_taps.pre_hc_head(position).unwrap(),
                "pre-hc-head row {position}"
            );
            assert_eq!(
                taps.aux_flat(1, position).unwrap(),
                seq_taps.aux_flat(1, position).unwrap(),
                "aux row {position}"
            );
        }
    }

    // ---- Real-model gates (ignored: need the checkpoint + exclusive GPU) ---

    /// The local DeepSeek-V4-Flash checkpoint, when present.
    pub(crate) fn real_model_path() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let path = std::path::PathBuf::from(home)
            .join(".hi/models/deepseek-v4-flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf");
        path.exists().then_some(path)
    }

    /// A prompt that tokenizes to well over `min_tokens` tokens.
    pub(crate) fn long_prompt(min_tokens: usize) -> String {
        "The quick brown fox jumps over the lazy dog while the seasoned engineer \
         profiles a batched prefill path, measuring tokens per second across chunk \
         boundaries, compressor blocks, and indexer selections. "
            .repeat(min_tokens / 8)
    }

    /// Parity gate (b) on the real model: a 64-token prompt prefilled chunked
    /// (B=64) vs sequentially must produce the same greedy next token with
    /// max logit drift <= 0.05. Run explicitly:
    /// `CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release --features native-cuda \
    ///  dsv4_real_model_chunked_prefill_parity -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    fn dsv4_real_model_chunked_prefill_parity() {
        let Some(path) = real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let tokens = engine.tokenizer().encode(&long_prompt(96)).unwrap();
        assert!(tokens.len() >= 64, "prompt too short: {}", tokens.len());
        let tokens = &tokens[..64];

        let started = std::time::Instant::now();
        let mut sequential_state = engine.new_state();
        let sequential = engine
            .prefill_with_chunk(&mut sequential_state, tokens, 1)
            .unwrap();
        let sequential_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let mut chunked_state = engine.new_state();
        let chunked = engine
            .prefill_with_chunk(&mut chunked_state, tokens, 64)
            .unwrap();
        let chunked_elapsed = started.elapsed();

        let sequential_next = crate::qwen_cpu::argmax(&sequential).unwrap();
        let chunked_next = crate::qwen_cpu::argmax(&chunked).unwrap();
        let mut max_drift = 0.0f32;
        for (seq, chk) in sequential.iter().zip(&chunked) {
            max_drift = max_drift.max((seq - chk).abs());
        }
        eprintln!(
            "real-model 64-token prefill: sequential {:.1}s ({:.2} tok/s), chunked {:.1}s ({:.2} tok/s), next token {sequential_next} vs {chunked_next}, max logit drift {max_drift:.4}",
            sequential_elapsed.as_secs_f64(),
            64.0 / sequential_elapsed.as_secs_f64(),
            chunked_elapsed.as_secs_f64(),
            64.0 / chunked_elapsed.as_secs_f64(),
        );
        assert_eq!(sequential_next, chunked_next, "greedy next token diverged");
        assert!(max_drift <= 0.05, "logit drift {max_drift} exceeds 0.05");
    }

    /// Stage-A verify gate on the real checkpoint: verify_tokens over the
    /// model's own greedy continuation must pick the SAME greedy token at
    /// every position as sequential decode steps (the speculative
    /// losslessness requirement) with near-zero logit drift, and the rewind
    /// round-trip must resume decode on the sequential trajectory. Run
    /// explicitly (a small expert pool keeps the load polite on a shared GPU):
    /// `HI_DSV4_EXPERT_POOL_GB=24 CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release \
    ///  --features native-cuda dsv4_real_model_verify_tokens_parity -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    fn dsv4_real_model_verify_tokens_parity() {
        let Some(path) = real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let tokens = engine.tokenizer().encode(&long_prompt(300)).unwrap();
        assert!(tokens.len() >= 256, "prompt too short: {}", tokens.len());
        let prompt = &tokens[..256];
        const VERIFY: usize = 6;

        // Prefill once with speculative ring slack, then split into the
        // sequential (device-step) and verify trajectories.
        let slack = 128 + VERIFY + 1;
        let started = std::time::Instant::now();
        let mut sequential_state = engine.new_state_with_ring_slack(slack);
        let mut logits = engine.prefill(&mut sequential_state, prompt).unwrap();
        eprintln!(
            "real-model verify: 256-token prefill in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        let mut verify_state = sequential_state.clone();

        // The model's own greedy continuation via the production decode path.
        let mut continuation = Vec::new();
        let mut sequential_rows = Vec::new();
        for _ in 0..VERIFY {
            let next = crate::qwen_cpu::argmax(&logits).unwrap();
            continuation.push(next);
            logits = engine.step(&mut sequential_state, next).unwrap();
            sequential_rows.push(logits.clone());
        }

        // Verify those tokens in one chunked forward; per-position greedy
        // choices must match the sequential rows exactly.
        let started = std::time::Instant::now();
        let verify_rows = engine
            .verify_tokens(&mut verify_state, &continuation)
            .unwrap();
        eprintln!(
            "real-model verify: {VERIFY}-token verify chunk in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        let mut max_drift = 0.0f32;
        for (position, (verify, sequential)) in verify_rows.iter().zip(&sequential_rows).enumerate()
        {
            assert_eq!(
                crate::qwen_cpu::argmax(verify).unwrap(),
                crate::qwen_cpu::argmax(sequential).unwrap(),
                "greedy choice diverged at verify position {position}"
            );
            for (v, s) in verify.iter().zip(sequential) {
                max_drift = max_drift.max((v - s).abs());
            }
        }
        eprintln!("real-model verify: max |logit drift| over {VERIFY} positions = {max_drift:e}");
        assert!(max_drift <= 1.0e-3, "verify drift {max_drift} exceeds 1e-3");

        // Rewind round-trip: keep 3 of the 6 verified tokens, then decode must
        // continue on the sequential trajectory.
        let keep = 3;
        let mut history = prompt.to_vec();
        history.extend(&continuation);
        let target = prompt.len() + keep;
        engine
            .rewind_state_to(&mut verify_state, &history[..target], target, None)
            .unwrap();
        let resumed = engine.step(&mut verify_state, continuation[keep]).unwrap();
        assert_eq!(
            crate::qwen_cpu::argmax(&resumed).unwrap(),
            crate::qwen_cpu::argmax(&sequential_rows[keep]).unwrap(),
            "greedy choice diverged after rewind"
        );
        eprintln!("real-model verify: rewind to {target} resumed on the sequential trajectory");
    }

    /// Before/after timing on a ~1.5k-token prompt (the validation run the
    /// bring-up doc asks for). Run explicitly:
    /// `CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release --features native-cuda \
    ///  dsv4_real_model_prefill_benchmark -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint; the sequential baseline takes minutes"]
    fn dsv4_real_model_prefill_benchmark() {
        let Some(path) = real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let tokens = engine.tokenizer().encode(&long_prompt(1600)).unwrap();
        assert!(tokens.len() >= 1500, "prompt too short: {}", tokens.len());
        let tokens = &tokens[..1500];

        let started = std::time::Instant::now();
        let mut chunked_state = engine.new_state();
        let chunked = engine
            .prefill_with_chunk(&mut chunked_state, tokens, 64)
            .unwrap();
        let chunked_elapsed = started.elapsed();
        eprintln!(
            "real-model 1500-token prefill (chunk 64): {:.1}s = {:.2} tok/s",
            chunked_elapsed.as_secs_f64(),
            1500.0 / chunked_elapsed.as_secs_f64(),
        );

        let started = std::time::Instant::now();
        let mut sequential_state = engine.new_state();
        let sequential = engine
            .prefill_with_chunk(&mut sequential_state, tokens, 1)
            .unwrap();
        let sequential_elapsed = started.elapsed();
        eprintln!(
            "real-model 1500-token prefill (sequential): {:.1}s = {:.2} tok/s",
            sequential_elapsed.as_secs_f64(),
            1500.0 / sequential_elapsed.as_secs_f64(),
        );

        let sequential_next = crate::qwen_cpu::argmax(&sequential).unwrap();
        let chunked_next = crate::qwen_cpu::argmax(&chunked).unwrap();
        assert_eq!(sequential_next, chunked_next, "greedy next token diverged");
    }

    /// Copy-stream prefetch validation on the real model (roadmap item 5): a
    /// 256-token prompt + 64 greedy tokens, comparing the synchronous demand
    /// path against copy-stream prefetch (and speculative prefetch) on one
    /// model load. Asserts identical greedy tokens across all three paths and
    /// reports prefill/decode tok/s before/after plus the speculative hit rate.
    /// Run explicitly on the free GPU (copy stream is opt-in, so enable it):
    /// `HI_DSV4_COPY_STREAM=1 CUDA_VISIBLE_DEVICES=1 cargo test -p hi-cuda --release \
    ///  --features native-cuda dsv4_real_model_copy_stream_prefetch -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    fn dsv4_real_model_copy_stream_prefetch() {
        use std::time::Instant;
        let Some(path) = real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let lin = engine.linear();
        if !lin.copy_stream_present() {
            eprintln!(
                "skipping A/B: copy stream not built — re-run with HI_DSV4_COPY_STREAM=1 to compare paths"
            );
            return;
        }
        let tokens = engine.tokenizer().encode(&long_prompt(600)).unwrap();
        assert!(tokens.len() >= 256, "prompt too short: {}", tokens.len());
        let prompt = &tokens[..256];
        let chunk = engine.prefill_chunk_size().max(64);
        const GEN: usize = 64;

        // One 256-prefill + 64-greedy run; returns the generated tokens and the
        // prefill/decode tok/s, plus the pool-stats delta over the run.
        let run = |label: &str| -> (Vec<u32>, f64, f64, DsV4ExpertPoolStats) {
            let before = gpu.pool_stats().unwrap_or_default();
            let mut state = engine.new_state();
            let t0 = Instant::now();
            let mut logits = engine
                .prefill_with_chunk(&mut state, prompt, chunk)
                .unwrap();
            let prefill_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut generated = Vec::with_capacity(GEN);
            for _ in 0..GEN {
                let next = crate::qwen_cpu::argmax(&logits).unwrap();
                generated.push(next);
                logits = engine.step(&mut state, next).unwrap();
            }
            let decode_s = t1.elapsed().as_secs_f64();
            let after = gpu.pool_stats().unwrap_or_default();
            let delta = DsV4ExpertPoolStats {
                hits: after.hits - before.hits,
                misses: after.misses - before.misses,
                evictions: after.evictions - before.evictions,
                prefilled: 0,
                bytes_uploaded: after.bytes_uploaded - before.bytes_uploaded,
                prefetch_uploads: after.prefetch_uploads - before.prefetch_uploads,
                spec_uploads: after.spec_uploads - before.spec_uploads,
                spec_hits: after.spec_hits - before.spec_hits,
            };
            let pre_tps = prompt.len() as f64 / prefill_s;
            let dec_tps = GEN as f64 / decode_s;
            eprintln!(
                "[{label}] prefill {:.2}s ({pre_tps:.2} tok/s), decode {:.2}s ({dec_tps:.2} tok/s); pool {}",
                prefill_s,
                decode_s,
                format_pool_stats(&delta),
            );
            (generated, pre_tps, dec_tps, delta)
        };

        // Warm the pool to steady state so the A/B isolates the prefetch effect
        // rather than cold-start differences (discarded).
        lin.set_prefetch_enabled(true);
        lin.set_spec_prefetch(false);
        let _ = run("warmup");

        // Synchronous demand path (output-equivalent to HI_DSV4_NO_COPY_STREAM=1).
        lin.set_prefetch_enabled(false);
        let (toks_off, _pre_off, dec_off, _) = run("sync   ");

        // Copy-stream within-layer prefetch.
        lin.set_prefetch_enabled(true);
        lin.set_spec_prefetch(false);
        let (toks_on, _pre_on, dec_on, _) = run("prefetch");

        // Copy-stream prefetch + speculative next-layer warm.
        lin.set_spec_prefetch(true);
        let (toks_spec, _pre_spec, dec_spec, spec_delta) = run("spec    ");

        assert_eq!(
            toks_off, toks_on,
            "copy-stream prefetch changed the greedy continuation"
        );
        assert_eq!(
            toks_off, toks_spec,
            "speculative prefetch changed the greedy continuation"
        );

        let spec_rate = if spec_delta.spec_uploads > 0 {
            100.0 * spec_delta.spec_hits as f64 / spec_delta.spec_uploads as f64
        } else {
            0.0
        };
        eprintln!(
            "SUMMARY decode tok/s: sync {dec_off:.2} -> prefetch {dec_on:.2} ({:+.1}%), spec {dec_spec:.2}; speculative hit rate {spec_rate:.1}% ({}/{})",
            100.0 * (dec_on / dec_off - 1.0),
            spec_delta.spec_hits,
            spec_delta.spec_uploads,
        );
    }

    /// Wave-2 Stage 2b acceptance gate on the real model: a 256-token prompt
    /// + 64 greedy tokens must produce the IDENTICAL token sequence with the
    /// device decode step and with the host step (`HI_DSV4_HOST_STEP=1`
    /// semantics via the test toggle), on one model load. Reports decode and
    /// prefill tok/s for both paths, per-step launch/sync counts, restore
    /// counts, and the expert-pool upload volume per decoded token (the
    /// bottleneck decomposition). Run explicitly on the free GPU:
    /// `HI_DSV4_EXPERT_POOL_GB=40 CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release \
    ///  --features native-cuda dsv4_real_model_device_step_parity_and_speed -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    fn dsv4_real_model_device_step_parity_and_speed() {
        use std::time::Instant;
        let Some(path) = real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let lin = engine.linear();
        let tokens = engine.tokenizer().encode(&long_prompt(600)).unwrap();
        assert!(tokens.len() >= 256, "prompt too short: {}", tokens.len());
        let prompt = &tokens[..256];
        let chunk = engine.prefill_chunk_size();
        const GEN: usize = 64;

        // One 256-prefill + 64-greedy probe. Prefill stays on the chunked
        // host path (unchanged by Stage 2b); decode runs through engine.step,
        // which the toggle routes device- or host-side.
        let run = |label: &str| -> (Vec<u32>, f64, f64, DsV4StepStats, DsV4ExpertPoolStats) {
            let steps_before = gpu.step_stats();
            let pool_before = gpu.pool_stats().unwrap_or_default();
            let mut state = engine.new_state();
            let t0 = Instant::now();
            let mut logits = engine
                .prefill_with_chunk(&mut state, prompt, chunk)
                .unwrap();
            let prefill_s = t0.elapsed().as_secs_f64();
            let pool_mid = gpu.pool_stats().unwrap_or_default();
            let t1 = Instant::now();
            let mut generated = Vec::with_capacity(GEN);
            for _ in 0..GEN {
                let next = crate::qwen_cpu::argmax(&logits).unwrap();
                generated.push(next);
                logits = engine.step(&mut state, next).unwrap();
            }
            let decode_s = t1.elapsed().as_secs_f64();
            let steps_after = gpu.step_stats();
            let pool_after = gpu.pool_stats().unwrap_or_default();
            let delta = DsV4StepStats {
                device_steps: steps_after.device_steps - steps_before.device_steps,
                host_steps: steps_after.host_steps - steps_before.host_steps,
                restores: steps_after.restores - steps_before.restores,
                launches: steps_after.launches - steps_before.launches,
                syncs: steps_after.syncs - steps_before.syncs,
                restore_syncs: steps_after.restore_syncs - steps_before.restore_syncs,
            };
            let decode_upload =
                (pool_after.bytes_uploaded - pool_mid.bytes_uploaded) as f64 / GEN as f64;
            let decode_pool = DsV4ExpertPoolStats {
                hits: pool_after.hits - pool_mid.hits,
                misses: pool_after.misses - pool_mid.misses,
                ..Default::default()
            };
            let pre_tps = prompt.len() as f64 / prefill_s;
            let dec_tps = GEN as f64 / decode_s;
            let per_step = |value: u64| {
                if delta.device_steps > 0 {
                    value as f64 / delta.device_steps as f64
                } else {
                    0.0
                }
            };
            let served = decode_pool.hits + decode_pool.misses;
            let hit_rate = if served > 0 {
                100.0 * decode_pool.hits as f64 / served as f64
            } else {
                0.0
            };
            eprintln!(
                "[{label}] prefill {prefill_s:.2}s ({pre_tps:.2} tok/s), decode {decode_s:.2}s \
                 ({dec_tps:.2} tok/s); steps device {} host {}, restores {}, launches/step {:.0}, \
                 syncs/step {:.2} (+{:.1} restore copies total); decode pool {hit_rate:.1}% hit, \
                 {:.1} MiB/tok uploaded",
                delta.device_steps,
                delta.host_steps,
                delta.restores,
                per_step(delta.launches),
                per_step(delta.syncs),
                delta.restore_syncs as f64,
                decode_upload / (1u64 << 20) as f64,
            );
            let _ = pool_before;
            (generated, pre_tps, dec_tps, delta, pool_after)
        };

        // Warm the expert pool + scratch so the A/B measures steady state.
        let _ = run("warmup ");
        let (toks_dev, pre_dev, dec_dev, delta_dev, _) = run("device ");
        assert!(
            delta_dev.device_steps as usize >= GEN && delta_dev.host_steps == 0,
            "device probe must serve every decode step device-side: {delta_dev:?}"
        );
        let (toks_dev2, _pre_dev2, dec_dev2, _, _) = run("device2");
        assert_eq!(toks_dev, toks_dev2, "device decode must be deterministic");

        lin.set_device_step_enabled(false);
        let (toks_host, pre_host, dec_host, delta_host, _) = run("host   ");
        assert_eq!(
            delta_host.device_steps, 0,
            "host probe must not touch the device path"
        );
        lin.set_device_step_enabled(true);

        // THE Stage-2b parity gate: identical greedy continuations.
        assert_eq!(
            toks_dev, toks_host,
            "device decode step changed the greedy continuation"
        );
        eprintln!(
            "SUMMARY 256+64 probe: decode {dec_host:.2} -> {:.2} tok/s ({:+.1}%), \
             prefill {pre_host:.2} -> {pre_dev:.2} tok/s ({:+.1}%); 64 tokens identical",
            dec_dev.max(dec_dev2),
            100.0 * (dec_dev.max(dec_dev2) / dec_host - 1.0),
            100.0 * (pre_dev / pre_host - 1.0),
        );
    }

    /// Wave-2 Stage 2a acceptance gate on the real model: a 256-token prompt
    /// + 64 greedy tokens must produce the IDENTICAL token sequence with the
    /// device MoE block and with the host path (`HI_DSV4_NO_DEVICE_MOE=1`
    /// semantics via the test toggle), on one model load. Reports prefill and
    /// decode tok/s for both paths, the per-layer launch/sync counts of the
    /// device path, and a 1500-token prefill A/B. Run explicitly on the free
    /// GPU:
    /// `CUDA_VISIBLE_DEVICES=1 cargo test -p hi-cuda --release --features native-cuda \
    ///  dsv4_real_model_device_moe_parity_and_speed -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    fn dsv4_real_model_device_moe_parity_and_speed() {
        use std::time::Instant;
        let Some(path) = real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let gpu = DeepSeekV4GpuEngine::load(&path).unwrap();
        let engine = gpu.engine();
        let lin = engine.linear();
        let tokens = engine.tokenizer().encode(&long_prompt(600)).unwrap();
        assert!(tokens.len() >= 256, "prompt too short: {}", tokens.len());
        let prompt = &tokens[..256];
        let chunk = engine.prefill_chunk_size();
        const GEN: usize = 64;

        // One 256-prefill + 64-greedy probe; returns the generated tokens,
        // prefill/decode tok/s, and the moe_block stats delta. Pool-stats
        // deltas are reported per phase so the residual decode bottleneck
        // (expert-slice miss uploads on a pool that holds ~53% of the model)
        // is quantified, not guessed.
        let run = |label: &str| -> (Vec<u32>, f64, f64, DsV4MoeBlockStats) {
            let before = gpu.moe_block_stats();
            let pool0 = gpu.pool_stats().unwrap_or_default();
            let mut state = engine.new_state();
            let t0 = Instant::now();
            let mut logits = engine
                .prefill_with_chunk(&mut state, prompt, chunk)
                .unwrap();
            let prefill_s = t0.elapsed().as_secs_f64();
            let pool1 = gpu.pool_stats().unwrap_or_default();
            let t1 = Instant::now();
            let mut generated = Vec::with_capacity(GEN);
            for _ in 0..GEN {
                let next = crate::qwen_cpu::argmax(&logits).unwrap();
                generated.push(next);
                logits = engine.step(&mut state, next).unwrap();
            }
            let decode_s = t1.elapsed().as_secs_f64();
            let pool2 = gpu.pool_stats().unwrap_or_default();
            let after = gpu.moe_block_stats();
            let delta = DsV4MoeBlockStats {
                device_blocks: after.device_blocks - before.device_blocks,
                host_blocks: after.host_blocks - before.host_blocks,
                launches: after.launches - before.launches,
                syncs: after.syncs - before.syncs,
            };
            let pre_tps = prompt.len() as f64 / prefill_s;
            let dec_tps = GEN as f64 / decode_s;
            let per_block = |value: u64| {
                if delta.device_blocks > 0 {
                    value as f64 / delta.device_blocks as f64
                } else {
                    0.0
                }
            };
            let phase_miss = |a: &DsV4ExpertPoolStats, b: &DsV4ExpertPoolStats, tokens: f64| {
                let (hits, misses) = (b.hits - a.hits, b.misses - a.misses);
                let rate = if hits + misses > 0 {
                    100.0 * hits as f64 / (hits + misses) as f64
                } else {
                    0.0
                };
                format!(
                    "{rate:.1}% hit, {:.0} MiB/tok uploaded",
                    (b.bytes_uploaded - a.bytes_uploaded) as f64 / tokens / (1u64 << 20) as f64,
                )
            };
            eprintln!(
                "[{label}] prefill {prefill_s:.2}s ({pre_tps:.2} tok/s; pool {}), \
                 decode {decode_s:.2}s ({dec_tps:.2} tok/s; pool {}); \
                 moe blocks: device {} host {}, launches/block {:.1}, syncs/block {:.2}",
                phase_miss(&pool0, &pool1, prompt.len() as f64),
                phase_miss(&pool1, &pool2, GEN as f64),
                delta.device_blocks,
                delta.host_blocks,
                per_block(delta.launches),
                per_block(delta.syncs),
            );
            (generated, pre_tps, dec_tps, delta)
        };

        // Warm the expert pool + scratch so the A/B measures steady state.
        let _ = run("warmup ");
        let (toks_dev, pre_dev, dec_dev, delta_dev) = run("device ");
        assert!(
            delta_dev.device_blocks > 0 && delta_dev.host_blocks == 0,
            "device probe must serve every MoE block device-side: {delta_dev:?}"
        );
        // With the copy engine built (HI_DSV4_COPY_STREAM=1), A/B the miss
        // path: prefetched (copy-stream, pinned staging) vs synchronous
        // demand uploads. Miss uploads are the dominant residual decode cost
        // on a pool that cannot hold the whole model, so this now matters.
        if lin.copy_stream_present() {
            lin.set_prefetch_enabled(false);
            let (toks_sync, _pre_sync, dec_sync, _) = run("dev-syn");
            lin.set_prefetch_enabled(true);
            assert_eq!(
                toks_dev, toks_sync,
                "copy-stream prefetch changed the greedy continuation"
            );
            eprintln!(
                "copy-stream A/B: decode {dec_sync:.2} tok/s (sync misses) vs {dec_dev:.2} tok/s (prefetched)"
            );
        }
        lin.set_device_moe_enabled(false);
        let (toks_host, pre_host, dec_host, delta_host) = run("host   ");
        assert_eq!(
            delta_host.device_blocks, 0,
            "host probe must not touch the device path"
        );
        lin.set_device_moe_enabled(true);

        // THE Stage-2a parity gate: identical greedy continuations.
        assert_eq!(
            toks_dev, toks_host,
            "device MoE changed the greedy continuation"
        );
        eprintln!(
            "SUMMARY 256+64 probe: decode {dec_host:.2} -> {dec_dev:.2} tok/s ({:+.1}%), \
             prefill {pre_host:.2} -> {pre_dev:.2} tok/s ({:+.1}%); tokens identical",
            100.0 * (dec_dev / dec_host - 1.0),
            100.0 * (pre_dev / pre_host - 1.0),
        );

        // 1500-token prefill A/B (same load); greedy next token must agree.
        let long = engine.tokenizer().encode(&long_prompt(1600)).unwrap();
        assert!(long.len() >= 1500, "prompt too short: {}", long.len());
        let long = &long[..1500];
        let prefill_ab = |label: &str| -> (u32, f64) {
            let mut state = engine.new_state();
            let t0 = Instant::now();
            let logits = engine.prefill_with_chunk(&mut state, long, chunk).unwrap();
            let seconds = t0.elapsed().as_secs_f64();
            let tps = long.len() as f64 / seconds;
            eprintln!("[{label}] 1500-token prefill {seconds:.1}s = {tps:.2} tok/s");
            (crate::qwen_cpu::argmax(&logits).unwrap(), tps)
        };
        let (next_dev, long_dev_tps) = prefill_ab("device ");
        lin.set_device_moe_enabled(false);
        let (next_host, long_host_tps) = prefill_ab("host   ");
        lin.set_device_moe_enabled(true);
        assert_eq!(
            next_dev, next_host,
            "1500-token prefill next-token diverged"
        );
        eprintln!(
            "SUMMARY 1500-token prefill: {long_host_tps:.2} -> {long_dev_tps:.2} tok/s ({:+.1}%)",
            100.0 * (long_dev_tps / long_host_tps - 1.0),
        );
    }
}
