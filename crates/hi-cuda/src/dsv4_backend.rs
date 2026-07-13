//! Single-request serving backend for DeepSeek-V4-Flash (`deepseek4`), Stage 2
//! of `docs/deepseek-v4-flash-gpu-bringup.md`.
//!
//! Wraps [`DeepSeekV4GpuEngine`] (the host-orchestrated Stage-1 CUDA engine) in
//! an [`InferenceBackend`] so `hi-local serve --backend cuda` can serve V4
//! GGUFs. Deliberately NOT wired into the continuous scheduler / paged-KV
//! machinery: V4's ring+compressed KV cache is engine-internal and the engine
//! decodes one token at a time, so the backend is a plain FIFO over a single
//! generation slot (max batch size is effectively 1).
//!
//! Prompts prefill through the engine's chunked path (`HI_DSV4_PREFILL_CHUNK`,
//! default 64 tokens per batched pass; `HI_DSV4_PREFILL_GEMM=1` opts into the
//! faster non-bit-exact GEMM batching — see `dsv4_gpu`), and the worker keeps a
//! block-hash prefix cache ([`DsV4PrefixCache`]) of engine-state snapshots
//! shared across every conversation it has served:
//!
//! - Snapshots are taken at fixed block boundaries (`HI_DSV4_PREFIX_BLOCK_SIZE`
//!   tokens, default 256) and keyed by a vLLM-style rolling hash chain
//!   `block_hash_N = hash(block_hash_{N-1}, tokens_in_block_N)`, so any prefix
//!   ending on a boundary has one stable key regardless of which conversation
//!   produced it. Because the engine's per-position state is a full host-side
//!   snapshot (raw-KV ring + compressor/indexer state), a boundary snapshot is
//!   an exact, resumable clone of the engine at that position.
//! - A new request hashes its prompt blocks left to right, restores the
//!   deepest cached snapshot (a clone), and prefills only the remainder. Two
//!   different conversations that share a common prefix (e.g. the same system
//!   preamble) therefore reuse each other's blocks — the win over the old
//!   single conversation slot.
//! - The cache is a `HashMap<BlockHash, _>` bounded by `HI_DSV4_PREFIX_CACHE_MB`
//!   (default 2048) with LRU eviction; a snapshot's measured
//!   [`DsV4State::snapshot_bytes`] footprint is charged against the budget.
//!
//! `HI_DSV4_NO_PREFIX_REUSE=1` disables reuse per request (no restore, no
//! snapshotting); `/health`'s dsv4 object reports `reused_tokens` (tokens
//! restored from a snapshot) / `prefilled_tokens` counters.
//!
//! CUDA thread discipline: the engine is created, used, and dropped on ONE
//! dedicated OS thread that owns every device resource (the engine's provider
//! is single-threaded by design — RefCell state, thread-local buffer pool).
//! Requests flow in over a `std::sync::mpsc` channel (FIFO); per-token events
//! flow back through a per-request `tokio::sync::mpsc` unbounded channel that
//! the async `stream_generate` wraps into a [`GenerationStream`]. Dropping the
//! backend closes the request channel, which ends the worker loop and drops
//! the engine on its owning thread; dropping a response stream mid-generation
//! closes its event channel, which the decode loop observes and aborts.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures_util::stream;
use hi_gguf::{GgufFile, GgufTokenizer, StreamingTokenDecoder, inspect_model};
use hi_local_core::backend::{
    BackendHealth, GenerationEvent, GenerationOutput, GenerationRequest, GenerationStream,
    InferenceBackend, SamplingDefaults,
};
use hi_local_core::model::ModelInfo;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::dsv4_cpu::{DsV4Engine, DsV4State, DsV4TapConfig, DsV4Taps};
use crate::dsv4_gpu::{DeepSeekV4GpuEngine, DsV4GpuLinear};
use crate::qwen_cpu::{argmax, sample_from_logits_with_rng};

/// Advertised context window. The GGUF declares 1M, but long-context behavior
/// (YARN is deliberately ignored at bring-up, matching the MLX reference) is
/// unvalidated, so `/v1/models` and `/health` advertise this instead. The
/// engine itself still enforces the GGUF's true context length.
const DSV4_ADVERTISED_CONTEXT: u32 = 32_768;
/// Advertised per-request output budget (same bring-up caution).
const DSV4_ADVERTISED_MAX_OUTPUT_TOKENS: u32 = 4_096;

/// Default prefix-cache block size in tokens (`HI_DSV4_PREFIX_BLOCK_SIZE`).
/// A multiple of every real compress ratio (128/4/2), so snapshots land on
/// clean compressor boundaries. Overridable so the tiny fixture (64-token
/// context) can still form blocks.
const DSV4_PREFIX_BLOCK_SIZE_DEFAULT: usize = 256;

/// Default prefix-cache byte budget in MiB (`HI_DSV4_PREFIX_CACHE_MB`). At
/// ~100-200 MB per snapshot on the real model this holds ~10-20 snapshots.
const DSV4_PREFIX_CACHE_MB_DEFAULT: usize = 2048;

/// Constant seed folded into every block hash so the rolling chain is stable
/// within a process run (`DefaultHasher` uses fixed keys). The cache is never
/// persisted, so cross-version hash stability is not required.
const DSV4_BLOCK_HASH_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// A block-boundary rolling hash: `hash(seed, parent_hash, block_tokens)`,
/// mirroring vLLM's `hash_block_tokens` Merkle chain. Changing any token in a
/// block (or any earlier block) changes this and every later block hash.
type BlockHash = u64;

/// `HI_DSV4_NO_PREFIX_REUSE=1` kills conversation prefix reuse: the request
/// neither restores from nor writes to the cache and prefills from scratch.
/// Read per request so the switch works without reloading the model.
fn prefix_reuse_disabled() -> bool {
    std::env::var("HI_DSV4_NO_PREFIX_REUSE").ok().as_deref() == Some("1")
}

/// Default draft length per verify step (`HI_DSV4_SPEC_K`) — the oracle-test
/// cap; Stage B/C drafters may want other values.
const DSV4_SPEC_K_DEFAULT: usize = 4;

/// `HI_DSV4_SPEC_K`: draft tokens proposed per verify step. Read per request
/// (like the prefix-reuse switch) so it can be tuned without a reload; 0 is
/// legal and degrades the speculative loop to sequential decode through
/// 1-token verify chunks.
fn spec_k_from_env() -> usize {
    let Ok(raw) = std::env::var("HI_DSV4_SPEC_K") else {
        return DSV4_SPEC_K_DEFAULT;
    };
    match raw.trim().parse::<usize>() {
        Ok(k) => k,
        Err(_) => {
            eprintln!(
                "ignoring invalid HI_DSV4_SPEC_K '{raw}' (want an integer >= 0); using {DSV4_SPEC_K_DEFAULT}"
            );
            DSV4_SPEC_K_DEFAULT
        }
    }
}

/// Everything a [`Drafter`] sees at proposal time.
pub(crate) struct DraftContext<'a> {
    /// Prompt plus every emitted token so far, INCLUDING the pending token the
    /// verify step is about to feed — drafts continue after the last entry.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) tokens: &'a [u32],
    /// Hidden taps captured so far: absolute positions
    /// `taps.base() .. tokens.len()-1` (the pending token has not been
    /// forwarded yet, and nothing exists below the base — a prefix-cache
    /// restore skips the restored prefix, so drafters cold-start their own
    /// state at the base). `None` unless the drafter's
    /// [`Drafter::tap_config`] requested capture.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) taps: Option<&'a DsV4Taps>,
    /// Maximum drafts the loop will verify this step.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) k: usize,
}

/// A speculative-decoding draft source for the greedy verify loop (Stage A of
/// `docs/deepseek-v4-spec-decode-plan.md`). Implementations live on the
/// engine worker thread; `MtpDrafter` (Stage B) and `DFlashDrafter` (Stage C)
/// implement this and get selected through `HI_DSV4_SPEC` in
/// [`drafter_from_env`]. Stage A ships the machinery plus the test-only
/// oracle implementation.
pub(crate) trait Drafter {
    /// Hidden-state capture this drafter needs (default: none). Read once per
    /// request BEFORE prefill, and the capture buffer attaches at the
    /// prefix-cache restore point ([`DsV4Taps::base`]): positions from the
    /// base onward are gap-free, positions below it were restored — not
    /// recomputed — so no taps exist there and drafters cold-start at the
    /// base.
    fn tap_config(&self) -> DsV4TapConfig {
        DsV4TapConfig::default()
    }

    /// Propose up to `ctx.k` draft tokens continuing `ctx.tokens`. Returning
    /// fewer (or none) is fine — the loop verifies whatever comes back and
    /// truncates any excess.
    fn propose(&mut self, ctx: &DraftContext<'_>) -> Vec<u32>;

    /// Verify-outcome feedback: of `proposed` drafts, the first `accepted`
    /// were accepted, and `emitted` is the correction-or-bonus token the loop
    /// emitted right after them (the next iteration's pending token).
    fn observe_accepted(&mut self, proposed: usize, accepted: usize, emitted: u32) {
        let _ = (proposed, accepted, emitted);
    }
}

/// Builds the worker's drafter after the engine has loaded ON the worker
/// thread (Stage B/C drafters own device resources, so construction must
/// happen there). `None` disables speculative decoding.
type DsV4DrafterFactory = Box<dyn FnOnce(&DeepSeekV4GpuEngine) -> Option<Box<dyn Drafter>> + Send>;

/// `HI_DSV4_SPEC`: drafter selection, resolved on the engine worker thread so
/// drafters can own device resources. Named drafters land in the follow-up
/// commits; until then every selection logs and stays off.
fn drafter_from_env(engine: &DeepSeekV4GpuEngine) -> Option<Box<dyn Drafter>> {
    let raw = std::env::var("HI_DSV4_SPEC").ok()?;
    match raw.trim() {
        "" | "0" | "off" | "none" => None,
        other => {
            eprintln!(
                "HI_DSV4_SPEC={other} is not available yet; speculative decoding stays off"
            );
            None
        }
    }
}

/// Speculative-decoding counters surfaced through the `/health` dsv4 object.
#[derive(Debug, Default)]
struct DsV4SpecStats {
    /// Draft tokens proposed by the drafter (verified but not necessarily
    /// accepted).
    proposed_tokens: AtomicU64,
    /// Proposed tokens greedily accepted.
    accepted_tokens: AtomicU64,
    /// Verify forwards executed (each covers one proposal batch + 1).
    verify_steps: AtomicU64,
}

/// Static prefix-cache configuration, resolved once at load from the
/// environment (or injected directly by tests).
#[derive(Debug, Clone, Copy)]
struct PrefixCacheConfig {
    block_size: usize,
    budget_bytes: usize,
}

impl PrefixCacheConfig {
    /// Read `HI_DSV4_PREFIX_BLOCK_SIZE` (default 256, must be > 0) and
    /// `HI_DSV4_PREFIX_CACHE_MB` (default 2048).
    fn from_env() -> Self {
        let block_size = std::env::var("HI_DSV4_PREFIX_BLOCK_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DSV4_PREFIX_BLOCK_SIZE_DEFAULT);
        let mb = std::env::var("HI_DSV4_PREFIX_CACHE_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DSV4_PREFIX_CACHE_MB_DEFAULT);
        Self {
            block_size,
            budget_bytes: mb.saturating_mul(1024 * 1024),
        }
    }
}

/// Prefix-reuse counters surfaced through the `/health` dsv4 object.
#[derive(Debug, Default)]
struct DsV4ReuseStats {
    /// Tokens restored from a snapshot instead of prefilled.
    reused_tokens: AtomicU64,
    /// Tokens actually prefilled (full prompts and suffixes alike).
    prefilled_tokens: AtomicU64,
}

/// One cached engine-state snapshot at a block boundary.
struct CachedSnapshot {
    /// Exact engine state after processing `position` tokens; cloned on a hit.
    state: DsV4State,
    /// Token count the snapshot has processed; always a multiple of the cache
    /// block size and equal to `state.pos()`.
    position: usize,
    /// Measured host footprint charged against the LRU budget.
    bytes: usize,
    /// Monotonic tick of the most recent restore/insert touch (LRU key).
    last_used: u64,
}

/// Block-hash prefix cache shared across every conversation the worker serves:
/// a `HashMap<BlockHash, CachedSnapshot>` bounded by a byte budget with
/// least-recently-used eviction. Single-threaded — owned by the engine worker
/// thread and borrowed mutably per request.
struct DsV4PrefixCache {
    block_size: usize,
    budget_bytes: usize,
    used_bytes: usize,
    clock: u64,
    map: HashMap<BlockHash, CachedSnapshot>,
}

impl DsV4PrefixCache {
    fn new(config: PrefixCacheConfig) -> Self {
        Self {
            block_size: config.block_size,
            budget_bytes: config.budget_bytes,
            used_bytes: 0,
            clock: 0,
            map: HashMap::new(),
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// vLLM-style rolling Merkle chain link: `hash(seed, parent, tokens)`.
    fn hash_block(parent: Option<BlockHash>, tokens: &[u32]) -> BlockHash {
        let mut hasher = DefaultHasher::new();
        DSV4_BLOCK_HASH_SEED.hash(&mut hasher);
        // A distinct discriminant for the first block (no parent) vs a parent
        // that happens to hash to 0.
        match parent {
            Some(parent) => (1u8, parent).hash(&mut hasher),
            None => 0u8.hash(&mut hasher),
        }
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Chained hash of every FULL block of `tokens`, left to right. Entry `i`
    /// fingerprints the prefix ending at token `(i + 1) * block_size`.
    fn chain_hashes(&self, tokens: &[u32]) -> Vec<BlockHash> {
        let mut hashes = Vec::with_capacity(tokens.len() / self.block_size);
        let mut parent = None;
        let mut start = 0;
        while start + self.block_size <= tokens.len() {
            let hash = Self::hash_block(parent, &tokens[start..start + self.block_size]);
            hashes.push(hash);
            parent = Some(hash);
            start += self.block_size;
        }
        hashes
    }

    /// Deepest cached snapshot that is a prefix of `prompt`, capped so at least
    /// one prompt token is always left to prefill (fresh last-token logits).
    /// Returns the restore point on a hit and marks it most-recently-used.
    fn restore(&mut self, prompt: &[u32]) -> Option<PrefixRestore> {
        let hashes = self.chain_hashes(prompt);
        let cap = prompt.len().saturating_sub(1);
        for (index, &hash) in hashes.iter().enumerate().rev() {
            let position = (index + 1) * self.block_size;
            if position > cap {
                continue;
            }
            if self.map.contains_key(&hash) {
                let tick = self.next_tick();
                let entry = self.map.get_mut(&hash).expect("checked above");
                entry.last_used = tick;
                debug_assert_eq!(entry.position, position);
                return Some(PrefixRestore {
                    state: entry.state.clone(),
                    position,
                    parent_hash: hash,
                    blocks_done: index + 1,
                });
            }
        }
        None
    }

    /// Insert a boundary snapshot, or refresh its LRU tick if already present.
    /// Clones `state` only on a miss; the caller guarantees
    /// `state.pos() == position` (an exact block boundary).
    fn store(&mut self, hash: BlockHash, state: &DsV4State, position: usize) {
        let tick = self.next_tick();
        if let Some(entry) = self.map.get_mut(&hash) {
            entry.last_used = tick;
            return;
        }
        let bytes = state.snapshot_bytes();
        // A snapshot larger than the whole budget can never be cached; don't
        // evict the rest of the cache trying to make room for it.
        if bytes > self.budget_bytes {
            return;
        }
        self.evict_to_fit(bytes);
        self.used_bytes += bytes;
        self.map.insert(
            hash,
            CachedSnapshot {
                state: state.clone(),
                position,
                bytes,
                last_used: tick,
            },
        );
    }

    /// Evict least-recently-used snapshots until `incoming` more bytes fit.
    fn evict_to_fit(&mut self, incoming: usize) {
        while self.used_bytes + incoming > self.budget_bytes {
            let Some((&victim, _)) = self.map.iter().min_by_key(|(_, entry)| entry.last_used)
            else {
                break;
            };
            let entry = self.map.remove(&victim).expect("min key exists");
            self.used_bytes -= entry.bytes;
        }
    }
}

/// The restore point returned by [`DsV4PrefixCache::restore`].
struct PrefixRestore {
    /// Clone of the cached engine state at `position`.
    state: DsV4State,
    /// Restored token count (`state.pos()`), a block boundary.
    position: usize,
    /// Hash of the deepest restored block; the parent for the next block the
    /// request completes.
    parent_hash: BlockHash,
    /// Number of full blocks already covered by the restore.
    blocks_done: usize,
}

/// Per-request bookkeeping that snapshots each new block boundary the request
/// crosses (during prefill and decode alike), extending the rolling hash chain
/// over the tokens the engine has fed.
struct BlockTracker {
    block_size: usize,
    /// Full blocks already hashed/snapshotted (restored blocks included).
    blocks_done: usize,
    /// Hash of the last completed block, parent of the next one.
    parent_hash: Option<BlockHash>,
}

impl BlockTracker {
    fn fresh(block_size: usize) -> Self {
        Self {
            block_size,
            blocks_done: 0,
            parent_hash: None,
        }
    }

    fn resumed(block_size: usize, restore: &PrefixRestore) -> Self {
        Self {
            block_size,
            blocks_done: restore.blocks_done,
            parent_hash: Some(restore.parent_hash),
        }
    }

    /// Called after every feed with `state.pos() == fed.len()`. When `fed`
    /// has just reached the next block boundary, snapshot the exact state
    /// there and advance the chain. The caller feeds in boundary-aligned
    /// segments, so at most one block completes per call.
    fn on_feed(&mut self, cache: &mut DsV4PrefixCache, fed: &[u32], state: &DsV4State) {
        debug_assert_eq!(state.pos(), fed.len());
        let boundary = (self.blocks_done + 1) * self.block_size;
        if fed.len() != boundary {
            return;
        }
        let start = self.blocks_done * self.block_size;
        let hash = DsV4PrefixCache::hash_block(self.parent_hash, &fed[start..boundary]);
        cache.store(hash, state, boundary);
        self.parent_hash = Some(hash);
        self.blocks_done += 1;
    }

    /// [`Self::on_feed`] for multi-token feeds (the speculative loop accepts
    /// several tokens per verify): advance the hash chain over EVERY newly
    /// completed boundary so later boundaries keep working, snapshotting the
    /// state directly when a boundary lands exactly at `fed.len()` and
    /// through a clone truncated back to the boundary otherwise (block sizes
    /// are compressor-aligned in production, so the exact truncate normally
    /// succeeds; when it cannot, only that boundary's snapshot is skipped).
    fn on_feed_spanning(
        &mut self,
        cache: &mut DsV4PrefixCache,
        engine: &DsV4Engine<DsV4GpuLinear>,
        fed: &[u32],
        state: &DsV4State,
    ) {
        debug_assert_eq!(state.pos(), fed.len());
        loop {
            let boundary = (self.blocks_done + 1) * self.block_size;
            if boundary > fed.len() {
                return;
            }
            let start = self.blocks_done * self.block_size;
            let hash = DsV4PrefixCache::hash_block(self.parent_hash, &fed[start..boundary]);
            if boundary == fed.len() {
                cache.store(hash, state, boundary);
            } else {
                let mut clone = state.clone();
                if engine.truncate_state_to_at_most(&mut clone, boundary) == Some(boundary) {
                    cache.store(hash, &clone, boundary);
                }
            }
            self.parent_hash = Some(hash);
            self.blocks_done += 1;
        }
    }
}

/// One queued generation: the request plus the stream half its events feed.
struct DsV4Job {
    request: GenerationRequest,
    events: tokio::sync::mpsc::UnboundedSender<Result<GenerationEvent>>,
}

/// `InferenceBackend` over the DeepSeek-V4 GPU engine. Construction blocks
/// until the engine thread has loaded weights onto the device (or reports why
/// it could not).
#[derive(Debug)]
pub struct DeepSeekV4Backend {
    model: ModelInfo,
    chat_template: Option<String>,
    /// `BackendHealth::quantization` is assembled per health() call (the
    /// server parses `execution=`/`dsv4=`/`scheduler=` segments out of it):
    /// these are its static ingredients, and `reuse` supplies the live
    /// prefix-reuse counters.
    quantization_label: String,
    engine_context_length: u32,
    advertised_context_length: u32,
    reuse: Arc<DsV4ReuseStats>,
    spec: Arc<DsV4SpecStats>,
    prefix_config: PrefixCacheConfig,
    memory_estimate_bytes: u64,
    jobs: mpsc::Sender<DsV4Job>,
}

impl DeepSeekV4Backend {
    pub fn load(path: impl AsRef<Path>, model_id: Option<String>) -> Result<Self> {
        let path = path.as_ref();
        Self::from_gguf(GgufFile::open(path)?, path, model_id)
    }

    /// Takes ownership of the GGUF (expert weights stream from the mmap for the
    /// engine's lifetime) and moves it onto the dedicated engine thread. The
    /// prefix cache is sized from the environment, and the speculative drafter
    /// from `HI_DSV4_SPEC`.
    pub fn from_gguf(gguf: GgufFile, path: &Path, model_id: Option<String>) -> Result<Self> {
        Self::from_gguf_with_prefix_config(
            gguf,
            path,
            model_id,
            PrefixCacheConfig::from_env(),
            Box::new(drafter_from_env),
        )
    }

    /// Open `path` with an explicit prefix-cache configuration. Test-only entry
    /// so fixture suites can pick a small block size (the fixture's 64-token
    /// context cannot span a 256-token block) and a tiny budget for eviction
    /// coverage, without racing on process-global environment variables.
    #[cfg(test)]
    pub(crate) fn load_with_prefix_config(
        path: impl AsRef<Path>,
        model_id: Option<String>,
        block_size: usize,
        budget_bytes: usize,
    ) -> Result<Self> {
        Self::load_with_drafter(path, model_id, block_size, budget_bytes, Box::new(|_| None))
    }

    /// [`Self::load_with_prefix_config`] plus an injected drafter factory —
    /// the oracle speculative-decoding suites' entry point.
    #[cfg(test)]
    pub(crate) fn load_with_drafter(
        path: impl AsRef<Path>,
        model_id: Option<String>,
        block_size: usize,
        budget_bytes: usize,
        drafter: DsV4DrafterFactory,
    ) -> Result<Self> {
        let path = path.as_ref();
        Self::from_gguf_with_prefix_config(
            GgufFile::open(path)?,
            path,
            model_id,
            PrefixCacheConfig {
                block_size,
                budget_bytes,
            },
            drafter,
        )
    }

    fn from_gguf_with_prefix_config(
        gguf: GgufFile,
        path: &Path,
        model_id: Option<String>,
        prefix_config: PrefixCacheConfig,
        drafter: DsV4DrafterFactory,
    ) -> Result<Self> {
        let config = gguf.qwen_config()?;
        if !config.is_deepseek4() {
            bail!(
                "DeepSeek-V4 backend requires a deepseek4 GGUF, got architecture '{}'",
                config.architecture
            );
        }
        let chat_template = gguf.chat_template().map(ToString::to_string);
        let mut model = inspect_model(path, model_id)?;
        let context_length = config.context_length.min(DSV4_ADVERTISED_CONTEXT);
        model.context_length = Some(context_length);
        model.max_output_tokens = DSV4_ADVERTISED_MAX_OUTPUT_TOKENS.min(context_length);
        let memory_estimate_bytes = config.total_tensor_bytes;
        let reuse = Arc::new(DsV4ReuseStats::default());
        let spec = Arc::new(DsV4SpecStats::default());
        let jobs = spawn_engine_worker(gguf, reuse.clone(), spec.clone(), prefix_config, drafter)?;
        Ok(Self {
            model,
            chat_template,
            quantization_label: config.quantization_label(),
            engine_context_length: config.context_length,
            advertised_context_length: context_length,
            reuse,
            spec,
            prefix_config,
            memory_estimate_bytes,
            jobs,
        })
    }
}

#[async_trait]
impl InferenceBackend for DeepSeekV4Backend {
    fn model(&self) -> &ModelInfo {
        &self.model
    }

    fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    fn health(&self) -> BackendHealth {
        let reused_tokens = self.reuse.reused_tokens.load(Ordering::Relaxed);
        let prefilled_tokens = self.reuse.prefilled_tokens.load(Ordering::Relaxed);
        let spec_proposed = self.spec.proposed_tokens.load(Ordering::Relaxed);
        let spec_accepted = self.spec.accepted_tokens.load(Ordering::Relaxed);
        let spec_verify_steps = self.spec.verify_steps.load(Ordering::Relaxed);
        let quantization = format!(
            "{}; execution=gpu; \
             dsv4=enabled(engine=cuda-dsv4,scheduling=single-request-fifo,max_batch_size=1,engine_context_length={},advertised_context_length={},prefix_reuse={},prefix_block_size={},prefix_cache_mb={},reused_tokens={reused_tokens},prefilled_tokens={prefilled_tokens},spec_k={},spec_proposed={spec_proposed},spec_accepted={spec_accepted},spec_verify_steps={spec_verify_steps}); \
             scheduler=disabled; sampling=single",
            self.quantization_label,
            self.engine_context_length,
            self.advertised_context_length,
            if prefix_reuse_disabled() { "off" } else { "on" },
            self.prefix_config.block_size,
            self.prefix_config.budget_bytes / (1024 * 1024),
            spec_k_from_env(),
        );
        BackendHealth {
            backend: "cuda".to_string(),
            ready: true,
            family: self.model.family.label().to_string(),
            quantization,
            context_length: self.model.context_length,
            memory_estimate_bytes: Some(self.memory_estimate_bytes),
        }
    }

    /// Greedy by default, matching the CUDA GPU convention (and the CPU-oracle
    /// parity discipline this engine is validated under).
    fn sampling_defaults(&self) -> SamplingDefaults {
        SamplingDefaults {
            temperature: 0.0,
            top_p: 1.0,
        }
    }

    async fn stream_generate(&self, request: GenerationRequest) -> Result<GenerationStream> {
        crate::validate_generation_sampling_parameters(&request)?;
        crate::validate_generation_max_tokens(&request, self.model.max_output_tokens)?;
        if !request.media_inputs.is_empty() {
            bail!("the DeepSeek-V4 backend is text-only; multimodal inputs are not supported");
        }
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.jobs
            .send(DsV4Job {
                request,
                events: tx,
            })
            .map_err(|_| anyhow!("DeepSeek-V4 engine thread is stopped"))?;
        Ok(Box::pin(stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        })))
    }
}

/// Spawn the dedicated engine thread and block until it reports the engine
/// loaded. The returned sender is the FIFO request queue; dropping it (with
/// every clone) ends the worker loop.
fn spawn_engine_worker(
    gguf: GgufFile,
    reuse: Arc<DsV4ReuseStats>,
    spec: Arc<DsV4SpecStats>,
    prefix_config: PrefixCacheConfig,
    drafter_factory: DsV4DrafterFactory,
) -> Result<mpsc::Sender<DsV4Job>> {
    let (job_tx, job_rx) = mpsc::channel::<DsV4Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
    thread::Builder::new()
        .name("hi-cuda-dsv4-engine".to_string())
        .spawn(move || {
            // Engine construction allocates every CUDA resource on this
            // thread; jobs, the drafter, and the final drop stay here too.
            let engine = match DeepSeekV4GpuEngine::from_gguf(gguf) {
                Ok(engine) => {
                    let _ = ready_tx.send(Ok(()));
                    engine
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let mut drafter = drafter_factory(&engine);
            // FIFO: one generation at a time, in submission order. The
            // block-hash prefix cache is shared across every conversation this
            // worker serves; snapshots are inserted incrementally as each
            // request crosses block boundaries, so an error or a client that
            // disconnects mid-request simply leaves the boundaries reached so
            // far cached for the next request.
            let mut cache = DsV4PrefixCache::new(prefix_config);
            while let Ok(job) = job_rx.recv() {
                let DsV4Job { request, events } = job;
                if let Err(err) = stream_tokens(
                    &engine,
                    &request,
                    &events,
                    &mut cache,
                    &reuse,
                    &spec,
                    &mut drafter,
                ) {
                    let _ = events.send(Err(err));
                }
            }
        })
        .map_err(|err| anyhow!("failed to spawn DeepSeek-V4 engine thread: {err}"))?;
    ready_rx
        .recv()
        .context("DeepSeek-V4 engine thread exited before signalling readiness")?
        .context("loading DeepSeek-V4 GPU engine")?;
    Ok(job_tx)
}

/// Smallest block boundary strictly greater than `position`.
fn next_block_boundary(position: usize, block_size: usize) -> usize {
    (position / block_size + 1) * block_size
}

/// Prefill + decode for one request, emitting a `TokenDelta` per completed
/// text fragment and a final `Finished`. A closed event channel (client gone)
/// aborts generation silently; errors are returned for the caller to forward.
/// The shared prefix cache is read (deepest-prefix restore) and written
/// (boundary snapshots) in place as the request progresses; nothing is
/// restored or written under the `HI_DSV4_NO_PREFIX_REUSE=1` kill switch.
/// A configured `drafter` engages the speculative verify loop
/// ([`decode_speculative`]) for greedy requests; sampled requests fall back
/// to the sequential loop below (greedy acceptance is the only mode whose
/// output is provably identical — Stage A scope).
fn stream_tokens(
    engine: &DeepSeekV4GpuEngine,
    request: &GenerationRequest,
    events: &tokio::sync::mpsc::UnboundedSender<Result<GenerationEvent>>,
    cache: &mut DsV4PrefixCache,
    stats: &DsV4ReuseStats,
    spec: &DsV4SpecStats,
    drafter: &mut Option<Box<dyn Drafter>>,
) -> Result<()> {
    let inner = engine.engine();
    let tokenizer = inner.tokenizer();
    let config = inner.config();

    let prompt_tokens = tokenizer.encode(&request.prompt)?;
    if prompt_tokens.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    let max_tokens =
        usize::try_from(request.max_tokens).context("max_tokens does not fit usize")?;
    validate_context_budget(config.context_length, prompt_tokens.len(), max_tokens)?;

    // Speculative decode is greedy-only in Stage A; sampled requests decode
    // sequentially.
    let drafter = drafter
        .as_deref_mut()
        .filter(|_| request.temperature <= 0.0);
    let spec_k = spec_k_from_env();

    // Restore the deepest cached block prefix of this prompt (shared across
    // conversations), then prefill only the remainder. `fed` tracks exactly the
    // tokens the state has processed; `tracker` snapshots every new boundary the
    // request crosses. Both are inert under the kill switch (`tracker` = None).
    let reuse_disabled = prefix_reuse_disabled();
    let block_size = cache.block_size;
    let restore = (!reuse_disabled)
        .then(|| cache.restore(&prompt_tokens))
        .flatten();
    let (mut state, mut fed, mut tracker) = match restore {
        Some(restore) => {
            stats
                .reused_tokens
                .fetch_add(restore.position as u64, Ordering::Relaxed);
            let fed = prompt_tokens[..restore.position].to_vec();
            let tracker = BlockTracker::resumed(block_size, &restore);
            (restore.state, fed, Some(tracker))
        }
        None => {
            let tracker = (!reuse_disabled).then(|| BlockTracker::fresh(block_size));
            (inner.new_state(), Vec::new(), tracker)
        }
    };
    // Drafter taps attach AT the restore point: the restored prefix's
    // activations were never recomputed, so the buffer's base marks where
    // capture (and the drafter's own context) begins — fresh-session restores
    // stay as cheap with speculation on as off.
    let tap_config = drafter
        .as_ref()
        .map(|drafter| drafter.tap_config())
        .unwrap_or_default();
    let mut taps = if tap_config.is_empty() {
        None
    } else {
        Some(inner.new_taps_at(tap_config, fed.len())?)
    };
    if drafter.is_some() {
        // Verify rollbacks rewind up to spec_k rejected positions, plus a
        // compressor block-boundary round-down (see `rewind_state_to`), so
        // the request's ring retention must cover that window. Restored
        // snapshots may carry less slack; bumping the field only extends
        // FUTURE retention (the first rewinds may re-feed more — correct,
        // just slower), and severs any device-state link so the documented
        // "equal (tag, pos) implies equal content" invariant keeps holding.
        let slack = spec_ring_slack(inner, spec_k);
        if state.ring_slack < slack {
            state.ring_slack = slack;
            state.device_tag = 0;
        }
    }

    // Prefill the (possibly suffix-only) prompt remainder in block-aligned
    // segments so each crossed boundary is snapshotted at its exact position;
    // `inner.prefill` still batches each segment by `HI_DSV4_PREFILL_CHUNK`.
    // Bail out between segments if the client has already gone away.
    let mut logits = Vec::new();
    while fed.len() < prompt_tokens.len() {
        if events.is_closed() {
            return Ok(());
        }
        let seg_end = next_block_boundary(fed.len(), block_size).min(prompt_tokens.len());
        let piece = &prompt_tokens[fed.len()..seg_end];
        logits = inner.prefill_with_taps(&mut state, piece, taps.as_mut())?;
        stats
            .prefilled_tokens
            .fetch_add(piece.len() as u64, Ordering::Relaxed);
        fed.extend_from_slice(piece);
        if let Some(tracker) = tracker.as_mut() {
            tracker.on_feed(cache, &fed, &state);
        }
    }
    if logits.is_empty() {
        bail!("DeepSeek-V4 prefill produced no logits");
    }

    if let Some(drafter) = drafter {
        return decode_speculative(SpecDecodeArgs {
            inner,
            request,
            events,
            cache,
            spec,
            drafter,
            state,
            fed,
            tracker,
            taps,
            logits,
            prompt_len: prompt_tokens.len() as u64,
            max_tokens,
            k_cap: spec_k,
        });
    }

    // Same sampling semantics as the engine's QwenCpuRunOptions path: the
    // shared `sample_from_logits_with_rng` (greedy at temperature<=0, top-k
    // truncation, top-p nucleus), seeded deterministically when requested.
    let mut seeded_rng = request.seed.map(StdRng::seed_from_u64);
    let mut thread_rng = rand::thread_rng();
    let mut decoder = tokenizer.streaming_decoder(true);
    let mut text = String::new();
    let mut completion_tokens = 0u64;
    for step in 0..max_tokens {
        let next = match &mut seeded_rng {
            Some(rng) => sample_from_logits_with_rng(
                &logits,
                request.temperature,
                request.top_p,
                request.top_k,
                rng,
            )?,
            None => sample_from_logits_with_rng(
                &logits,
                request.temperature,
                request.top_p,
                request.top_k,
                &mut thread_rng,
            )?,
        };
        completion_tokens += 1;
        let delta = decoder.push(tokenizer, next)?;
        if !delta.is_empty() {
            text.push_str(&delta);
            if events
                .send(Ok(GenerationEvent::TokenDelta {
                    token_id: next,
                    text: delta,
                }))
                .is_err()
            {
                return Ok(());
            }
        }
        if Some(next) == config.eos_token_id {
            break;
        }
        // Stop sequences end generation here; the served text is truncated at
        // the match by the HTTP layer (`truncate_at_stop` / StopStreamFilter),
        // exactly as it is for the scheduler backends.
        if stop_sequence_hit(&text, &request.stop_sequences) {
            break;
        }
        if step + 1 < max_tokens {
            if events.is_closed() {
                return Ok(());
            }
            logits = inner.step(&mut state, next)?;
            fed.push(next);
            // Generated tokens extend the conversation, so decode crosses block
            // boundaries too; snapshot them for the next turn's deeper reuse.
            if let Some(tracker) = tracker.as_mut() {
                tracker.on_feed(cache, &fed, &state);
            }
        }
    }

    let _ = events.send(Ok(GenerationEvent::Finished {
        output: GenerationOutput {
            text,
            prompt_tokens: prompt_tokens.len() as u64,
            completion_tokens,
        },
    }));
    Ok(())
}

/// Ring slack for speculative-decode states: a verify overshoots by up to
/// `k` rejected positions, and rewinding across a completed compressor block
/// rounds down to that block's boundary — up to `max_ratio - 1` further —
/// so retain `max_ratio + k + 1` extra ring entries (~12 MB on the real
/// model; see `DsV4Engine::rewind_state_to`).
fn spec_ring_slack(engine: &DsV4Engine<DsV4GpuLinear>, k: usize) -> usize {
    let max_ratio = engine
        .layers()
        .iter()
        .filter_map(|layer| layer.compressor.as_ref().map(|weights| weights.ratio))
        .max()
        .unwrap_or(0);
    max_ratio + k + 1
}

/// Borrowed request context for [`decode_speculative`] (one bundle instead of
/// a dozen loose parameters).
struct SpecDecodeArgs<'a> {
    inner: &'a DsV4Engine<DsV4GpuLinear>,
    request: &'a GenerationRequest,
    events: &'a tokio::sync::mpsc::UnboundedSender<Result<GenerationEvent>>,
    cache: &'a mut DsV4PrefixCache,
    spec: &'a DsV4SpecStats,
    drafter: &'a mut dyn Drafter,
    state: DsV4State,
    /// Tokens the state validly contains (prompt so far; grows by the pending
    /// token + accepted drafts after every verify rollback).
    fed: Vec<u32>,
    tracker: Option<BlockTracker>,
    taps: Option<DsV4Taps>,
    /// Logits after the last prefilled prompt token.
    logits: Vec<f32>,
    prompt_len: u64,
    max_tokens: usize,
    k_cap: usize,
}

/// What one emitted token means for the speculative loop.
enum SpecEmit {
    Continue,
    /// eos, a stop sequence, or the max_tokens budget: send Finished.
    Finish,
    /// The client dropped the stream: abort silently.
    ClientGone,
}

/// The greedy speculative decode loop (Stage A of
/// `docs/deepseek-v4-spec-decode-plan.md`): per iteration, propose up to K
/// drafts, verify the pending token + drafts in ONE chunked forward
/// ([`DsV4Engine::verify_tokens_with_taps`], bit-exact with sequential
/// steps), greedily accept the longest draft prefix whose token i equals the
/// argmax of position i's logits, emit the accepted tokens plus the
/// correction-or-bonus token, and rewind the state to the accepted end. The
/// emitted stream is byte-identical to sequential greedy decode by
/// construction — every emitted token IS a sequential argmax — which the
/// oracle-drafter suites gate end to end.
fn decode_speculative(args: SpecDecodeArgs<'_>) -> Result<()> {
    let SpecDecodeArgs {
        inner,
        request,
        events,
        cache,
        spec,
        drafter,
        mut state,
        mut fed,
        mut tracker,
        mut taps,
        logits,
        prompt_len,
        max_tokens,
        k_cap,
    } = args;
    let tokenizer = inner.tokenizer();
    let eos = inner.config().eos_token_id;
    let mut decoder = tokenizer.streaming_decoder(true);
    let mut text = String::new();
    let mut completion_tokens = 0u64;
    let finish = |text: String, completion_tokens: u64| {
        let _ = events.send(Ok(GenerationEvent::Finished {
            output: GenerationOutput {
                text,
                prompt_tokens: prompt_len,
                completion_tokens,
            },
        }));
        Ok(())
    };

    // The first token comes straight off the prompt logits; `argmax` is the
    // sequential sampler at temperature <= 0 (the speculative gate).
    let mut pending = argmax(&logits)?;
    completion_tokens += 1;
    match emit_spec_token(
        tokenizer,
        &mut decoder,
        events,
        &mut text,
        pending,
        eos,
        &request.stop_sequences,
        completion_tokens,
        max_tokens,
    )? {
        SpecEmit::Continue => {}
        SpecEmit::Finish => return finish(text, completion_tokens),
        SpecEmit::ClientGone => return Ok(()),
    }

    while (completion_tokens as usize) < max_tokens {
        if events.is_closed() {
            return Ok(());
        }
        // The bonus/correction token takes one budget slot, so never propose
        // more drafts than the budget can still emit after it.
        let remaining = max_tokens - completion_tokens as usize;
        let k = k_cap.min(remaining - 1);
        let mut context_tokens = Vec::with_capacity(fed.len() + 1);
        context_tokens.extend_from_slice(&fed);
        context_tokens.push(pending);
        let mut drafts = drafter.propose(&DraftContext {
            tokens: &context_tokens,
            taps: taps.as_ref(),
            k,
        });
        drafts.truncate(k);
        // An out-of-vocab draft would make the verify forward bail; clamp the
        // proposal at the first invalid id instead (drafter bug, not fatal).
        if let Some(bad) = drafts
            .iter()
            .position(|&token| token as usize >= tokenizer.token_count())
        {
            drafts.truncate(bad);
        }

        // Verify pending + drafts in one chunked forward: logits row i is the
        // sequential-greedy distribution for the token AFTER verify_seq[..=i].
        let mut verify_seq = Vec::with_capacity(drafts.len() + 1);
        verify_seq.push(pending);
        verify_seq.extend_from_slice(&drafts);
        let rows = inner.verify_tokens_with_taps(&mut state, &verify_seq, taps.as_mut())?;
        spec.verify_steps.fetch_add(1, Ordering::Relaxed);
        spec.proposed_tokens
            .fetch_add(drafts.len() as u64, Ordering::Relaxed);

        let mut accepted = 0usize;
        while accepted < drafts.len() && drafts[accepted] == argmax(&rows[accepted])? {
            accepted += 1;
        }
        spec.accepted_tokens
            .fetch_add(accepted as u64, Ordering::Relaxed);
        // Correction (a rejected draft's true argmax) or bonus (all accepted).
        let next = argmax(&rows[accepted])?;
        drafter.observe_accepted(drafts.len(), accepted, next);

        // Emit the accepted drafts then the correction/bonus token; any of
        // them can end the request (eos / stop sequence / budget), exactly
        // where the sequential loop would have stopped.
        let mut finished = false;
        let mut client_gone = false;
        for &token in drafts[..accepted].iter().chain(std::iter::once(&next)) {
            completion_tokens += 1;
            match emit_spec_token(
                tokenizer,
                &mut decoder,
                events,
                &mut text,
                token,
                eos,
                &request.stop_sequences,
                completion_tokens,
                max_tokens,
            )? {
                SpecEmit::Continue => {}
                SpecEmit::Finish => {
                    finished = true;
                    break;
                }
                SpecEmit::ClientGone => {
                    client_gone = true;
                    break;
                }
            }
        }
        if client_gone {
            return Ok(());
        }
        if finished {
            return finish(text, completion_tokens);
        }

        // The verify advanced the state over every draft; rewind to the
        // accepted end (`next` is NOT fed — it is the next pending token) and
        // only then extend the prefix-cache chain, so snapshots never contain
        // rejected positions.
        fed.push(pending);
        fed.extend_from_slice(&drafts[..accepted]);
        inner.rewind_state_to(&mut state, &fed, fed.len(), taps.as_mut())?;
        if let Some(tracker) = tracker.as_mut() {
            tracker.on_feed_spanning(cache, inner, &fed, &state);
        }
        pending = next;
    }
    finish(text, completion_tokens)
}

/// Push one token of the speculative loop through the streaming decoder and
/// the event channel, mirroring the sequential loop's emission semantics
/// (delta only when non-empty; eos, then stop sequences, then the budget).
#[allow(clippy::too_many_arguments)]
fn emit_spec_token(
    tokenizer: &GgufTokenizer,
    decoder: &mut StreamingTokenDecoder,
    events: &tokio::sync::mpsc::UnboundedSender<Result<GenerationEvent>>,
    text: &mut String,
    token: u32,
    eos: Option<u32>,
    stop_sequences: &[String],
    completion_tokens: u64,
    max_tokens: usize,
) -> Result<SpecEmit> {
    let delta = decoder.push(tokenizer, token)?;
    if !delta.is_empty() {
        text.push_str(&delta);
        if events
            .send(Ok(GenerationEvent::TokenDelta {
                token_id: token,
                text: delta,
            }))
            .is_err()
        {
            return Ok(SpecEmit::ClientGone);
        }
    }
    if Some(token) == eos
        || stop_sequence_hit(text, stop_sequences)
        || completion_tokens as usize >= max_tokens
    {
        return Ok(SpecEmit::Finish);
    }
    Ok(SpecEmit::Continue)
}

/// Mirror of the qwen path's context budget check, against the ENGINE's true
/// context length (the advertised window only caps what `/v1/models` reports).
fn validate_context_budget(
    context_length: u32,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<()> {
    let context =
        usize::try_from(context_length).context("deepseek4 context_length does not fit usize")?;
    if max_tokens == 0 {
        bail!("invalid_request_parameter: max_tokens must be greater than 0");
    }
    if prompt_len > context {
        bail!(
            "context_length_exceeded: prompt length {prompt_len} exceeds deepseek4 context length {context}"
        );
    }
    if prompt_len.saturating_add(max_tokens) > context {
        bail!(
            "context_length_exceeded: prompt length {prompt_len} plus max_tokens {max_tokens} exceeds deepseek4 context length {context}"
        );
    }
    Ok(())
}

/// Has any stop sequence fully appeared in the accumulated completion text?
fn stop_sequence_hit(text: &str, stop_sequences: &[String]) -> bool {
    stop_sequences
        .iter()
        .any(|stop| !stop.is_empty() && text.contains(stop.as_str()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::{Request, StatusCode};
    use futures_util::StreamExt;
    use hi_local_core::backend::SharedBackend;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use std::collections::HashSet;
    use std::sync::Mutex;

    use crate::dsv4_cpu::DeepSeekV4CpuReference;
    use crate::dsv4_cpu::fixture::{
        tempfile_path, write_deepseek4_gguf, write_deepseek4_spec_gguf,
    };
    use crate::qwen_cpu::QwenCpuRunOptions;

    use super::*;

    fn fixture_backend(name: &str) -> Arc<DeepSeekV4Backend> {
        let path = tempfile_path(name);
        write_deepseek4_gguf(&path);
        Arc::new(DeepSeekV4Backend::load(&path, Some("dsv4-fixture".to_string())).unwrap())
    }

    /// A budget large enough to never evict on the fixture (snapshots are a few
    /// KiB; see `dsv4_snapshot_bytes_grows_with_position`).
    const BIG_PREFIX_BUDGET: usize = 1 << 20;

    /// Fixture backend with an explicit prefix-cache block size and byte budget.
    /// The fixture's 64-token context cannot span the 256-token default block,
    /// so the prefix suites use a small block size; a tiny budget drives the
    /// eviction coverage.
    fn fixture_backend_with_prefix(
        name: &str,
        block_size: usize,
        budget_bytes: usize,
    ) -> Arc<DeepSeekV4Backend> {
        let path = tempfile_path(name);
        write_deepseek4_gguf(&path);
        Arc::new(
            DeepSeekV4Backend::load_with_prefix_config(
                &path,
                Some("dsv4-fixture".to_string()),
                block_size,
                budget_bytes,
            )
            .unwrap(),
        )
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

    #[tokio::test]
    async fn dsv4_backend_reports_v4_contract() {
        let backend = fixture_backend("contract");

        let model = backend.model();
        assert_eq!(model.id, "dsv4-fixture");
        assert_eq!(model.family, hi_local_core::model::ModelFamily::DeepSeek);
        // Fixture context (64) is below the advertised ceiling, so both caps
        // clamp to it; the real model reports 32768 / 4096.
        assert_eq!(model.context_length, Some(64));
        assert_eq!(model.max_output_tokens, 64);

        let health = backend.health();
        assert_eq!(health.backend, "cuda");
        assert!(health.ready);
        assert_eq!(health.family, "deepseek");
        assert!(health.quantization.contains("execution=gpu"));
        assert!(
            health
                .quantization
                .contains("dsv4=enabled(engine=cuda-dsv4")
        );
        assert!(health.quantization.contains("scheduler=disabled"));

        let template = backend.chat_template().unwrap();
        assert!(template.contains("<｜User｜>"));
        assert!(template.contains("</think>"));
        assert!(template.contains("｜DSML｜"));
    }

    #[tokio::test]
    async fn dsv4_backend_streams_deltas_then_finish_deterministically() {
        let backend = fixture_backend("stream");

        let mut deltas = Vec::new();
        let mut finished = None;
        let mut stream = backend
            .stream_generate(generation_request("abcab", 4))
            .await
            .unwrap();
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                GenerationEvent::TokenDelta { token_id, text } => deltas.push((token_id, text)),
                GenerationEvent::Finished { output } => finished = Some(output),
            }
        }
        let finished = finished.expect("stream must end with Finished");
        assert_eq!(finished.prompt_tokens, 5);
        assert!(finished.completion_tokens >= 1 && finished.completion_tokens <= 4);
        let collected: String = deltas.iter().map(|(_, text)| text.as_str()).collect();
        assert_eq!(collected, finished.text);

        // Greedy decode over the same prompt is deterministic, and the second
        // request (queued FIFO behind nothing) reproduces the first exactly.
        let second = backend
            .generate(generation_request("abcab", 4))
            .await
            .unwrap();
        assert_eq!(second.text, finished.text);
        assert_eq!(second.completion_tokens, finished.completion_tokens);
    }

    #[tokio::test]
    async fn dsv4_backend_rejects_overlong_and_zero_budgets() {
        let backend = fixture_backend("budget");

        let err = backend
            .generate(generation_request("abc", 0))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("max_tokens must be greater than 0")
        );

        let err = backend
            .generate(generation_request(&"a".repeat(80), 4))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("context_length_exceeded"),
            "unexpected error: {err}"
        );
    }

    /// Numeric field out of the health quantization string (e.g.
    /// "reused_tokens=5" -> 5).
    fn health_counter(backend: &DeepSeekV4Backend, key: &str) -> u64 {
        let quantization = backend.health().quantization;
        let needle = format!("{key}=");
        let start = quantization
            .find(&needle)
            .unwrap_or_else(|| panic!("{key} missing in {quantization}"))
            + needle.len();
        quantization[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap()
    }

    /// Gate (ii): a second turn extending the first restores the deepest cached
    /// block and prefills only the remainder, producing exactly the output a
    /// cold run produces; a diverging prompt shares no block and prefills fully.
    #[tokio::test]
    async fn dsv4_backend_prefix_reuse_extends_previous_turn() {
        let backend = fixture_backend_with_prefix("reuse", 4, BIG_PREFIX_BUDGET);
        // 24 tokens == 6 whole blocks of 4; the fixture vocab is single-char
        // tokens, so token counts equal char counts.
        let turn1 = "abcabcabcabcabcabcabcabc";
        let first = backend
            .generate(generation_request(turn1, 4))
            .await
            .unwrap();
        let prefilled_turn1 = health_counter(&backend, "prefilled_tokens");
        assert_eq!(prefilled_turn1, 24, "turn 1 prefills the whole prompt");
        assert_eq!(health_counter(&backend, "reused_tokens"), 0);

        // Turn 2 extends turn 1 the way a chat client would: previous prompt
        // + the generated text + new user text. Turn 1 cached block snapshots
        // through position 24 (its last prompt boundary), so turn 2 restores
        // there and prefills only the tail.
        let turn2 = format!("{turn1}{}ba", first.text);
        let second = backend
            .generate(generation_request(&turn2, 4))
            .await
            .unwrap();
        let reused = health_counter(&backend, "reused_tokens");
        let prefilled_turn2 = health_counter(&backend, "prefilled_tokens") - prefilled_turn1;
        assert_eq!(reused, 24, "turn 2 restores turn 1's deepest cached block");
        assert!(
            prefilled_turn2 < turn2.len() as u64,
            "turn 2 must prefill only the suffix ({prefilled_turn2} of {})",
            turn2.len()
        );
        assert_eq!(
            reused + prefilled_turn2,
            turn2.len() as u64,
            "restored + prefilled tokens must cover the prompt exactly"
        );

        // Cold oracle: a fresh backend answering turn 2 from scratch.
        let cold_backend = fixture_backend_with_prefix("reuse-cold", 4, BIG_PREFIX_BUDGET);
        let cold = cold_backend
            .generate(generation_request(&turn2, 4))
            .await
            .unwrap();
        assert_eq!(second.text, cold.text, "reused turn must match a cold run");
        assert_eq!(second.completion_tokens, cold.completion_tokens);

        // A diverging prompt (shares no block prefix) falls back to a full
        // prefill: "cccc" hashes differently from any cached "abc…" block.
        let prefilled_before = health_counter(&backend, "prefilled_tokens");
        backend
            .generate(generation_request("ccccc", 2))
            .await
            .unwrap();
        assert_eq!(health_counter(&backend, "reused_tokens"), reused);
        assert_eq!(
            health_counter(&backend, "prefilled_tokens"),
            prefilled_before + 5
        );
    }

    /// Gate (i): two different conversations sharing a long common system
    /// prefix. The second restores the shared block snapshot the first cached
    /// (not the first's divergent tail) and still matches a cold run.
    #[tokio::test]
    async fn dsv4_backend_shares_prefix_across_conversations() {
        let backend = fixture_backend_with_prefix("shared", 4, BIG_PREFIX_BUDGET);
        // A 16-token (4-block) shared system preamble, then per-conversation
        // user text that diverges at position 16.
        let system = "abababababababab";
        let conv_a = format!("{system}aa");
        let conv_b = format!("{system}bb");

        // Conversation A (cold) caches the shared system blocks at 4/8/12/16.
        backend
            .generate(generation_request(&conv_a, 2))
            .await
            .unwrap();
        assert_eq!(health_counter(&backend, "reused_tokens"), 0);

        // Conversation B is a *different* conversation but shares the system
        // prefix; it must restore the shared block at position 16.
        let served_b = backend
            .generate(generation_request(&conv_b, 2))
            .await
            .unwrap();
        let reused_b = health_counter(&backend, "reused_tokens");
        assert_eq!(
            reused_b,
            system.len() as u64,
            "conversation B restores the whole shared system prefix"
        );

        // Restoring A's shared blocks must not leak A's divergent tail: B still
        // reproduces a cold run of B exactly.
        let cold = fixture_backend_with_prefix("shared-cold", 4, BIG_PREFIX_BUDGET)
            .generate(generation_request(&conv_b, 2))
            .await
            .unwrap();
        assert_eq!(
            served_b.text, cold.text,
            "shared-prefix reuse must match cold"
        );
        assert_eq!(served_b.completion_tokens, cold.completion_tokens);
    }

    /// Gate (iii): under a tiny byte budget the cache evicts, so a continuation
    /// falls back to a shallower cached block (or, when nothing fits, a cold
    /// prefill) — always with cold-identical output. Budgets are chosen from
    /// the measured fixture snapshot sizes (block@4=608, @8=832, @12=1056 B;
    /// see `dsv4_snapshot_bytes_grows_with_position`).
    #[tokio::test]
    async fn dsv4_backend_prefix_cache_evicts_under_budget() {
        // "aaaabbbbaaaa" == blocks @4/@8/@12; the continuation appends "aa".
        let base = "aaaabbbbaaaa";
        let cont = format!("{base}aa");
        let cold = fixture_backend_with_prefix("evict-cold", 4, BIG_PREFIX_BUDGET)
            .generate(generation_request(&cont, 2))
            .await
            .unwrap();

        // Ample budget: the continuation restores the deepest block (@12).
        let big = fixture_backend_with_prefix("evict-big", 4, BIG_PREFIX_BUDGET);
        big.generate(generation_request(base, 2)).await.unwrap();
        let big_out = big.generate(generation_request(&cont, 2)).await.unwrap();
        assert_eq!(health_counter(&big, "reused_tokens"), 12);
        assert_eq!(big_out.text, cold.text);

        // 900 B holds @8 (832) but not @12 (1056), and inserting @8 evicts @4:
        // the continuation falls back to the shallower @8 block.
        let shallow = fixture_backend_with_prefix("evict-shallow", 4, 900);
        shallow.generate(generation_request(base, 2)).await.unwrap();
        let shallow_out = shallow
            .generate(generation_request(&cont, 2))
            .await
            .unwrap();
        assert_eq!(
            health_counter(&shallow, "reused_tokens"),
            8,
            "evicting @12 forces the continuation onto the shallower @8 block"
        );
        assert_eq!(
            shallow_out.text, cold.text,
            "shallower reuse must match cold"
        );

        // 500 B is below even the smallest snapshot (@4=608): nothing caches,
        // so the continuation is a full cold prefill.
        let tiny = fixture_backend_with_prefix("evict-tiny", 4, 500);
        tiny.generate(generation_request(base, 2)).await.unwrap();
        let tiny_out = tiny.generate(generation_request(&cont, 2)).await.unwrap();
        assert_eq!(
            health_counter(&tiny, "reused_tokens"),
            0,
            "a budget below one snapshot caches nothing"
        );
        assert_eq!(tiny_out.text, cold.text, "cold fallback must match cold");
    }

    /// Gate (iv): the rolling hash chain invalidates every block after an
    /// edited token. Re-running the identical prompt reuses two blocks, but
    /// changing a token inside block 2 drops reuse back to block 1 only —
    /// even though block 3's tokens are unchanged.
    #[tokio::test]
    async fn dsv4_backend_hash_chain_invalidates_later_blocks() {
        let backend = fixture_backend_with_prefix("hashchain", 4, BIG_PREFIX_BUDGET);
        // Blocks: "aaaa" | "bbbb" | "aaaa" (12 tokens, positions 4/8/12).
        let conv1 = "aaaabbbbaaaa";
        backend
            .generate(generation_request(conv1, 2))
            .await
            .unwrap();

        // Identical rerun reuses blocks 1-2 (block 3 at position 12 is held back
        // by the "leave one token to prefill" cap on the 12-token prompt).
        let before_rerun = health_counter(&backend, "reused_tokens");
        backend
            .generate(generation_request(conv1, 2))
            .await
            .unwrap();
        assert_eq!(
            health_counter(&backend, "reused_tokens") - before_rerun,
            8,
            "an identical prompt reuses blocks 1 and 2"
        );

        // Edit the first token of block 2 ("bbbb" -> "abbb"). Block 1 is
        // unchanged, so it is still reused, but block 2's hash and therefore
        // block 3's chained hash both change: reuse must stop at block 1.
        let conv2 = "aaaaabbbaaaa";
        let before_edit = health_counter(&backend, "reused_tokens");
        let served = backend
            .generate(generation_request(conv2, 2))
            .await
            .unwrap();
        assert_eq!(
            health_counter(&backend, "reused_tokens") - before_edit,
            4,
            "a mid-prompt edit invalidates every later block despite matching tail tokens"
        );

        let cold = fixture_backend_with_prefix("hashchain-cold", 4, BIG_PREFIX_BUDGET)
            .generate(generation_request(conv2, 2))
            .await
            .unwrap();
        assert_eq!(served.text, cold.text);
    }

    /// The `HI_DSV4_NO_PREFIX_REUSE=1` kill switch: every prompt prefills from
    /// scratch (reused stays 0) regardless of cached blocks. Ignored by default
    /// because it depends on a process-global env var; run it in isolation:
    /// `HI_DSV4_NO_PREFIX_REUSE=1 cargo test -p hi-cuda --release --features \
    ///  native-cuda dsv4_backend_kill_switch -- --ignored`
    #[tokio::test]
    #[ignore = "reads the process-global HI_DSV4_NO_PREFIX_REUSE; run in isolation"]
    async fn dsv4_backend_kill_switch_disables_reuse() {
        assert_eq!(
            std::env::var("HI_DSV4_NO_PREFIX_REUSE").ok().as_deref(),
            Some("1"),
            "run this test with HI_DSV4_NO_PREFIX_REUSE=1 set"
        );
        let backend = fixture_backend_with_prefix("killswitch", 4, BIG_PREFIX_BUDGET);
        assert!(backend.health().quantization.contains("prefix_reuse=off"));

        let prompt = "aaaabbbbaaaa";
        backend
            .generate(generation_request(prompt, 2))
            .await
            .unwrap();
        let prefilled_before = health_counter(&backend, "prefilled_tokens");
        // Re-issuing the identical prompt would restore a block with reuse on;
        // the kill switch forces a full prefill and leaves reused at 0.
        backend
            .generate(generation_request(prompt, 2))
            .await
            .unwrap();
        assert_eq!(health_counter(&backend, "reused_tokens"), 0);
        assert_eq!(
            health_counter(&backend, "prefilled_tokens"),
            prefilled_before + prompt.len() as u64
        );
    }

    /// Real-model prefix-reuse demonstration: turn 2 extending a ~1.5k-token
    /// turn 1 must prefill only the suffix and finish far faster. Run
    /// explicitly:
    /// `CUDA_VISIBLE_DEVICES=0 cargo test -p hi-cuda --release --features native-cuda \
    ///  dsv4_real_model_prefix_reuse -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    async fn dsv4_real_model_prefix_reuse_timing() {
        let Some(path) = crate::dsv4_gpu::tests::real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let backend =
            Arc::new(DeepSeekV4Backend::load(&path, Some("dsv4-real".to_string())).unwrap());
        let turn1 = crate::dsv4_gpu::tests::long_prompt(1500);

        let started = std::time::Instant::now();
        let first = backend
            .generate(generation_request(&turn1, 8))
            .await
            .unwrap();
        let turn1_elapsed = started.elapsed();
        let prefilled_turn1 = health_counter(&backend, "prefilled_tokens");

        // Turn 2: the same conversation extended by the model's reply and a
        // new user message.
        let turn2 = format!(
            "{turn1}{} Now summarize the previous text in one sentence.",
            first.text
        );
        let started = std::time::Instant::now();
        backend
            .generate(generation_request(&turn2, 8))
            .await
            .unwrap();
        let turn2_elapsed = started.elapsed();
        let reused = health_counter(&backend, "reused_tokens");
        let prefilled_turn2 = health_counter(&backend, "prefilled_tokens") - prefilled_turn1;
        eprintln!(
            "real-model prefix reuse: turn 1 {:.1}s ({prefilled_turn1} tokens prefilled), \
             turn 2 {:.1}s ({reused} reused, {prefilled_turn2} prefilled)",
            turn1_elapsed.as_secs_f64(),
            turn2_elapsed.as_secs_f64(),
        );
        assert!(reused > 0, "turn 2 must reuse the cached conversation");
        assert!(
            prefilled_turn2 < prefilled_turn1 / 4,
            "turn 2 must prefill only the suffix ({prefilled_turn2} vs turn 1's {prefilled_turn1})"
        );
    }

    /// The Stage-2b serving-regression trio (production repro): (1) a cold
    /// ~4.6k-token prompt — long enough to cross the 2048-token indexer
    /// top-512 threshold (where the original serial device selection cost
    /// ~275 ms per decoded token) and ~18 prefix-cache boundaries — must stay
    /// in the pre-regression time class; (2) a second identical conversation
    /// must restore the deep prefix (reused > 0) and finish in seconds-class
    /// time; (3) a ~1.3k-token prompt (below the threshold) stays healthy.
    /// Mirrors the production service env when run with
    /// `HI_DSV4_PREFILL_GEMM=1`. Run explicitly:
    /// `HI_DSV4_PREFILL_GEMM=1 HI_DSV4_EXPERT_POOL_GB=40 CUDA_VISIBLE_DEVICES=0 \
    ///  cargo test -p hi-cuda --release --features native-cuda \
    ///  dsv4_real_model_long_context_reuse_trio -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "needs the real DeepSeek-V4-Flash checkpoint and an otherwise-idle GPU"]
    async fn dsv4_real_model_long_context_reuse_trio() {
        let Some(path) = crate::dsv4_gpu::tests::real_model_path() else {
            eprintln!("skipping: real model not found");
            return;
        };
        let backend =
            Arc::new(DeepSeekV4Backend::load(&path, Some("dsv4-real".to_string())).unwrap());
        // long_prompt repeats a ~37-token sentence min_tokens/8 times, so it
        // overshoots its argument ~4.6x; 1000 yields ~4.6k tokens — the
        // production scenario (crosses the 2048-token indexer threshold and
        // ~18 prefix-cache boundaries at block size 256).
        let preamble = crate::dsv4_gpu::tests::long_prompt(1000);
        // Seeded sampling instead of greedy: the repetitive preamble makes
        // greedy emit EOS almost immediately, which would leave the >2048
        // decode path (where the regression lived) unexercised.
        let decode_request = |prompt: &str, max_tokens: u32| GenerationRequest {
            temperature: 0.8,
            seed: Some(7),
            ..generation_request(prompt, max_tokens)
        };

        // (1) Cold session: full prefill + 64 decode tokens past the
        // indexer-selection threshold.
        let started = std::time::Instant::now();
        let first = backend
            .generate(decode_request(&preamble, 64))
            .await
            .unwrap();
        let cold_s = started.elapsed().as_secs_f64();
        let reused_baseline = health_counter(&backend, "reused_tokens");
        eprintln!(
            "[cold ] {cold_s:.1}s for {} prompt + {} completion tokens",
            first.prompt_tokens, first.completion_tokens
        );
        assert!(
            first.prompt_tokens > 2600,
            "cold prompt must cross the 2048-token selection threshold, got {}",
            first.prompt_tokens
        );
        assert!(
            first.completion_tokens >= 32,
            "cold decode must actually exercise the >2048 decode path ({} tokens)",
            first.completion_tokens
        );

        // (2) Second identical conversation: must restore the deepest cached
        // block (~prompt/256 boundaries stored by the cold run) and complete
        // in seconds-class time.
        let started = std::time::Instant::now();
        let second = backend
            .generate(decode_request(&preamble, 64))
            .await
            .unwrap();
        let warm_s = started.elapsed().as_secs_f64();
        let reused = health_counter(&backend, "reused_tokens") - reused_baseline;
        eprintln!(
            "[warm ] {warm_s:.1}s, reused {reused} tokens, {} completion tokens",
            second.completion_tokens
        );
        assert!(
            reused >= (first.prompt_tokens / 2),
            "second conversation must restore the deep prefix (reused {reused} of {} prompt tokens)",
            first.prompt_tokens
        );
        assert!(
            warm_s < cold_s / 3.0,
            "warm conversation must be far faster than cold ({warm_s:.1}s vs {cold_s:.1}s)"
        );

        // (3) Short-prompt probe (below the selection threshold): healthy
        // wall time, the production direct-curl case (~1.3k tokens).
        let short = crate::dsv4_gpu::tests::long_prompt(283);
        let started = std::time::Instant::now();
        let probe = backend.generate(decode_request(&short, 16)).await.unwrap();
        let short_s = started.elapsed().as_secs_f64();
        eprintln!(
            "[short] {short_s:.1}s for {} prompt + {} completion tokens (~{:.1} tok/s equivalent)",
            probe.prompt_tokens,
            probe.completion_tokens,
            (probe.prompt_tokens + probe.completion_tokens) as f64 / short_s,
        );
        assert!(
            probe.prompt_tokens < 2000,
            "short probe must stay below the selection threshold, got {}",
            probe.prompt_tokens
        );
        // Cold wall on GPU 0 is dominated by GEMM-prefill expert uploads at
        // the mandated 40 GiB validation pool (29% of slices resident; the
        // production GPU runs the 72 GiB default and measured ~49 tok/s
        // prefill ⇒ ~110s-class cold). The bound here catches the >2048-token
        // decode regression class (598s+ from per-token selection stacking)
        // without asserting the pool-budget-bound prefill; the sharp decode
        // gates are the warm and short bounds above/below.
        assert!(
            cold_s < 480.0,
            "cold ~4.6k session regressed to {cold_s:.1}s (>2048-context decode must not stack per-token selection costs)"
        );
        assert!(
            warm_s < 60.0,
            "warm reuse session must be seconds-class, got {warm_s:.1}s"
        );
    }

    /// Test-only [`Drafter`] replaying a preloaded continuation (a known
    /// prompt's true greedy continuation), optionally corrupting chosen
    /// continuation indices to force rejections — a corrupted draft is
    /// `(token + 1) % vocab`, guaranteed unequal to the true argmax. Every
    /// `observe_accepted` call is logged through the shared handle so tests
    /// can assert the feedback contract from outside the worker thread.
    struct OracleDrafter {
        prompt_len: usize,
        continuation: Vec<u32>,
        corrupt: HashSet<usize>,
        vocab: u32,
        observed: Arc<Mutex<Vec<(usize, usize, u32)>>>,
        /// Capture request for the loop (default: none). When non-empty, every
        /// propose logs the taps' coverage into `tap_log` and any coverage
        /// violation into `tap_violations` — asserting inside the worker
        /// thread would only kill the worker, so tests assert on the logs.
        tap_config: DsV4TapConfig,
        /// Per-propose (taps.base(), taps.positions(), ctx.tokens.len()).
        tap_log: Arc<Mutex<Vec<(usize, usize, usize)>>>,
        tap_violations: Arc<Mutex<Vec<String>>>,
    }

    impl OracleDrafter {
        fn new(
            prompt_len: usize,
            continuation: &[u32],
            corrupt: &[usize],
            vocab: u32,
            observed: Arc<Mutex<Vec<(usize, usize, u32)>>>,
        ) -> Self {
            Self {
                prompt_len,
                continuation: continuation.to_vec(),
                corrupt: corrupt.iter().copied().collect(),
                vocab,
                observed,
                tap_config: DsV4TapConfig::default(),
                tap_log: Arc::new(Mutex::new(Vec::new())),
                tap_violations: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl Drafter for OracleDrafter {
        fn tap_config(&self) -> DsV4TapConfig {
            self.tap_config.clone()
        }

        fn propose(&mut self, ctx: &DraftContext<'_>) -> Vec<u32> {
            if !self.tap_config.is_empty() {
                let mut violations = self.tap_violations.lock().unwrap();
                match ctx.taps {
                    None => violations.push("tap-requesting drafter got no taps".to_string()),
                    Some(taps) => {
                        let (base, end) = (taps.base(), taps.positions());
                        self.tap_log
                            .lock()
                            .unwrap()
                            .push((base, end, ctx.tokens.len()));
                        if end + 1 != ctx.tokens.len() {
                            violations.push(format!(
                                "taps end {end} does not cover tokens {} - 1",
                                ctx.tokens.len()
                            ));
                        }
                        // Coverage: rows exist exactly for base..end and
                        // NEVER below the base (a misaligned row would poison
                        // a real drafter silently).
                        if base > 0 && taps.pre_hc_head(base - 1).is_some() {
                            violations.push(format!("row exists below base {base}"));
                        }
                        for position in base..end {
                            if self.tap_config.pre_hc_head && taps.pre_hc_head(position).is_none() {
                                violations.push(format!("missing pre-hc-head row {position}"));
                            }
                            for &layer in &self.tap_config.aux_layers {
                                if taps.aux_flat(layer, position).is_none() {
                                    violations.push(format!("missing aux {layer} row {position}"));
                                }
                            }
                        }
                    }
                }
            }
            // ctx.tokens = prompt + every emitted token (pending included), so
            // the next tokens to draft start at continuation[generated].
            let generated = ctx.tokens.len().saturating_sub(self.prompt_len);
            (0..ctx.k)
                .map_while(|offset| {
                    let idx = generated + offset;
                    let token = *self.continuation.get(idx)?;
                    Some(if self.corrupt.contains(&idx) {
                        (token + 1) % self.vocab
                    } else {
                        token
                    })
                })
                .collect()
        }

        fn observe_accepted(&mut self, proposed: usize, accepted: usize, emitted: u32) {
            self.observed
                .lock()
                .unwrap()
                .push((proposed, accepted, emitted));
        }
    }

    /// Drive one request to completion; returns (delta token ids, text,
    /// completion count). Deltas carry every emitted token EXCEPT
    /// special-skipped ids (the spec fixture's `a b c` are all visible; only
    /// its never-emitted eos `d` would skip), and their concatenated text
    /// must reassemble the final text.
    async fn collect_generation(
        backend: &DeepSeekV4Backend,
        request: GenerationRequest,
    ) -> (Vec<u32>, String, u64) {
        let mut stream = backend.stream_generate(request).await.unwrap();
        let mut ids = Vec::new();
        let mut collected = String::new();
        let mut finished = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                GenerationEvent::TokenDelta { token_id, text } => {
                    ids.push(token_id);
                    collected.push_str(&text);
                }
                GenerationEvent::Finished { output } => finished = Some(output),
            }
        }
        let finished = finished.expect("stream must end with Finished");
        assert_eq!(collected, finished.text);
        (ids, finished.text, finished.completion_tokens)
    }

    /// The greedy continuation the spec-fixture model produces after
    /// `prompt`, from the CPU oracle (bit-exact with the GPU engine's greedy
    /// tokens per the Stage-1 parity gate) — the OracleDrafter's replay
    /// source and the token-exactness reference.
    fn reference_continuation(path: &std::path::Path, prompt: &str, max_tokens: usize) -> Vec<u32> {
        let cpu = DeepSeekV4CpuReference::load(path).unwrap();
        let tokens = cpu.tokenizer().encode(prompt).unwrap();
        let out = cpu
            .run_tokens(
                &tokens,
                QwenCpuRunOptions {
                    max_tokens,
                    ..QwenCpuRunOptions::default()
                },
            )
            .unwrap();
        out.generated_tokens
    }

    /// Oracle backend over the spec fixture at `path` with the given
    /// corruption set (continuation indices whose draft is deliberately
    /// wrong).
    fn oracle_backend(
        path: &std::path::Path,
        prompt_len: usize,
        continuation: &[u32],
        corrupt: &[usize],
        observed: Arc<Mutex<Vec<(usize, usize, u32)>>>,
    ) -> Arc<DeepSeekV4Backend> {
        let drafter = OracleDrafter::new(prompt_len, continuation, corrupt, 4, observed);
        oracle_backend_with(path, drafter)
    }

    fn oracle_backend_with(
        path: &std::path::Path,
        drafter: OracleDrafter,
    ) -> Arc<DeepSeekV4Backend> {
        Arc::new(
            DeepSeekV4Backend::load_with_drafter(
                path,
                Some("dsv4-fixture".to_string()),
                4,
                BIG_PREFIX_BUDGET,
                Box::new(move |_| Some(Box::new(drafter))),
            )
            .unwrap(),
        )
    }

    /// The Stage-A speculative acceptance gate (oracle drafting): a perfect
    /// draft accepts everything, an adversarial draft rejects everything, and
    /// mixed corruption lands in between — with the emitted token stream
    /// identical to sequential greedy decode in EVERY case (losslessness is
    /// the whole point). The spec fixture streams every emitted token id and
    /// greedy runs to the budget, so acceptance has real depth.
    #[tokio::test]
    async fn dsv4_backend_oracle_drafter_outputs_match_sequential_greedy() {
        let prompt = "abcabcab";
        let max_tokens = 8u32;

        // Sequential baseline (no drafter) + CPU-oracle continuation over the
        // same model file; the delta stream must carry exactly those tokens.
        let baseline_path = tempfile_path("oracle-baseline");
        write_deepseek4_spec_gguf(&baseline_path);
        let continuation = reference_continuation(&baseline_path, prompt, max_tokens as usize);
        assert_eq!(
            continuation.len(),
            max_tokens as usize,
            "spec fixture greedy must run to the budget: {continuation:?}"
        );
        let baseline = Arc::new(
            DeepSeekV4Backend::load_with_prefix_config(
                &baseline_path,
                Some("dsv4-fixture".to_string()),
                4,
                BIG_PREFIX_BUDGET,
            )
            .unwrap(),
        );
        let (base_ids, base_text, base_completion) =
            collect_generation(&baseline, generation_request(prompt, max_tokens)).await;
        assert_eq!(
            base_ids, continuation,
            "sequential backend must reproduce the CPU-oracle continuation"
        );

        let adversarial: Vec<usize> = (0..continuation.len()).collect();
        for (name, corrupt) in [
            ("oracle-perfect", Vec::new()),
            ("oracle-adversarial", adversarial),
            ("oracle-mixed", vec![1usize]),
        ] {
            let path = tempfile_path(name);
            write_deepseek4_spec_gguf(&path);
            let observed = Arc::new(Mutex::new(Vec::new()));
            let backend = oracle_backend(
                &path,
                prompt.len(),
                &continuation,
                &corrupt,
                observed.clone(),
            );
            let (ids, text, completion) =
                collect_generation(&backend, generation_request(prompt, max_tokens)).await;
            assert_eq!(ids, continuation, "{name}: tokens must match sequential");
            assert_eq!(text, base_text, "{name}: text must match sequential");
            assert_eq!(completion, base_completion, "{name}");

            let verify_steps = health_counter(&backend, "spec_verify_steps");
            let proposed = health_counter(&backend, "spec_proposed");
            let accepted = health_counter(&backend, "spec_accepted");
            assert!(verify_steps >= 1, "{name}: no verify steps ran");
            assert!(proposed >= 1, "{name}: drafter never proposed");
            let observations = observed.lock().unwrap();
            assert_eq!(
                observations.len() as u64,
                verify_steps,
                "{name}: observe_accepted must fire once per verify"
            );
            assert_eq!(
                observations.iter().map(|(_, a, _)| *a as u64).sum::<u64>(),
                accepted,
                "{name}: observed accepted counts must match the health counter"
            );
            drop(observations);
            match name {
                "oracle-perfect" => {
                    assert_eq!(accepted, proposed, "perfect drafts must all be accepted");
                }
                "oracle-adversarial" => {
                    assert_eq!(accepted, 0, "adversarial drafts must all be rejected");
                }
                _ => {
                    assert!(
                        accepted > 0 && accepted < proposed,
                        "mixed corruption must accept some and reject some \
                         (accepted {accepted} of {proposed})"
                    );
                }
            }

            // Prefix-cache interaction: an identical rerun restores cached
            // blocks (snapshots taken only at accepted boundaries, including
            // interior boundaries crossed mid-verify) and still reproduces
            // the sequential stream exactly.
            let (rerun_ids, rerun_text, _) =
                collect_generation(&backend, generation_request(prompt, max_tokens)).await;
            assert_eq!(rerun_ids, continuation, "{name}: rerun after restore");
            assert_eq!(rerun_text, base_text, "{name}");
            assert!(
                health_counter(&backend, "reused_tokens") > 0,
                "{name}: rerun must restore a cached block"
            );
        }
    }

    /// Speculative decode across turns: turn 2 extends turn 1 (the chat
    /// pattern), restoring boundary snapshots the speculative loop wrote —
    /// including interior boundaries snapshotted via clone + exact truncate
    /// mid-verify — and its output still matches a cold sequential backend.
    /// The oracle's replay no longer lines up on turn 2, so its drafts are
    /// effectively adversarial there.
    #[tokio::test]
    async fn dsv4_backend_oracle_drafter_turn_two_matches_cold_sequential() {
        let prompt = "abcabcab";
        let max_tokens = 8u32;
        let baseline_path = tempfile_path("oracle-turn2-base");
        write_deepseek4_spec_gguf(&baseline_path);
        let continuation = reference_continuation(&baseline_path, prompt, max_tokens as usize);
        let baseline = Arc::new(
            DeepSeekV4Backend::load_with_prefix_config(
                &baseline_path,
                Some("dsv4-fixture".to_string()),
                4,
                BIG_PREFIX_BUDGET,
            )
            .unwrap(),
        );
        let (_, turn1_text, _) =
            collect_generation(&baseline, generation_request(prompt, max_tokens)).await;

        let path = tempfile_path("oracle-turn2");
        write_deepseek4_spec_gguf(&path);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let backend = oracle_backend(&path, prompt.len(), &continuation, &[], observed);
        let (_, spec_turn1_text, _) =
            collect_generation(&backend, generation_request(prompt, max_tokens)).await;
        assert_eq!(spec_turn1_text, turn1_text);

        // Turn 2: previous prompt + reply + new user text.
        let turn2 = format!("{prompt}{turn1_text}ba");
        let reused_before = health_counter(&backend, "reused_tokens");
        let (spec_ids, spec_text, _) =
            collect_generation(&backend, generation_request(&turn2, max_tokens)).await;
        assert!(
            health_counter(&backend, "reused_tokens") > reused_before,
            "turn 2 must restore turn 1's cached blocks"
        );
        let (cold_ids, cold_text, _) =
            collect_generation(&baseline, generation_request(&turn2, max_tokens)).await;
        assert_eq!(spec_ids, cold_ids, "turn 2 must match cold sequential");
        assert_eq!(spec_text, cold_text);
    }

    /// Early stop INSIDE the accepted-draft emission: a stop sequence (and,
    /// on the standard fixture, eos below) triggers mid-verify exactly where
    /// sequential decode stops — the emitted stream is cut at the same token.
    #[tokio::test]
    async fn dsv4_backend_oracle_drafter_stops_mid_accepted_drafts() {
        // Stop sequence on the spec fixture (visible text, token-exact).
        let prompt = "abcabcab";
        let max_tokens = 8u32;
        let path = tempfile_path("oracle-stop-base");
        write_deepseek4_spec_gguf(&path);
        let continuation = reference_continuation(&path, prompt, max_tokens as usize);
        let baseline = Arc::new(
            DeepSeekV4Backend::load_with_prefix_config(
                &path,
                Some("dsv4-fixture".to_string()),
                4,
                BIG_PREFIX_BUDGET,
            )
            .unwrap(),
        );
        let stop_request = |prompt: &str| GenerationRequest {
            stop_sequences: vec!["aca".to_string()],
            ..generation_request(prompt, max_tokens)
        };
        let (base_ids, base_text, base_completion) =
            collect_generation(&baseline, stop_request(prompt)).await;
        assert!(
            (base_completion as u32) < max_tokens,
            "stop sequence must cut the baseline short: {base_ids:?}"
        );

        let oracle_path = tempfile_path("oracle-stop");
        write_deepseek4_spec_gguf(&oracle_path);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let backend = oracle_backend(&oracle_path, prompt.len(), &continuation, &[], observed);
        let (ids, text, completion) = collect_generation(&backend, stop_request(prompt)).await;
        assert_eq!(ids, base_ids, "stop-sequence cut must be token-identical");
        assert_eq!(text, base_text);
        assert_eq!(completion, base_completion);

        // eos inside an accepted draft, on the standard fixture (greedy
        // reaches eos "c" within two tokens; ids are delta-invisible there,
        // so the gate is text/completion equality + the observed log).
        let eos_prompt = "abcab";
        let eos_base_path = tempfile_path("oracle-eos-base");
        write_deepseek4_gguf(&eos_base_path);
        let eos_continuation = reference_continuation(&eos_base_path, eos_prompt, 6);
        assert!(
            eos_continuation.len() < 6,
            "standard fixture must stop at eos before the budget"
        );
        let eos_baseline = Arc::new(
            DeepSeekV4Backend::load_with_prefix_config(
                &eos_base_path,
                Some("dsv4-fixture".to_string()),
                4,
                BIG_PREFIX_BUDGET,
            )
            .unwrap(),
        );
        let (_, eos_base_text, eos_base_completion) =
            collect_generation(&eos_baseline, generation_request(eos_prompt, 6)).await;

        let eos_path = tempfile_path("oracle-eos");
        write_deepseek4_gguf(&eos_path);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let drafter = OracleDrafter::new(
            eos_prompt.len(),
            &eos_continuation,
            &[],
            3,
            observed.clone(),
        );
        let backend = oracle_backend_with(&eos_path, drafter);
        let (_, text, completion) =
            collect_generation(&backend, generation_request(eos_prompt, 6)).await;
        assert_eq!(text, eos_base_text);
        assert_eq!(completion, eos_base_completion);
        let observations = observed.lock().unwrap();
        assert_eq!(observations.len(), 1, "one verify covers the eos");
        let (proposed, accepted, _) = observations[0];
        assert_eq!(
            (proposed, accepted),
            (eos_continuation.len() - 1, eos_continuation.len() - 1),
            "the accepted drafts must include the eos token"
        );
    }

    /// The spec-decoding redeployment gate: a tap-requesting drafter no
    /// longer disables prefix-cache restores. Request 2 with the same prefix
    /// RESTORES (reused_tokens > 0) with taps attached at the restore point
    /// (base == restored length; complete coverage above it, nothing below
    /// it), the drafter keeps proposing, and every output stays byte-identical
    /// to the sequential spec-off run — the 2-second fresh-session restore
    /// survives with speculation on.
    #[tokio::test]
    async fn dsv4_backend_taps_coexist_with_prefix_restore() {
        let prompt = "abcabcab";
        let max_tokens = 8u32;
        let baseline_path = tempfile_path("taps-restore-base");
        write_deepseek4_spec_gguf(&baseline_path);
        let continuation = reference_continuation(&baseline_path, prompt, max_tokens as usize);
        let baseline = Arc::new(
            DeepSeekV4Backend::load_with_prefix_config(
                &baseline_path,
                Some("dsv4-fixture".to_string()),
                4,
                BIG_PREFIX_BUDGET,
            )
            .unwrap(),
        );
        let (base_ids, base_text, _) =
            collect_generation(&baseline, generation_request(prompt, max_tokens)).await;
        assert_eq!(base_ids, continuation);

        let path = tempfile_path("taps-restore");
        write_deepseek4_spec_gguf(&path);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut drafter = OracleDrafter::new(prompt.len(), &continuation, &[], 4, observed);
        drafter.tap_config = DsV4TapConfig {
            pre_hc_head: true,
            aux_layers: vec![0, 2],
        };
        let tap_log = drafter.tap_log.clone();
        let violations = drafter.tap_violations.clone();
        let backend = oracle_backend_with(&path, drafter);

        // Request 1 (cold): taps attach at 0, spec active, output identical.
        let (ids1, text1, _) =
            collect_generation(&backend, generation_request(prompt, max_tokens)).await;
        assert_eq!(ids1, continuation);
        assert_eq!(text1, base_text);
        assert_eq!(health_counter(&backend, "reused_tokens"), 0);
        let proposed_cold = health_counter(&backend, "spec_proposed");
        assert!(proposed_cold > 0, "the drafter never proposed cold");
        assert!(
            tap_log.lock().unwrap().iter().all(|&(base, ..)| base == 0),
            "cold request taps must attach at 0"
        );

        // Request 2 (same prefix): restores AND speculates, identically.
        tap_log.lock().unwrap().clear();
        let (ids2, text2, _) =
            collect_generation(&backend, generation_request(prompt, max_tokens)).await;
        assert_eq!(ids2, continuation, "restored spec run must stay identical");
        assert_eq!(text2, base_text);
        let reused = health_counter(&backend, "reused_tokens");
        assert!(reused > 0, "request 2 must restore a cached block");
        assert!(
            health_counter(&backend, "spec_proposed") > proposed_cold,
            "the drafter must keep proposing after a restore"
        );
        {
            let log = tap_log.lock().unwrap();
            assert!(!log.is_empty(), "no tapped proposals after the restore");
            assert!(
                log.iter().all(|&(base, ..)| base as u64 == reused),
                "taps must attach at the restore point: {log:?}"
            );
        }

        // Turn 2 extends turn 1 (the chat pattern): restores deeper via the
        // decode-time snapshots and still matches a cold sequential run.
        let turn2 = format!("{prompt}{base_text}ba");
        let reused_before = health_counter(&backend, "reused_tokens");
        let (spec_ids, spec_text, _) =
            collect_generation(&backend, generation_request(&turn2, max_tokens)).await;
        assert!(
            health_counter(&backend, "reused_tokens") > reused_before,
            "turn 2 must restore turn 1's blocks"
        );
        let (cold_ids, cold_text, _) =
            collect_generation(&baseline, generation_request(&turn2, max_tokens)).await;
        assert_eq!(spec_ids, cold_ids, "turn 2 must match cold sequential");
        assert_eq!(spec_text, cold_text);

        let violations = violations.lock().unwrap();
        assert!(
            violations.is_empty(),
            "tap coverage violated: {violations:?}"
        );
    }

    /// Stage A is greedy-only: a sampled request on a drafter-configured
    /// backend falls back to the sequential loop (no verify steps) and still
    /// serves normally.
    #[tokio::test]
    async fn dsv4_backend_drafter_skipped_for_sampled_requests() {
        let path = tempfile_path("oracle-sampled");
        write_deepseek4_spec_gguf(&path);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let backend = oracle_backend(&path, 5, &[0, 1, 2], &[], observed);
        let request = GenerationRequest {
            temperature: 0.8,
            seed: Some(11),
            ..generation_request("abcab", 4)
        };
        let output = backend.generate(request).await.unwrap();
        assert!(output.completion_tokens >= 1);
        assert_eq!(health_counter(&backend, "spec_verify_steps"), 0);
        assert_eq!(health_counter(&backend, "spec_proposed"), 0);
    }

    /// The Stage-2 smoke test: the axum server (hi-local-core) over the V4
    /// backend serves /health and /v1/chat/completions in both modes.
    #[tokio::test]
    async fn dsv4_backend_serves_chat_completions_end_to_end() {
        let backend = fixture_backend("serve");
        let app = hi_local_core::server::app(backend as SharedBackend);

        // /health carries the dsv4 marker object next to the standard fields.
        let response = app
            .clone()
            .oneshot(
                Request::get("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let health = body_json(response).await;
        assert_eq!(health["status"], "ok");
        assert_eq!(health["backend"], "cuda");
        assert_eq!(health["ready"], true);
        assert_eq!(health["family"], "deepseek");
        assert_eq!(health["execution"]["status"], "gpu");
        assert_eq!(health["dsv4"]["status"], "enabled");
        assert_eq!(health["dsv4"]["engine"], "cuda-dsv4");
        assert_eq!(health["scheduler"]["status"], "disabled");

        // Non-streaming chat completion.
        let response = app
            .clone()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "dsv4-fixture",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 6,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert!(body["choices"][0]["message"]["content"].is_string());
        assert!(body["usage"]["completion_tokens"].as_u64().unwrap() >= 1);

        // Streaming chat completion: SSE chunks then the [DONE] sentinel.
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "dsv4-fixture",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 6,
                            "stream": true,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            body.contains("chat.completion.chunk"),
            "missing SSE chunks:\n{body}"
        );
        assert!(
            body.trim_end().ends_with("data: [DONE]"),
            "missing [DONE] sentinel:\n{body}"
        );
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
}
