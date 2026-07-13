//! Bounded pinned-host RAM tier + disk I/O policy for streamed MoE experts.
//!
//! On a limited-RAM box (24-64 GB) serving a 169-466 GB MoE GGUF, the implicit
//! "RAM tier" is the unbounded OS page cache, and it thrashes. This module
//! gives the expert pool an explicit, budgeted middle tier and colibri-style
//! disk discipline:
//!
//! * [`RamTier`]: a fixed pinned-host arena (LRU, recency tickets, sticky
//!   pre-warmed entries) fronting disk reads. Hits upload with a true async
//!   DMA (already page-locked) and cost zero disk I/O.
//! * [`plan_tier_budget`]: explicit `HI_CUDA_EXPERT_RAM_GB` budget, else a
//!   conservative auto budget derived from `MemAvailable` minus itemized
//!   slack (page-cache reserve, working-set slab, staging, misc). The plan is
//!   printed once at startup so the projected peak is known before the first
//!   token (the "OOM-killer never fires" rule).
//! * [`ExpertUsage`]: per-(layer, expert) selection counters persisted
//!   atomically to `<model_dir>/.hi_expert_usage`; at construction the
//!   hottest experts are pre-warmed into the tier as sticky entries
//!   (profile-ranked placement beats heat-blind placement ~3x at equal
//!   capacity in colibri's measurements).
//! * [`ExpertFetcher`]: the disk read path — buffered mmap copies with
//!   `MADV_RANDOM` extents + per-extent `WILLNEED`, or an optional
//!   `O_DIRECT` twin-fd pread path (`HI_CUDA_EXPERT_ODIRECT=1`) that bypasses
//!   the page cache entirely.
//! * [`WillNeedThread`]: batched `WILLNEED` readahead for upcoming expert
//!   extents issued from a dedicated thread (an inline fadvise on a saturated
//!   queue costs ~0.5 ms/call; threaded it overlaps the reads).
//! * [`UploadEngine`]: the dsv4 CopyEngine pattern rebuilt on `runtime.rs`
//!   primitives — a non-blocking copy stream, fork/done events against the
//!   engine stream, and double-buffered pinned staging for bytes that do not
//!   live in the tier.
//!
//! Placement/caching never changes router semantics or precision: cache
//! pressure affects speed, never output.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{
    GgufDirectReader, GgufFile, GgufMemoryAdvice, qwen_moe_packed_expert_weight_names,
    qwen_moe_router_weight_names,
};
use serde::{Deserialize, Serialize};

use super::{ExpertKey, ExpertSource};
use crate::runtime::{DeviceBuffer, Event, PinnedBuffer, Stream};

pub(crate) const GIB: u64 = 1 << 30;

/// Reserve left to the page cache so buffered reads keep working. colibri
/// measured buffered preads collapsing 800 -> 180 MB/s when the cache is
/// starved and reserves 2.5 GB; we round up.
const PAGE_CACHE_RESERVE: u64 = 3 * GIB;
/// Decode working set outside this pool: activations, host mirrors of routed
/// ids/pointer tables, tokenizer, session buffers.
const WORKING_SET_SLAB: u64 = 3 * GIB / 2;
/// Allocator slop, thread stacks, mmap metadata, incidental process growth.
const MISC_RESERVE: u64 = GIB / 2;
/// Fraction of post-slack MemAvailable the auto budget takes.
const AUTO_FRACTION: f64 = 0.5;

/// Pinned-staging generations (double buffering, dsv4 pattern) and slot-sized
/// regions per generation.
pub(crate) const STAGING_GENERATIONS: usize = 2;
pub(crate) const STAGING_SLOTS: usize = 4;

const USAGE_FILE_NAME: &str = ".hi_expert_usage";
const USAGE_FILE_VERSION: u32 = 1;

/// `HI_CUDA_EXPERT_IOURING_QD` default.
#[cfg(target_os = "linux")]
pub(crate) const DEFAULT_IOURING_QD: u32 = crate::expert_uring::DEFAULT_QD;
#[cfg(not(target_os = "linux"))]
pub(crate) const DEFAULT_IOURING_QD: u32 = 256;

/// O_DIRECT block alignment required of ring-mode slot bases. `cudaHostAlloc`
/// suballocates small pinned buffers without page alignment, so the pool
/// over-allocates by one block and shifts the slot region to the next
/// boundary (`RamTier::new_with_stride`'s `base_offset`).
pub(crate) const SLOT_DMA_ALIGN: usize = 4096;
#[cfg(target_os = "linux")]
const _: () = assert!(SLOT_DMA_ALIGN == crate::expert_uring::URING_BLOCK);

// ---------------------------------------------------------------------------
// Env knobs
// ---------------------------------------------------------------------------

/// All expert-tier env knobs, read once at pool construction (no statics, so
/// tests can vary them per instance).
#[derive(Debug, Clone)]
pub(crate) struct TierEnvConfig {
    /// `HI_CUDA_EXPERT_RAM_GB`: explicit pinned-tier budget in GiB
    /// (fractional allowed; `0` disables the tier). Unset = auto.
    pub explicit_ram_gb: Option<f64>,
    /// `HI_CUDA_EXPERT_PREWARM_FRAC` (default 0.5): max fraction of tier
    /// slots pre-warmed sticky from usage history; 0 disables pre-warming.
    pub prewarm_frac: f64,
    /// `HI_CUDA_EXPERT_ODIRECT=1`: O_DIRECT twin-fd expert reads.
    pub odirect: bool,
    /// `HI_CUDA_EXPERT_IOURING=1`: batch-submitted io_uring O_DIRECT reads
    /// (Linux; opt-in). Probed at construction; on any failure the pool falls
    /// back to the O_DIRECT thread path, then mmap — never a load failure.
    pub iouring: bool,
    /// `HI_CUDA_EXPERT_IOURING_QD` (default 256): io_uring submission queue
    /// depth, clamped to a power of two in [8, 4096].
    pub iouring_qd: u32,
    /// `HI_CUDA_EXPERT_WILLNEED=0` disables WILLNEED readahead (default on).
    pub willneed: bool,
    /// `HI_CUDA_EXPERT_MADVISE=0` disables MADV_RANDOM on expert extents
    /// (default on).
    pub madvise_random: bool,
    /// `HI_CUDA_EXPERT_SYNC_UPLOAD=1`: keep the original synchronous
    /// mmap->Vec->cudaMemcpy upload path (bisection escape hatch). The tier
    /// then stores pageable memory instead of pinned.
    pub sync_upload: bool,
    /// `HI_CUDA_EXPERT_USAGE_SAVE_SECS` (default 60): usage persistence
    /// cadence; 0 disables loading and saving `.hi_expert_usage`.
    pub usage_save_secs: u64,
}

impl TierEnvConfig {
    pub(crate) fn from_env() -> Self {
        let parse_f64 = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok());
        let flag_on = |name: &str| std::env::var(name).is_ok_and(|v| v != "0");
        let flag_default_on = |name: &str| std::env::var(name).map_or(true, |v| v != "0");
        Self {
            explicit_ram_gb: parse_f64("HI_CUDA_EXPERT_RAM_GB"),
            prewarm_frac: parse_f64("HI_CUDA_EXPERT_PREWARM_FRAC")
                .unwrap_or(0.5)
                .clamp(0.0, 1.0),
            odirect: flag_on("HI_CUDA_EXPERT_ODIRECT"),
            iouring: flag_on("HI_CUDA_EXPERT_IOURING"),
            iouring_qd: std::env::var("HI_CUDA_EXPERT_IOURING_QD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_IOURING_QD),
            willneed: flag_default_on("HI_CUDA_EXPERT_WILLNEED"),
            madvise_random: flag_default_on("HI_CUDA_EXPERT_MADVISE"),
            sync_upload: flag_on("HI_CUDA_EXPERT_SYNC_UPLOAD"),
            usage_save_secs: std::env::var("HI_CUDA_EXPERT_USAGE_SAVE_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
        }
    }
}

/// `MemAvailable` from /proc/meminfo in bytes (Linux; None elsewhere or on
/// parse failure, which disables the auto budget).
pub(crate) fn mem_available_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Budget plan
// ---------------------------------------------------------------------------

/// The itemized pinned-tier budget, computed once at pool construction and
/// printed so the projected peak is visible before the first decode.
#[derive(Debug, Clone)]
pub(crate) struct TierBudgetPlan {
    pub mem_available: Option<u64>,
    pub page_cache_reserve: u64,
    pub working_set_slab: u64,
    pub staging_bytes: u64,
    pub misc_reserve: u64,
    /// Final pinned-arena budget in bytes (0 = tier disabled).
    pub budget_bytes: u64,
    pub slots: usize,
    pub slot_bytes: usize,
    /// Where the budget came from, for the startup line.
    pub source: &'static str,
}

impl TierBudgetPlan {
    pub(crate) fn enabled(&self) -> bool {
        self.slots >= 2
    }

    fn gib(bytes: u64) -> f64 {
        bytes as f64 / GIB as f64
    }

    /// One line with every accounting term; `projected peak` is what this
    /// machinery itself will pin/allocate (tier arena + staging), and the
    /// slack terms are what it deliberately leaves for everyone else.
    pub(crate) fn describe(&self) -> String {
        let mem = match self.mem_available {
            Some(bytes) => format!("{:.1}GiB", Self::gib(bytes)),
            None => "unknown".to_string(),
        };
        let slack = self.page_cache_reserve
            + self.working_set_slab
            + self.staging_bytes
            + self.misc_reserve;
        format!(
            "MemAvailable={mem} slack[page-cache={:.1}GiB working-set={:.1}GiB staging={:.2}GiB misc={:.1}GiB]={:.1}GiB budget={:.1}GiB ({}, {} slots x {:.2}MiB) projected-peak(tier+staging)={:.1}GiB",
            Self::gib(self.page_cache_reserve),
            Self::gib(self.working_set_slab),
            Self::gib(self.staging_bytes),
            Self::gib(self.misc_reserve),
            Self::gib(slack),
            Self::gib(self.budget_bytes),
            self.source,
            self.slots,
            self.slot_bytes as f64 / (1 << 20) as f64,
            Self::gib(self.budget_bytes + self.staging_bytes),
        )
    }
}

/// Compute the pinned-tier budget. Pure so the accounting is unit-testable:
/// pass `MemAvailable`, the arena slot size (the stride, in ring mode), the
/// arena bytes it would take to cache every streamable expert slice — never
/// pin more than that — and the staging overhead.
pub(crate) fn plan_tier_budget(
    explicit_ram_gb: Option<f64>,
    mem_available: Option<u64>,
    slot_bytes: usize,
    max_useful_bytes: u64,
    staging_bytes: u64,
) -> TierBudgetPlan {
    let mut plan = TierBudgetPlan {
        mem_available,
        page_cache_reserve: PAGE_CACHE_RESERVE,
        working_set_slab: WORKING_SET_SLAB,
        staging_bytes,
        misc_reserve: MISC_RESERVE,
        budget_bytes: 0,
        slots: 0,
        slot_bytes,
        source: "disabled",
    };
    if slot_bytes == 0 {
        return plan;
    }
    let requested = match explicit_ram_gb {
        Some(gb) if gb <= 0.0 => {
            plan.source = "disabled (HI_CUDA_EXPERT_RAM_GB=0)";
            return plan;
        }
        Some(gb) => {
            plan.source = "HI_CUDA_EXPERT_RAM_GB";
            (gb * GIB as f64) as u64
        }
        None => {
            let Some(available) = mem_available else {
                plan.source = "disabled (MemAvailable unknown)";
                return plan;
            };
            plan.source = "auto";
            let slack =
                plan.page_cache_reserve + plan.working_set_slab + staging_bytes + plan.misc_reserve;
            (available.saturating_sub(slack) as f64 * AUTO_FRACTION) as u64
        }
    };
    // Never pin more than it takes to cache every streamable slice.
    let requested = requested.min(max_useful_bytes);
    let slots = usize::try_from(requested / slot_bytes as u64).unwrap_or(0);
    if slots < 2 {
        plan.source = "disabled (budget below 2 slots)";
        return plan;
    }
    plan.slots = slots;
    plan.budget_bytes = slots as u64 * slot_bytes as u64;
    plan
}

// ---------------------------------------------------------------------------
// Tier storage + LRU
// ---------------------------------------------------------------------------

/// Backing storage of the RAM tier. Pinned host memory when the async upload
/// engine is available (hits then DMA to device without a staging copy);
/// plain heap when it is not (sync-upload mode, pinned-alloc failure, tests).
/// Either way the tier bounds host RAM to its budget.
pub(crate) enum TierArena {
    Pinned(PinnedBuffer),
    Heap(Vec<u8>),
}

impl TierArena {
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        match self {
            Self::Pinned(pinned) => pinned.copy_in(offset, bytes),
            Self::Heap(heap) => {
                let end = offset
                    .checked_add(bytes.len())
                    .ok_or_else(|| anyhow!("tier arena write range overflows usize"))?;
                let Some(slot) = heap.get_mut(offset..end) else {
                    bail!(
                        "tier arena write of {} bytes at {offset} exceeds the {}-byte arena",
                        bytes.len(),
                        heap.len()
                    );
                };
                slot.copy_from_slice(bytes);
                Ok(())
            }
        }
    }

    pub(crate) fn pinned(&self) -> Option<&PinnedBuffer> {
        match self {
            Self::Pinned(pinned) => Some(pinned),
            Self::Heap(_) => None,
        }
    }

    pub(crate) fn heap_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        match self {
            Self::Pinned(_) => None,
            Self::Heap(heap) => heap.get(offset..offset + len),
        }
    }

    pub(crate) fn is_pinned(&self) -> bool {
        matches!(self, Self::Pinned(_))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct RamTierStats {
    /// Device-pool misses served from the tier (no disk I/O).
    pub hits: u64,
    /// Device-pool misses that went to disk.
    pub misses: u64,
    pub evictions: u64,
    pub inserts: u64,
    /// Bytes served from the tier to the device (hit uploads).
    pub bytes_served: u64,
    /// Sticky entries pre-warmed from usage history at construction.
    pub prewarmed: u64,
}

struct TierSlot {
    key: Option<ExpertKey>,
    last_use: u64,
    /// Pre-warmed hot expert: evicted only when nothing else is evictable.
    sticky: bool,
    /// Guards the slot from eviction while the current pass still needs it.
    pinned_pass: u64,
    len: usize,
    /// Payload start within the slot region. 0 for CPU-written entries; the
    /// sub-block head of the extent for io_uring O_DIRECT reads that DMA the
    /// block-aligned span straight into the slot.
    data_offset: usize,
}

/// Fixed-budget pinned-host expert cache with LRU recency tickets, keyed like
/// the VRAM pool.
///
/// Slots are `slot_stride` bytes apart but hold at most `slot_bytes` of
/// payload. The two differ only in io_uring mode, where the stride is rounded
/// up so every slot base is a legal O_DIRECT destination with room for the
/// block-aligned span of any payload ([`crate::expert_uring::tier_slot_stride`]).
pub(crate) struct RamTier {
    arena: TierArena,
    slot_bytes: usize,
    slot_stride: usize,
    /// Arena byte offset of slot 0 (aligns ring-mode slot bases to
    /// [`SLOT_DMA_ALIGN`] within an unaligned pinned allocation; 0 otherwise).
    base_offset: usize,
    slots: Vec<TierSlot>,
    resident: HashMap<ExpertKey, usize>,
    tick: u64,
    stats: RamTierStats,
    budget_bytes: u64,
}

impl RamTier {
    pub(crate) fn new_with_stride(
        arena: TierArena,
        slot_bytes: usize,
        slot_stride: usize,
        slot_count: usize,
        base_offset: usize,
    ) -> Self {
        assert!(slot_stride >= slot_bytes, "stride must hold the payload");
        Self {
            arena,
            slot_bytes,
            slot_stride,
            base_offset,
            slots: (0..slot_count)
                .map(|_| TierSlot {
                    key: None,
                    last_use: 0,
                    sticky: false,
                    pinned_pass: 0,
                    len: 0,
                    data_offset: 0,
                })
                .collect(),
            resident: HashMap::new(),
            tick: 0,
            stats: RamTierStats::default(),
            budget_bytes: slot_count as u64 * slot_stride as u64,
        }
    }

    pub(crate) fn slot_stride(&self) -> usize {
        self.slot_stride
    }

    pub(crate) fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Arena byte offset of a slot's base.
    fn slot_base(&self, slot: usize) -> usize {
        self.base_offset + slot * self.slot_stride
    }

    pub(crate) fn arena(&self) -> &TierArena {
        &self.arena
    }

    pub(crate) fn stats(&self) -> RamTierStats {
        self.stats
    }

    pub(crate) fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn resident_count(&self) -> usize {
        self.resident.len()
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.slots
            .iter()
            .filter(|slot| slot.key.is_some())
            .map(|slot| slot.len as u64)
            .sum()
    }

    pub(crate) fn sticky_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.key.is_some() && slot.sticky)
            .count()
    }

    /// Recency bump for an expert the router selected this pass, wherever it
    /// is served from (device hits included), so tier residency tracks actual
    /// routing heat rather than only device-miss traffic.
    pub(crate) fn touch(&mut self, key: &ExpertKey, pass: u64) {
        if let Some(&slot) = self.resident.get(key) {
            self.tick += 1;
            self.slots[slot].last_use = self.tick;
            self.slots[slot].pinned_pass = pass;
        }
    }

    /// Tier lookup for a device-pool miss: `Some((arena_offset, len))` serves
    /// the upload straight from host RAM.
    pub(crate) fn lookup(&mut self, key: &ExpertKey, pass: u64) -> Option<(usize, usize)> {
        match self.resident.get(key).copied() {
            Some(slot) => {
                self.tick += 1;
                self.slots[slot].last_use = self.tick;
                self.slots[slot].pinned_pass = pass;
                self.stats.hits += 1;
                self.stats.bytes_served += self.slots[slot].len as u64;
                Some((
                    self.slot_base(slot) + self.slots[slot].data_offset,
                    self.slots[slot].len,
                ))
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Cache one expert's bytes, evicting the LRU non-sticky entry if needed
    /// (sticky entries are only reclaimed when nothing else is evictable).
    /// Returns the arena offset, or None when every slot is pinned by the
    /// current pass (the caller simply skips caching — never an error).
    pub(crate) fn insert(
        &mut self,
        key: ExpertKey,
        bytes: &[u8],
        sticky: bool,
        pass: u64,
    ) -> Option<usize> {
        if bytes.len() > self.slot_bytes {
            return None;
        }
        if let Some(&slot) = self.resident.get(&key) {
            // Already cached (e.g. pre-warmed): refresh recency only.
            self.tick += 1;
            self.slots[slot].last_use = self.tick;
            self.slots[slot].pinned_pass = pass;
            return Some(self.slot_base(slot) + self.slots[slot].data_offset);
        }
        let slot = self.take_slot(pass)?;
        let offset = self.slot_base(slot);
        if self.arena.write(offset, bytes).is_err() {
            return None;
        }
        self.tick += 1;
        let entry = &mut self.slots[slot];
        entry.key = Some(key);
        entry.last_use = self.tick;
        entry.sticky = sticky;
        entry.pinned_pass = pass;
        entry.len = bytes.len();
        entry.data_offset = 0;
        self.resident.insert(key, slot);
        self.stats.inserts += 1;
        if sticky {
            self.stats.prewarmed += 1;
        }
        Some(offset)
    }

    /// Claim a slot for `key` so a DMA/O_DIRECT read can land directly in the
    /// arena (no CPU write). Returns `(slot_index, slot_base_arena_offset)`;
    /// the entry is provisional and MUST be finished with
    /// [`RamTier::commit_reserved`] (payload at `data_offset` within the
    /// slot, `len` bytes) or rolled back with [`RamTier::abort_reserved`]
    /// before any other tier call for that key. Declines exactly like
    /// [`RamTier::insert`]: payload too large or every slot pinned by the
    /// current pass.
    pub(crate) fn reserve(
        &mut self,
        key: ExpertKey,
        len: usize,
        sticky: bool,
        pass: u64,
    ) -> Option<(usize, usize)> {
        if len > self.slot_bytes {
            return None;
        }
        if let Some(&slot) = self.resident.get(&key) {
            // Already cached: re-reading identical bytes into the same slot
            // is harmless; refresh recency and let commit update the offsets.
            self.tick += 1;
            self.slots[slot].last_use = self.tick;
            self.slots[slot].pinned_pass = pass;
            return Some((slot, self.slot_base(slot)));
        }
        let slot = self.take_slot(pass)?;
        self.tick += 1;
        let entry = &mut self.slots[slot];
        entry.key = Some(key);
        entry.last_use = self.tick;
        entry.sticky = sticky;
        entry.pinned_pass = pass;
        entry.len = len;
        entry.data_offset = 0;
        self.resident.insert(key, slot);
        Some((slot, self.slot_base(slot)))
    }

    /// Publish a reserved slot's payload location once its read landed.
    pub(crate) fn commit_reserved(&mut self, slot: usize, data_offset: usize) {
        let entry = &mut self.slots[slot];
        debug_assert!(entry.key.is_some(), "committing an unreserved slot");
        debug_assert!(
            data_offset + entry.len <= self.slot_stride,
            "committed payload exceeds the slot stride"
        );
        entry.data_offset = data_offset;
        self.stats.inserts += 1;
        if entry.sticky {
            self.stats.prewarmed += 1;
        }
    }

    /// Roll back a reserved slot whose read failed: the key is forgotten and
    /// the slot returns to the free pool.
    pub(crate) fn abort_reserved(&mut self, slot: usize) {
        if let Some(key) = self.slots[slot].key.take() {
            self.resident.remove(&key);
        }
        let entry = &mut self.slots[slot];
        entry.sticky = false;
        entry.pinned_pass = 0;
        entry.len = 0;
        entry.data_offset = 0;
    }

    fn take_slot(&mut self, pass: u64) -> Option<usize> {
        if let Some(free) = self.slots.iter().position(|slot| slot.key.is_none()) {
            return Some(free);
        }
        let victim = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.sticky && slot.pinned_pass != pass)
            .min_by_key(|(_, slot)| slot.last_use)
            .map(|(idx, _)| idx)
            .or_else(|| {
                // Sticky fallback: reclaim the coldest sticky entry rather
                // than refusing to cache anything ever again.
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.pinned_pass != pass)
                    .min_by_key(|(_, slot)| slot.last_use)
                    .map(|(idx, _)| idx)
            })?;
        if let Some(old_key) = self.slots[victim].key.take() {
            self.resident.remove(&old_key);
            self.slots[victim].sticky = false;
            self.stats.evictions += 1;
        }
        Some(victim)
    }
}

// ---------------------------------------------------------------------------
// Usage stats (the learning cache)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct UsageFileV1 {
    version: u32,
    model: String,
    passes: u64,
    /// "layer:expert" -> selection count (BTreeMap for stable file output).
    counts: BTreeMap<String, u64>,
}

/// Per-(layer, expert) selection counters with atomic (tmp+rename)
/// persistence to `<model_dir>/.hi_expert_usage` on a cadence and at drop.
pub(crate) struct ExpertUsage {
    path: PathBuf,
    model: String,
    counts: HashMap<(u32, u32), u64>,
    passes: u64,
    dirty: bool,
    save_every: Duration,
    last_save: Instant,
}

impl ExpertUsage {
    /// Load history for `model_path` (never fails: unreadable, corrupt, or
    /// other-model files are ignored and overwritten on the next save).
    pub(crate) fn load_or_new(model_path: &Path, save_secs: u64) -> Self {
        let dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let model = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
            .to_string();
        let path = dir.join(USAGE_FILE_NAME);
        let mut usage = Self {
            path,
            model,
            counts: HashMap::new(),
            passes: 0,
            dirty: false,
            save_every: Duration::from_secs(save_secs.max(1)),
            last_save: Instant::now(),
        };
        if let Ok(raw) = std::fs::read_to_string(&usage.path)
            && let Ok(file) = serde_json::from_str::<UsageFileV1>(&raw)
            && file.version == USAGE_FILE_VERSION
            && file.model == usage.model
        {
            usage.passes = file.passes;
            for (key, count) in file.counts {
                if let Some((layer, expert)) = key.split_once(':')
                    && let (Ok(layer), Ok(expert)) = (layer.parse(), expert.parse())
                {
                    usage.counts.insert((layer, expert), count);
                }
            }
        }
        usage
    }

    pub(crate) fn record_pass(&mut self, selected: impl IntoIterator<Item = (u32, u32)>) {
        let mut any = false;
        for key in selected {
            *self.counts.entry(key).or_insert(0) += 1;
            any = true;
        }
        if any {
            self.passes += 1;
            self.dirty = true;
        }
    }

    /// Cadenced save; call once per ensure pass (cheap: an Instant compare).
    pub(crate) fn maybe_save(&mut self) {
        if self.dirty && self.last_save.elapsed() >= self.save_every {
            let _ = self.save();
        }
    }

    /// Atomic write: serialize to `<file>.tmp` in the same directory, fsync,
    /// rename over the target.
    pub(crate) fn save(&mut self) -> Result<()> {
        let file = UsageFileV1 {
            version: USAGE_FILE_VERSION,
            model: self.model.clone(),
            passes: self.passes,
            counts: self
                .counts
                .iter()
                .map(|((layer, expert), count)| (format!("{layer}:{expert}"), *count))
                .collect(),
        };
        let json = serde_json::to_string(&file).context("serializing expert usage")?;
        let tmp = self.path.with_extension("tmp");
        {
            use std::io::Write;
            let mut out = std::fs::File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            out.write_all(json.as_bytes())
                .with_context(|| format!("writing {}", tmp.display()))?;
            out.sync_all().ok();
        }
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("renaming {} over {}", tmp.display(), self.path.display()))?;
        self.dirty = false;
        self.last_save = Instant::now();
        Ok(())
    }

    pub(crate) fn save_if_dirty(&mut self) {
        if self.dirty {
            let _ = self.save();
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.counts.len()
    }

    /// (layer, expert) pairs hottest-first, for pre-warm placement.
    pub(crate) fn ranked(&self) -> Vec<(u32, u32)> {
        let mut entries: Vec<_> = self
            .counts
            .iter()
            .map(|(&key, &count)| (key, count))
            .collect();
        // Deterministic order: count desc, then key asc.
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.into_iter().map(|(key, _)| key).collect()
    }
}

// ---------------------------------------------------------------------------
// Source discovery (for pre-warm, madvise extents and readahead)
// ---------------------------------------------------------------------------

/// Rebuild the streamable (layer, projection) -> source map from the GGUF the
/// pool re-opened, mirroring the qwen GPU loader's tensor-name resolution
/// (same `hi_gguf` alias helpers), so pre-warm keys match `ensure_resident`
/// request keys exactly. Best-effort: any surprise yields an empty map and
/// the pool simply skips pre-warm/madvise.
pub(crate) fn discover_sources(gguf: &GgufFile) -> BTreeMap<(u32, u8), ExpertSource> {
    let mut sources = BTreeMap::new();
    let Ok(config) = gguf.qwen_config() else {
        return sources;
    };
    let Some(experts) = config.expert_count else {
        return sources;
    };
    let experts_u64 = u64::from(experts);
    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        let router_present = qwen_moe_router_weight_names(&prefix)
            .iter()
            .any(|name| gguf.tensor_info(name).is_some());
        if !router_present {
            continue;
        }
        let mut layer_sources = Vec::new();
        for (proj_idx, projection) in ["gate", "up", "down"].iter().enumerate() {
            let Some((name, info)) = qwen_moe_packed_expert_weight_names(&prefix, projection)
                .into_iter()
                .find_map(|name| gguf.tensor_info(&name).cloned().map(|info| (name, info)))
            else {
                layer_sources.clear();
                break;
            };
            if info.dimensions.len() != 3 || info.dimensions[2] != experts_u64 {
                layer_sources.clear();
                break;
            }
            let Some(per_expert_elements) = info.dimensions[0].checked_mul(info.dimensions[1])
            else {
                layer_sources.clear();
                break;
            };
            let Ok(bytes_per_expert) = info
                .dtype
                .byte_len(per_expert_elements)
                .and_then(|bytes| usize::try_from(bytes).context("bytes"))
            else {
                layer_sources.clear();
                break;
            };
            layer_sources.push((
                (layer, proj_idx as u8),
                ExpertSource {
                    tensor_name: name,
                    bytes_per_expert,
                    rows: info.dimensions[1] as usize,
                    cols: info.dimensions[0] as usize,
                    dtype: info.dtype,
                    expert_count: experts as usize,
                },
            ));
        }
        for (key, source) in layer_sources {
            sources.insert(key, source);
        }
    }
    sources
}

/// Total bytes of streamable routed experts on disk (tier budget cap).
pub(crate) fn total_expert_bytes(sources: &BTreeMap<(u32, u8), ExpertSource>) -> u64 {
    sources
        .values()
        .map(|source| source.bytes_per_expert as u64 * source.expert_count as u64)
        .sum()
}

/// Total streamable expert slices (one per (layer, projection, expert)). In
/// ring mode the tier budget caps at `slices x stride` rather than payload
/// bytes, since every cached slice occupies a full stride-widened slot.
pub(crate) fn total_expert_slices(sources: &BTreeMap<(u32, u8), ExpertSource>) -> u64 {
    sources
        .values()
        .map(|source| source.expert_count as u64)
        .sum()
}

// ---------------------------------------------------------------------------
// Disk fetch path
// ---------------------------------------------------------------------------

/// How expert extents leave the disk.
pub(crate) enum FetchBackend {
    /// Buffered mmap copies, optionally WILLNEED-hinted per extent (since
    /// MADV_RANDOM turns off the kernel's own readahead).
    Mmap { willneed_inline: bool },
    /// O_DIRECT positioned reads on twin fds (≤ READ_WORKERS threads deep).
    Direct(GgufDirectReader),
    /// Batch-submitted io_uring O_DIRECT reads (whole miss batches at queue
    /// depth; zero-copy into pinned tier slots where the pool arranges it).
    #[cfg(target_os = "linux")]
    Uring(crate::expert_uring::IoUringReader),
}

/// Reads one expert's contiguous byte extent through the chosen
/// [`FetchBackend`]. Thread-safe (`&self` only), so the pool's read workers
/// share it without locking.
pub(crate) struct ExpertFetcher {
    gguf: Arc<GgufFile>,
    backend: FetchBackend,
}

impl ExpertFetcher {
    pub(crate) fn new(gguf: Arc<GgufFile>, backend: FetchBackend) -> Self {
        Self { gguf, backend }
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn uring(&self) -> Option<&crate::expert_uring::IoUringReader> {
        match &self.backend {
            FetchBackend::Uring(reader) => Some(reader),
            _ => None,
        }
    }

    /// `io=` label for the startup line and the /health stats segment.
    pub(crate) fn io_label(&self) -> String {
        match &self.backend {
            FetchBackend::Mmap { .. } => "mmap".to_string(),
            FetchBackend::Direct(_) => "odirect".to_string(),
            #[cfg(target_os = "linux")]
            FetchBackend::Uring(reader) => format!(
                "iouring(qd={}{})",
                reader.queue_depth(),
                if reader.buffers_registered() {
                    ",regbuf"
                } else {
                    ""
                }
            ),
        }
    }

    /// The expert's byte extent within its tensor: (tensor name, offset, len).
    pub(crate) fn extent(key: ExpertKey, source: &ExpertSource) -> Result<(u64, u64)> {
        let expert = key.2 as usize;
        if expert >= source.expert_count {
            bail!(
                "expert {} out of range for {} ({} experts)",
                expert,
                source.tensor_name,
                source.expert_count
            );
        }
        let start = (expert as u64)
            .checked_mul(source.bytes_per_expert as u64)
            .ok_or_else(|| anyhow!("expert byte offset overflows u64"))?;
        Ok((start, source.bytes_per_expert as u64))
    }

    /// The expert's absolute on-disk extent: `(shard, file_offset, len)`,
    /// bounds-checked against the tensor (the direct-I/O and io_uring paths).
    pub(crate) fn file_extent(
        &self,
        key: ExpertKey,
        source: &ExpertSource,
    ) -> Result<(usize, u64, usize)> {
        let (start, len) = Self::extent(key, source)?;
        let range = self.gguf.tensor_file_range(&source.tensor_name)?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("expert byte range overflows u64"))?;
        if end > range.len {
            bail!(
                "expert {} byte range {start}..{end} exceeds tensor {} ({} bytes)",
                key.2,
                source.tensor_name,
                range.len
            );
        }
        Ok((
            range.shard,
            range.file_offset + start,
            usize::try_from(len).context("expert byte length does not fit usize")?,
        ))
    }

    pub(crate) fn fetch(&self, key: ExpertKey, source: &ExpertSource) -> Result<Vec<u8>> {
        let willneed_inline = match &self.backend {
            FetchBackend::Direct(direct) => {
                let (shard, offset, len) = self.file_extent(key, source)?;
                return direct.read_range(shard, offset, len);
            }
            #[cfg(target_os = "linux")]
            FetchBackend::Uring(reader) => {
                let (shard, offset, len) = self.file_extent(key, source)?;
                return reader
                    .read_owned(&[(shard, offset, len)])
                    .pop()
                    .expect("one job submitted");
            }
            FetchBackend::Mmap { willneed_inline } => *willneed_inline,
        };
        let (start, len) = Self::extent(key, source)?;
        if willneed_inline {
            // MADV_RANDOM disabled the kernel's speculative readahead, so ask
            // for exactly this extent before faulting through it; issued from
            // the read worker, never the routing thread.
            let _ = self.gguf.advise_tensor_range(
                &source.tensor_name,
                start,
                len,
                GgufMemoryAdvice::WillNeed,
            );
        }
        let view = self
            .gguf
            .tensor(&source.tensor_name)
            .ok_or_else(|| anyhow!("expert tensor {} missing from GGUF", source.tensor_name))?;
        let start = usize::try_from(start).context("expert byte offset does not fit usize")?;
        let len = usize::try_from(len).context("expert byte length does not fit usize")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("expert byte range overflows usize"))?;
        let bytes = view.bytes.get(start..end).ok_or_else(|| {
            anyhow!(
                "expert {} byte range {start}..{end} exceeds tensor {} ({} bytes)",
                key.2,
                source.tensor_name,
                view.bytes.len()
            )
        })?;
        Ok(bytes.to_vec())
    }

    /// Batch fetch through the io_uring backend (owned buffers, whole batch
    /// at queue depth in one drive loop); `None` when the backend is not
    /// io_uring, in which case callers use the thread pool.
    pub(crate) fn fetch_batch_uring(
        &self,
        jobs: &[(ExpertKey, &ExpertSource)],
    ) -> Option<Vec<Result<Vec<u8>>>> {
        #[cfg(target_os = "linux")]
        {
            let FetchBackend::Uring(reader) = &self.backend else {
                return None;
            };
            let mut results: Vec<Option<Result<Vec<u8>>>> = (0..jobs.len()).map(|_| None).collect();
            let mut ring_jobs: Vec<(usize, (usize, u64, usize))> = Vec::with_capacity(jobs.len());
            for (idx, (key, source)) in jobs.iter().enumerate() {
                match self.file_extent(*key, source) {
                    Ok(extent) => ring_jobs.push((idx, extent)),
                    Err(err) => results[idx] = Some(Err(err)),
                }
            }
            let extents: Vec<(usize, u64, usize)> =
                ring_jobs.iter().map(|(_, extent)| *extent).collect();
            for ((idx, _), result) in ring_jobs.iter().zip(reader.read_owned(&extents)) {
                results[*idx] = Some(result);
            }
            Some(
                results
                    .into_iter()
                    .map(|slot| slot.expect("every job resolved or read"))
                    .collect(),
            )
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = jobs;
            None
        }
    }
}

// ---------------------------------------------------------------------------
// WILLNEED readahead thread
// ---------------------------------------------------------------------------

/// Dedicated madvise(WILLNEED) thread: the pool posts the extents of the next
/// expert block it is about to read and the kernel starts readahead while the
/// read workers are still working through earlier extents. Never issued
/// inline on the routing thread (colibri measured ~0.5 ms/call there).
pub(crate) struct WillNeedThread {
    tx: Option<std::sync::mpsc::Sender<Vec<(String, u64, u64)>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl WillNeedThread {
    pub(crate) fn spawn(gguf: Arc<GgufFile>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(String, u64, u64)>>();
        let handle = std::thread::Builder::new()
            .name("hi-expert-willneed".to_string())
            .spawn(move || {
                while let Ok(batch) = rx.recv() {
                    for (tensor, offset, len) in batch {
                        let _ = gguf.advise_tensor_range(
                            &tensor,
                            offset,
                            len,
                            GgufMemoryAdvice::WillNeed,
                        );
                    }
                }
            })
            .ok();
        Self {
            tx: handle.is_some().then_some(tx),
            handle,
        }
    }

    pub(crate) fn hint(&self, extents: Vec<(String, u64, u64)>) {
        if extents.is_empty() {
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(extents);
        }
    }
}

impl Drop for WillNeedThread {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Async upload engine (dsv4 CopyEngine pattern on runtime.rs primitives)
// ---------------------------------------------------------------------------

/// Double-buffered pinned staging + a non-blocking copy stream + events.
///
/// Per ensure pass: `begin_pass` host-waits the previous pass's `done` event
/// before any tier slot or staging region is overwritten (their DMAs must
/// have landed), then records `fork` on the engine stream so copies order
/// after every GEMV already enqueued (a copy may target an evicted expert's
/// device slot). Uploads whose source is the pinned tier skip staging
/// entirely; everything else stages through slot-sized regions that cycle
/// across [`STAGING_GENERATIONS`] pinned buffers with per-generation guard
/// events. `finish_pass` records `done`; with an engine stream the GEMVs wait
/// on it device-side (no host stall), without one it host-synchronizes.
pub(crate) struct UploadEngine {
    stream: Stream,
    fork: Event,
    done: Event,
    done_armed: bool,
    staging: Vec<StagingGen>,
    slot_bytes: usize,
    generation: usize,
    region: usize,
}

struct StagingGen {
    pinned: PinnedBuffer,
    guard: Event,
    armed: bool,
}

impl UploadEngine {
    pub(crate) fn create(slot_bytes: usize) -> Result<Self> {
        let mut staging = Vec::with_capacity(STAGING_GENERATIONS);
        let bytes = slot_bytes
            .checked_mul(STAGING_SLOTS)
            .context("expert staging byte count overflows usize")?;
        for _ in 0..STAGING_GENERATIONS {
            staging.push(StagingGen {
                pinned: PinnedBuffer::alloc(bytes).context("allocating pinned expert staging")?,
                guard: Event::create().context("creating expert staging guard event")?,
                armed: false,
            });
        }
        Ok(Self {
            stream: Stream::create_non_blocking().context("creating expert copy stream")?,
            fork: Event::create().context("creating expert copy fork event")?,
            done: Event::create().context("creating expert copy done event")?,
            done_armed: false,
            staging,
            slot_bytes,
            generation: 0,
            region: 0,
        })
    }

    pub(crate) fn staging_bytes(slot_bytes: usize) -> u64 {
        slot_bytes as u64 * STAGING_SLOTS as u64 * STAGING_GENERATIONS as u64
    }

    /// Start a pass. Must run before any tier-arena write of the pass.
    pub(crate) fn begin_pass(&mut self, engine_stream: Option<&Stream>) -> Result<()> {
        if self.done_armed {
            // Previous pass's copies read tier slots / staging we may now
            // overwrite; they are long complete (a whole forward happened in
            // between), so this is a formality, not a stall.
            self.done.synchronize()?;
            self.done_armed = false;
        }
        if let Some(engine) = engine_stream {
            self.fork.record(engine)?;
            self.stream.wait_event(&self.fork)?;
        }
        self.region = 0;
        Ok(())
    }

    /// Async H2D straight from pinned memory (a tier hit or fresh insert):
    /// no staging copy.
    pub(crate) fn upload_pinned(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &PinnedBuffer,
        src_offset: usize,
        len: usize,
    ) -> Result<()> {
        dst.copy_from_pinned_at_async(dst_offset, src, src_offset, len, &self.stream)
    }

    /// Stage pageable bytes through the current pinned region and enqueue the
    /// async H2D; regions cycle through the double-buffered generations.
    pub(crate) fn stage_upload(
        &mut self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        bytes: &[u8],
    ) -> Result<()> {
        if bytes.len() > self.slot_bytes {
            bail!(
                "staged expert upload of {} bytes exceeds the {}-byte staging region",
                bytes.len(),
                self.slot_bytes
            );
        }
        if self.region == 0 {
            let generation = &mut self.staging[self.generation];
            if generation.armed {
                generation.guard.synchronize()?;
                generation.armed = false;
            }
        }
        let generation_index = self.generation;
        let offset = self.region * self.slot_bytes;
        {
            let generation = &self.staging[generation_index];
            generation.pinned.copy_in(offset, bytes)?;
            dst.copy_from_pinned_at_async(
                dst_offset,
                &generation.pinned,
                offset,
                bytes.len(),
                &self.stream,
            )?;
        }
        self.region += 1;
        if self.region == STAGING_SLOTS {
            let generation = &mut self.staging[generation_index];
            generation.guard.record(&self.stream)?;
            generation.armed = true;
            self.generation = (generation_index + 1) % self.staging.len();
            self.region = 0;
        }
        Ok(())
    }

    /// Host-wait for any copies still in flight (teardown safety: the device
    /// arena and pinned staging must not be freed under an active DMA).
    pub(crate) fn drain(&mut self) {
        if self.done_armed {
            let _ = self.done.synchronize();
            self.done_armed = false;
        }
    }

    /// End a pass: guard any partially-filled generation, record `done`, and
    /// either chain the engine stream on it (overlap) or host-wait (callers
    /// without a stream must see completed uploads on return).
    pub(crate) fn finish_pass(&mut self, engine_stream: Option<&Stream>) -> Result<()> {
        if self.region > 0 {
            let generation = &mut self.staging[self.generation];
            generation.guard.record(&self.stream)?;
            generation.armed = true;
            self.generation = (self.generation + 1) % self.staging.len();
            self.region = 0;
        }
        self.done.record(&self.stream)?;
        match engine_stream {
            Some(engine) => {
                engine.wait_event(&self.done)?;
                self.done_armed = true;
            }
            None => {
                self.done.synchronize()?;
                self.done_armed = false;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests (CUDA-free: heap arenas and pure plans only)
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn key(layer: u32, proj: u8, expert: u32) -> ExpertKey {
        (layer, proj, expert)
    }

    fn heap_tier(slot_bytes: usize, slots: usize) -> RamTier {
        RamTier::new_with_stride(
            TierArena::Heap(vec![0u8; slot_bytes * slots]),
            slot_bytes,
            slot_bytes,
            slots,
            0,
        )
    }

    #[test]
    fn budget_plan_auto_subtracts_itemized_slack_and_halves() {
        let staging = UploadEngine::staging_bytes(1 << 20);
        let plan = plan_tier_budget(None, Some(32 * GIB), 1 << 20, u64::MAX, staging);
        let slack = PAGE_CACHE_RESERVE + WORKING_SET_SLAB + staging + MISC_RESERVE;
        let expected = ((32 * GIB - slack) as f64 * AUTO_FRACTION) as u64;
        let expected_slots = expected / (1 << 20);
        assert_eq!(plan.slots as u64, expected_slots);
        assert_eq!(plan.budget_bytes, expected_slots * (1 << 20));
        assert!(plan.enabled());
        assert_eq!(plan.source, "auto");
        let line = plan.describe();
        assert!(line.contains("page-cache=3.0GiB"), "{line}");
        assert!(line.contains("working-set=1.5GiB"), "{line}");
        assert!(line.contains("projected-peak"), "{line}");
    }

    #[test]
    fn budget_plan_explicit_env_wins_and_zero_disables() {
        let plan = plan_tier_budget(Some(2.0), Some(8 * GIB), 1 << 20, u64::MAX, 0);
        assert_eq!(plan.budget_bytes, 2 * GIB);
        assert_eq!(plan.slots, 2048);
        assert_eq!(plan.source, "HI_CUDA_EXPERT_RAM_GB");

        let off = plan_tier_budget(Some(0.0), Some(8 * GIB), 1 << 20, u64::MAX, 0);
        assert!(!off.enabled());
        assert_eq!(off.budget_bytes, 0);
    }

    #[test]
    fn budget_plan_caps_at_total_expert_bytes_and_disables_when_tiny() {
        // 10 slots of expert data on disk: never pin more than that.
        let plan = plan_tier_budget(Some(64.0), Some(64 * GIB), 1 << 20, 10 << 20, 0);
        assert_eq!(plan.budget_bytes, 10 << 20);
        assert_eq!(plan.slots, 10);

        // Under 2 slots -> disabled.
        let tiny = plan_tier_budget(Some(64.0), Some(64 * GIB), 1 << 20, 1 << 20, 0);
        assert!(!tiny.enabled());

        // No MemAvailable and no explicit budget -> disabled, not a guess.
        let unknown = plan_tier_budget(None, None, 1 << 20, u64::MAX, 0);
        assert!(!unknown.enabled());
    }

    #[test]
    fn ram_tier_is_bounded_and_evicts_lru_first() {
        let mut tier = heap_tier(16, 3);
        assert!(tier.insert(key(0, 0, 0), &[1u8; 16], false, 1).is_some());
        assert!(tier.insert(key(0, 0, 1), &[2u8; 16], false, 1).is_some());
        assert!(tier.insert(key(0, 0, 2), &[3u8; 16], false, 1).is_some());
        assert_eq!(tier.resident_count(), 3);
        assert_eq!(tier.resident_bytes(), 48);
        assert!(tier.resident_bytes() <= tier.budget_bytes());

        // Touch expert 0 so expert 1 is the LRU victim.
        tier.touch(&key(0, 0, 0), 2);
        assert!(tier.insert(key(0, 0, 3), &[4u8; 16], false, 2).is_some());
        assert_eq!(tier.resident_count(), 3);
        assert!(tier.lookup(&key(0, 0, 1), 3).is_none(), "LRU entry evicted");
        assert!(tier.lookup(&key(0, 0, 0), 3).is_some());
        let stats = tier.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.inserts, 4);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        // Bounded forever: hammer more inserts than capacity.
        for expert in 10..30 {
            tier.insert(key(1, 0, expert), &[9u8; 16], false, u64::from(expert));
            assert!(tier.resident_bytes() <= tier.budget_bytes());
            assert_eq!(tier.resident_count(), 3);
        }
    }

    #[test]
    fn ram_tier_sticky_entries_survive_pressure_and_pass_pinning_holds() {
        let mut tier = heap_tier(8, 3);
        assert!(tier.insert(key(0, 0, 7), &[7u8; 8], true, 0).is_some());
        for round in 0..10u32 {
            tier.insert(
                key(0, 0, round),
                &[round as u8; 8],
                false,
                u64::from(round) + 1,
            );
        }
        // The sticky pre-warmed expert is still resident despite pressure.
        assert!(tier.lookup(&key(0, 0, 7), 100).is_some());
        assert_eq!(tier.sticky_count(), 1);

        // Entries pinned by the current pass are not evictable: with all 3
        // slots pinned in pass 200, an insert must decline rather than evict.
        let mut tier = heap_tier(8, 3);
        assert!(tier.insert(key(0, 0, 0), &[0u8; 8], false, 200).is_some());
        assert!(tier.insert(key(0, 0, 1), &[1u8; 8], false, 200).is_some());
        assert!(tier.insert(key(0, 0, 2), &[2u8; 8], false, 200).is_some());
        assert!(tier.insert(key(0, 0, 3), &[3u8; 8], false, 200).is_none());
        // Next pass, eviction works again.
        assert!(tier.insert(key(0, 0, 3), &[3u8; 8], false, 201).is_some());
    }

    #[test]
    fn ram_tier_reserve_commit_abort_with_widened_stride() {
        // Ring-mode geometry: stride wider than the payload cap, payloads
        // landing at a non-zero data offset (the O_DIRECT head), slot 0
        // shifted to an aligned base within the arena.
        let (slot_bytes, stride, slots, base_offset) = (16, 24, 3, 8);
        let mut tier = RamTier::new_with_stride(
            TierArena::Heap(vec![0u8; base_offset + stride * slots]),
            slot_bytes,
            stride,
            slots,
            base_offset,
        );
        assert_eq!(tier.slot_stride(), stride);
        assert_eq!(tier.base_offset(), base_offset);
        assert_eq!(tier.budget_bytes(), (stride * slots) as u64);

        // Reserve hands out slot-base offsets on stride boundaries past the
        // aligned base.
        let (slot_a, base_a) = tier.reserve(key(0, 0, 1), 10, false, 1).unwrap();
        assert_eq!(base_a, base_offset + slot_a * stride);
        // Oversized payloads are declined exactly like insert.
        assert!(
            tier.reserve(key(0, 0, 2), slot_bytes + 1, false, 1)
                .is_none()
        );

        // Commit publishes the payload at its head offset; lookups serve it.
        tier.commit_reserved(slot_a, 3);
        let (offset, len) = tier.lookup(&key(0, 0, 1), 2).unwrap();
        assert_eq!(offset, base_a + 3);
        assert_eq!(len, 10);
        assert_eq!(tier.stats().inserts, 1);

        // Abort forgets the key and frees the slot.
        let (slot_b, _) = tier.reserve(key(0, 0, 3), 8, false, 2).unwrap();
        tier.abort_reserved(slot_b);
        assert!(tier.lookup(&key(0, 0, 3), 3).is_none());
        let (slot_c, _) = tier.reserve(key(0, 0, 4), 8, false, 3).unwrap();
        assert_eq!(slot_b, slot_c, "aborted slot is reusable");

        // Reserved entries pinned by the current pass are not evictable: with
        // all slots reserved in one pass, another reserve declines.
        let mut tier = RamTier::new_with_stride(
            TierArena::Heap(vec![0u8; stride * slots]),
            slot_bytes,
            stride,
            slots,
            0,
        );
        for expert in 0..slots as u32 {
            assert!(tier.reserve(key(1, 0, expert), 8, false, 7).is_some());
        }
        assert!(tier.reserve(key(1, 0, 99), 8, false, 7).is_none());
        // Next pass evicts LRU reservations as usual.
        assert!(tier.reserve(key(1, 0, 99), 8, false, 8).is_some());
        assert_eq!(tier.resident_count(), slots);

        // CPU inserts under a widened stride land on stride boundaries with
        // data_offset 0 and stay bounded.
        let mut tier = RamTier::new_with_stride(
            TierArena::Heap(vec![0u8; stride * slots]),
            slot_bytes,
            stride,
            slots,
            0,
        );
        for round in 0..10u32 {
            if let Some(offset) = tier.insert(
                key(2, 0, round),
                &[round as u8; 16],
                false,
                u64::from(round),
            ) {
                assert_eq!(offset % stride, 0);
            }
            assert!(tier.resident_bytes() <= tier.budget_bytes());
        }
    }

    #[test]
    fn ram_tier_heap_hits_serve_stored_bytes() {
        let mut tier = heap_tier(8, 2);
        let payload = [5u8, 6, 7, 8];
        let offset = tier.insert(key(1, 2, 3), &payload, false, 1).unwrap();
        let (hit_offset, len) = tier.lookup(&key(1, 2, 3), 2).unwrap();
        assert_eq!(offset, hit_offset);
        assert_eq!(len, payload.len());
        assert_eq!(tier.arena().heap_slice(hit_offset, len).unwrap(), &payload);
        // Oversized payloads are declined, never split.
        assert!(tier.insert(key(1, 2, 4), &[0u8; 9], false, 3).is_none());
    }

    #[test]
    fn expert_usage_round_trips_atomically() {
        let dir = std::env::temp_dir().join(format!(
            "hi-expert-usage-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("model.gguf");

        let mut usage = ExpertUsage::load_or_new(&model_path, 60);
        assert!(usage.is_empty());
        usage.record_pass([(0, 3), (0, 5), (1, 3)]);
        usage.record_pass([(0, 3)]);
        usage.save().unwrap();
        assert!(dir.join(USAGE_FILE_NAME).exists());
        assert!(
            !dir.join(".hi_expert_usage.tmp").exists(),
            "tmp renamed away"
        );

        let reloaded = ExpertUsage::load_or_new(&model_path, 60);
        assert_eq!(reloaded.len(), 3);
        assert_eq!(reloaded.passes, 2);
        assert_eq!(reloaded.ranked().first(), Some(&(0, 3)));

        // A different model in the same directory ignores the history.
        let other = ExpertUsage::load_or_new(&dir.join("other.gguf"), 60);
        assert!(other.is_empty());

        // Corrupt files are ignored, not fatal.
        std::fs::write(dir.join(USAGE_FILE_NAME), b"{not json").unwrap();
        let corrupt = ExpertUsage::load_or_new(&model_path, 60);
        assert!(corrupt.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn expert_usage_ranks_hottest_first_deterministically() {
        let dir = std::env::temp_dir();
        let mut usage = ExpertUsage::load_or_new(&dir.join("nonexistent-model.gguf"), 0);
        usage.record_pass([(2, 1), (0, 9)]);
        usage.record_pass([(2, 1)]);
        usage.record_pass([(2, 1), (1, 4)]);
        // (2,1) x3, then ties (0,9) and (1,4) x1 resolved by key order.
        assert_eq!(usage.ranked(), vec![(2, 1), (0, 9), (1, 4)]);
    }

    #[test]
    fn fetcher_extent_math_checks_bounds() {
        let source = ExpertSource {
            tensor_name: "blk.0.ffn_gate_exps.weight".to_string(),
            bytes_per_expert: 100,
            rows: 10,
            cols: 10,
            dtype: hi_gguf::GgufTensorType::F32,
            expert_count: 4,
        };
        assert_eq!(ExpertFetcher::extent((0, 0, 0), &source).unwrap(), (0, 100));
        assert_eq!(
            ExpertFetcher::extent((0, 0, 3), &source).unwrap(),
            (300, 100)
        );
        assert!(ExpertFetcher::extent((0, 0, 4), &source).is_err());
    }
}
