//! Fixed-size device pool for streamed MoE expert weights.
//!
//! Giant MoE models (GLM-5.2 class: hundreds of GB of routed experts) cannot
//! hold every expert resident. The dense trunk loads as usual; routed experts
//! stay in the (split) GGUF on disk and are paged into this pool on demand:
//! one fixed device arena divided into equal slots, an LRU over resident
//! experts, and per-(layer, projection) host mirrors of the device pointer
//! tables the grouped MoE GEMV kernels consume. Expert slices are contiguous
//! in the GGUF (rank-3 tensors, expert-major), so a miss is one disk read +
//! one host-to-device copy into the victim slot — no byte transformation
//! (quantized matrix normalization is validation-only).
//!
//! Between the device pool and the disk sits the bounded pinned-host RAM
//! tier ([`ram_tier`]): device misses first check a budgeted, LRU'd pinned
//! arena (hits upload as a true async DMA with zero disk I/O), disk reads go
//! through an madvise-disciplined mmap path or an optional O_DIRECT twin-fd
//! path, uploads overlap through a non-blocking copy stream with
//! double-buffered pinned staging (the dsv4 CopyEngine pattern), and
//! per-expert selection frequencies persist to `<model_dir>/.hi_expert_usage`
//! so the hottest experts pre-warm into the tier at startup. None of it
//! changes router semantics or math: cache pressure affects speed, never
//! output. Env knobs are documented on [`ram_tier::TierEnvConfig`].

#[path = "expert_ram_tier.rs"]
pub(crate) mod ram_tier;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{GgufFile, GgufMemoryAdvice, GgufTensorType};

use crate::runtime::{DeviceBuffer, Stream};

/// Identifies one routed expert's weights: (layer, projection, expert).
/// Projection: 0 = gate, 1 = up, 2 = down.
pub(crate) type ExpertKey = (u32, u8, u32);

/// Concurrent disk readers for miss batches (page faults / preads are the
/// expensive part; parallel reads fill the NVMe queue).
const READ_WORKERS: usize = 6;

/// Pre-warm reads are chunked so the transient pageable staging stays small.
const PREWARM_CHUNK: usize = 64;

/// Where one (layer, projection) group of experts lives on disk.
#[derive(Debug, Clone)]
pub(crate) struct ExpertSource {
    /// Rank-3 tensor name in the GGUF (e.g. `blk.7.ffn_gate_exps.weight`).
    pub tensor_name: String,
    pub bytes_per_expert: usize,
    pub rows: usize,
    pub cols: usize,
    pub dtype: GgufTensorType,
    pub expert_count: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ExpertPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Bytes fetched from DISK (misses served by the RAM tier cost no I/O).
    pub bytes_read: u64,
}

struct Slot {
    key: Option<ExpertKey>,
    last_use: u64,
    /// Guards a slot from eviction while the current ensure pass also needs it.
    pinned_pass: u64,
}

pub(crate) struct ExpertPool {
    // The re-opened model file(s) live behind an Arc shared by the fetcher
    // and the WILLNEED thread, keeping the mmaps alive independent of the
    // caller's `GgufFile` borrow during model construction.
    arena: DeviceBuffer,
    slot_bytes: usize,
    slots: Vec<Slot>,
    resident: HashMap<ExpertKey, usize>,
    tick: u64,
    pass: u64,
    stats: ExpertPoolStats,
    /// Bounded pinned-host cache between the device pool and disk.
    tier: Option<ram_tier::RamTier>,
    /// Async copy stream + pinned staging; None = legacy synchronous uploads
    /// (`HI_CUDA_EXPERT_SYNC_UPLOAD=1` or engine construction failure).
    engine: Option<ram_tier::UploadEngine>,
    fetcher: ram_tier::ExpertFetcher,
    usage: Option<ram_tier::ExpertUsage>,
    willneed: Option<ram_tier::WillNeedThread>,
    /// Cumulative wall time of disk-read phases, for the MB/s health stat.
    read_nanos: u64,
}

impl ExpertPool {
    pub(crate) fn new(
        model_path: &std::path::Path,
        slot_bytes: usize,
        budget_bytes: usize,
    ) -> Result<Self> {
        Self::new_with_env(
            model_path,
            slot_bytes,
            budget_bytes,
            ram_tier::TierEnvConfig::from_env(),
        )
    }

    /// [`ExpertPool::new`] with the tier knobs passed explicitly (tests use
    /// this to avoid process-global env mutation).
    fn new_with_env(
        model_path: &std::path::Path,
        slot_bytes: usize,
        budget_bytes: usize,
        env: ram_tier::TierEnvConfig,
    ) -> Result<Self> {
        if slot_bytes == 0 {
            bail!("expert pool slot size must be non-zero");
        }
        let slot_count = budget_bytes / slot_bytes;
        if slot_count < 2 {
            bail!(
                "expert pool budget {budget_bytes} bytes holds fewer than 2 slots of {slot_bytes} bytes; raise HI_CUDA_EXPERT_POOL_BYTES"
            );
        }
        let arena_bytes = slot_count
            .checked_mul(slot_bytes)
            .context("expert pool arena byte count overflows usize")?;
        let arena = DeviceBuffer::alloc(arena_bytes).context("allocating expert pool arena")?;
        let gguf = Arc::new(GgufFile::open(model_path).with_context(|| {
            format!("re-opening {} for expert streaming", model_path.display())
        })?);

        // The streamable sources as this pool's own GGUF resolves them (same
        // alias helpers as the GPU loader), for madvise extents, readahead and
        // pre-warm. Best-effort: empty just skips those features.
        let sources = ram_tier::discover_sources(&gguf);
        let total_expert_bytes = ram_tier::total_expert_bytes(&sources);

        // Buffered expert faults should not drag neighboring experts into the
        // page cache: mark every expert tensor extent random-access, and rely
        // on explicit WILLNEED for exact readahead. Trunk tensors keep the
        // default readahead. Irrelevant under O_DIRECT (no page cache at all).
        if env.madvise_random && !env.odirect {
            for source in sources.values() {
                let _ = gguf.advise_tensor(&source.tensor_name, GgufMemoryAdvice::Random);
            }
        }

        let engine = if env.sync_upload {
            None
        } else {
            match ram_tier::UploadEngine::create(slot_bytes) {
                Ok(engine) => Some(engine),
                Err(err) => {
                    eprintln!(
                        "hi-cuda expert streaming: async upload engine unavailable ({err:#}); falling back to synchronous uploads"
                    );
                    None
                }
            }
        };
        let staging_bytes = if engine.is_some() {
            ram_tier::UploadEngine::staging_bytes(slot_bytes)
        } else {
            0
        };

        let plan = ram_tier::plan_tier_budget(
            env.explicit_ram_gb,
            ram_tier::mem_available_bytes(),
            slot_bytes,
            total_expert_bytes,
            staging_bytes,
        );
        eprintln!("hi-cuda expert RAM tier: {}", plan.describe());
        let tier = if plan.enabled() {
            let arena = if engine.is_some() {
                match crate::runtime::PinnedBuffer::alloc(plan.budget_bytes as usize) {
                    Ok(pinned) => Some(ram_tier::TierArena::Pinned(pinned)),
                    Err(err) => {
                        eprintln!(
                            "hi-cuda expert RAM tier: pinned allocation of {} bytes failed ({err:#}); using pageable memory (uploads will stage)",
                            plan.budget_bytes
                        );
                        None
                    }
                }
            } else {
                None
            };
            let arena = arena.unwrap_or_else(|| {
                ram_tier::TierArena::Heap(vec![0u8; plan.budget_bytes as usize])
            });
            Some(ram_tier::RamTier::new(arena, slot_bytes, plan.slots))
        } else {
            None
        };

        let direct = if env.odirect {
            match gguf.direct_io_reader() {
                Ok(reader) => Some(reader),
                Err(err) => {
                    eprintln!(
                        "hi-cuda expert streaming: HI_CUDA_EXPERT_ODIRECT=1 but O_DIRECT is unavailable ({err:#}); using buffered mmap reads"
                    );
                    None
                }
            }
        } else {
            None
        };
        let odirect = direct.is_some();
        let fetcher =
            ram_tier::ExpertFetcher::new(Arc::clone(&gguf), direct, env.willneed && !odirect);

        let usage = (env.usage_save_secs > 0)
            .then(|| ram_tier::ExpertUsage::load_or_new(model_path, env.usage_save_secs));
        let willneed =
            (env.willneed && !odirect).then(|| ram_tier::WillNeedThread::spawn(Arc::clone(&gguf)));

        eprintln!(
            "hi-cuda expert streaming: io={} upload={} madvise_random={} willneed={} usage={} prewarm_frac={:.2}",
            if odirect { "odirect" } else { "mmap" },
            if engine.is_some() { "async" } else { "sync" },
            env.madvise_random && !odirect,
            env.willneed && !odirect,
            usage.is_some(),
            env.prewarm_frac,
        );

        let mut pool = Self {
            arena,
            slot_bytes,
            slots: (0..slot_count)
                .map(|_| Slot {
                    key: None,
                    last_use: 0,
                    pinned_pass: 0,
                })
                .collect(),
            resident: HashMap::new(),
            tick: 0,
            pass: 0,
            stats: ExpertPoolStats::default(),
            tier,
            engine,
            fetcher,
            usage,
            willneed,
            read_nanos: 0,
        };
        pool.prewarm_from_usage(&sources, env.prewarm_frac);
        Ok(pool)
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn resident_count(&self) -> usize {
        self.resident.len()
    }

    pub(crate) fn stats(&self) -> ExpertPoolStats {
        self.stats
    }

    /// The `/health` expert-streaming segment: device-pool counters plus the
    /// RAM-tier addition (hits/misses/evictions/pinned/budget/disk MB/s).
    pub(crate) fn stats_segment(&self) -> String {
        let stats = self.stats;
        let mut out = format!(
            "pool(hits={},misses={},evictions={},read_mb={})",
            stats.hits,
            stats.misses,
            stats.evictions,
            stats.bytes_read / (1024 * 1024)
        );
        match &self.tier {
            Some(tier) => {
                let tier_stats = tier.stats();
                let read_secs = self.read_nanos as f64 / 1e9;
                let disk_mbps = if read_secs > 0.0 {
                    stats.bytes_read as f64 / (1024.0 * 1024.0) / read_secs
                } else {
                    0.0
                };
                out.push_str(&format!(
                    "; ram(hits={},misses={},evictions={},served_mb={},cached={},pinned_mb={},sticky={},prewarmed={},budget_mb={},io={},disk_mbps={disk_mbps:.0})",
                    tier_stats.hits,
                    tier_stats.misses,
                    tier_stats.evictions,
                    tier_stats.bytes_served / (1024 * 1024),
                    tier.resident_count(),
                    tier.resident_bytes() / (1024 * 1024),
                    tier.sticky_count(),
                    tier_stats.prewarmed,
                    tier.budget_bytes() / (1024 * 1024),
                    if self.fetcher.is_odirect() {
                        "odirect"
                    } else {
                        "mmap"
                    },
                ));
            }
            None => out.push_str("; ram(off)"),
        }
        out
    }

    fn slot_device_addr(&self, slot: usize) -> u64 {
        self.arena.as_ptr() as u64 + (slot * self.slot_bytes) as u64
    }

    /// Pre-warm the hottest experts (by persisted selection frequency) into
    /// the RAM tier as sticky entries, up to `frac` of the tier's slots.
    /// Profile-ranked placement beats heat-blind placement ~3x at equal
    /// capacity (colibri: 0.94 vs 0.29 tok/s).
    fn prewarm_from_usage(&mut self, sources: &BTreeMap<(u32, u8), ExpertSource>, frac: f64) {
        let Some(tier) = &self.tier else { return };
        if frac <= 0.0 || sources.is_empty() {
            return;
        }
        let Some(usage) = &self.usage else { return };
        if usage.is_empty() {
            return;
        }
        let max_slots = (tier.slot_count() as f64 * frac) as usize;
        let mut jobs: Vec<(ExpertKey, &ExpertSource)> = Vec::new();
        'rank: for (layer, expert) in usage.ranked() {
            for proj in 0..3u8 {
                if jobs.len() >= max_slots {
                    break 'rank;
                }
                if let Some(source) = sources.get(&(layer, proj))
                    && (expert as usize) < source.expert_count
                    && source.bytes_per_expert <= self.slot_bytes
                {
                    jobs.push(((layer, proj, expert), source));
                }
            }
        }
        if jobs.is_empty() {
            return;
        }
        let started = Instant::now();
        let mut warmed = 0usize;
        let mut warmed_bytes = 0u64;
        for chunk in jobs.chunks(PREWARM_CHUNK) {
            let fetched = parallel_fetch(&self.fetcher, chunk);
            let tier = self.tier.as_mut().expect("tier checked above");
            for ((key, _), bytes) in chunk.iter().zip(fetched) {
                let Ok(bytes) = bytes else { continue };
                if tier.insert(*key, &bytes, true, 0).is_some() {
                    warmed += 1;
                    warmed_bytes += bytes.len() as u64;
                }
            }
        }
        self.read_nanos += started.elapsed().as_nanos() as u64;
        self.stats.bytes_read += warmed_bytes;
        eprintln!(
            "hi-cuda expert RAM tier: pre-warmed {warmed} expert slices ({} MiB) from {} usage entries in {:.1}s",
            warmed_bytes / (1024 * 1024),
            self.usage.as_ref().map(|usage| usage.len()).unwrap_or(0),
            started.elapsed().as_secs_f64(),
        );
    }

    /// Make every (key, source) pair resident, returning each expert's device
    /// address in order. One "pass" pins all requested experts so a miss never
    /// evicts another expert needed by the same forward step. Misses are
    /// served from the pinned RAM tier when cached (async DMA, zero disk I/O)
    /// and otherwise read from disk CONCURRENTLY (up to [`READ_WORKERS`]
    /// threads fill the NVMe queue), then uploaded through the non-blocking
    /// copy stream. Legacy synchronous behavior is preserved via
    /// [`ExpertPool::ensure_resident`] semantics: on return every requested
    /// expert's bytes are (or are ordered to be) on device.
    pub(crate) fn ensure_resident(
        &mut self,
        requests: &[(ExpertKey, &ExpertSource)],
    ) -> Result<Vec<u64>> {
        self.ensure_resident_on(requests, None)
    }

    /// [`ExpertPool::ensure_resident`] with an optional engine stream.
    ///
    /// * `Some(stream)`: uploads are event-ordered against `stream` (copies
    ///   wait for already-enqueued GEMVs via a fork event; subsequent GEMVs
    ///   wait for the copies via a done event). The host never blocks on the
    ///   copies — full phase-3b overlap.
    /// * `None`: the host waits for the copy stream before returning, so the
    ///   call behaves like the original synchronous implementation. The
    ///   caller must guarantee the device is idle with respect to pool slots
    ///   on entry (the qwen streaming path is: it downloads the routed ids
    ///   with a null-stream D2H immediately before, which drains the engine
    ///   stream).
    pub(crate) fn ensure_resident_on(
        &mut self,
        requests: &[(ExpertKey, &ExpertSource)],
        engine_stream: Option<&Stream>,
    ) -> Result<Vec<u64>> {
        self.pass += 1;
        let pass = self.pass;
        // The learning cache: count each distinct (layer, expert) selection.
        if let Some(usage) = &mut self.usage {
            let mut selected: Vec<(u32, u32)> =
                requests.iter().map(|(key, _)| (key.0, key.2)).collect();
            selected.sort_unstable();
            selected.dedup();
            usage.record_pass(selected);
            usage.maybe_save();
        }
        // First pass: mark hits so they cannot be evicted by this pass's
        // misses, and bump tier recency so the tier tracks routing heat even
        // for device-resident experts.
        for (key, _) in requests {
            if let Some(&slot) = self.resident.get(key) {
                self.tick += 1;
                self.slots[slot].last_use = self.tick;
                self.slots[slot].pinned_pass = pass;
                if let Some(tier) = &mut self.tier {
                    tier.touch(key, pass);
                }
            }
        }
        // Assign slots for the misses (serial: eviction order stays
        // deterministic), dedup within the request list.
        let mut addrs = vec![0u64; requests.len()];
        let mut misses: Vec<(ExpertKey, &ExpertSource, usize)> = Vec::new();
        for (idx, (key, source)) in requests.iter().enumerate() {
            if let Some(&slot) = self.resident.get(key) {
                self.stats.hits += 1;
                addrs[idx] = self.slot_device_addr(slot);
                continue;
            }
            let slot = self.take_slot(pass)?;
            self.tick += 1;
            self.slots[slot].key = Some(*key);
            self.slots[slot].last_use = self.tick;
            self.slots[slot].pinned_pass = pass;
            self.resident.insert(*key, slot);
            addrs[idx] = self.slot_device_addr(slot);
            misses.push((*key, source, slot));
        }
        if misses.is_empty() {
            return Ok(addrs);
        }
        for (key, source, _) in &misses {
            if source.bytes_per_expert > self.slot_bytes {
                bail!(
                    "expert {} of {} needs {} bytes; pool slots are {} bytes",
                    key.2,
                    source.tensor_name,
                    source.bytes_per_expert,
                    self.slot_bytes
                );
            }
        }
        // RAM-tier lookups: hits skip disk entirely.
        let tier_hits: Vec<Option<(usize, usize)>> = misses
            .iter()
            .map(|(key, _, _)| self.tier.as_mut().and_then(|tier| tier.lookup(key, pass)))
            .collect();
        // Batched WILLNEED for the extents about to be read, from the
        // dedicated thread: the kernel readahead overlaps the earlier reads.
        let disk_jobs: Vec<(ExpertKey, &ExpertSource)> = misses
            .iter()
            .zip(&tier_hits)
            .filter(|(_, hit)| hit.is_none())
            .map(|((key, source, _), _)| (*key, *source))
            .collect();
        if let Some(willneed) = &self.willneed {
            let extents = disk_jobs
                .iter()
                .filter_map(|(key, source)| {
                    ram_tier::ExpertFetcher::extent(*key, source)
                        .ok()
                        .map(|(offset, len)| (source.tensor_name.clone(), offset, len))
                })
                .collect();
            willneed.hint(extents);
        }
        // Concurrent disk reads for the tier misses.
        let fetched = if disk_jobs.is_empty() {
            Vec::new()
        } else {
            let started = Instant::now();
            let fetched = parallel_fetch(&self.fetcher, &disk_jobs);
            self.read_nanos += started.elapsed().as_nanos() as u64;
            fetched
        };
        // Upload phase.
        let mut fetched = fetched.into_iter();
        match &mut self.engine {
            Some(engine) => {
                engine.begin_pass(engine_stream)?;
                for ((key, source, slot), tier_hit) in misses.iter().zip(&tier_hits) {
                    let dst_offset = slot * self.slot_bytes;
                    match tier_hit {
                        Some((tier_offset, len)) => {
                            let tier = self.tier.as_ref().expect("tier hit without a tier");
                            match tier.arena().pinned() {
                                Some(pinned) => engine.upload_pinned(
                                    &self.arena,
                                    dst_offset,
                                    pinned,
                                    *tier_offset,
                                    *len,
                                )?,
                                None => {
                                    let bytes = tier
                                        .arena()
                                        .heap_slice(*tier_offset, *len)
                                        .ok_or_else(|| anyhow!("tier hit out of arena bounds"))?;
                                    engine.stage_upload(&self.arena, dst_offset, bytes)?;
                                }
                            }
                        }
                        None => {
                            let bytes = fetched.next().expect("disk read job completed")?;
                            let cached = self.tier.as_mut().and_then(|tier| {
                                tier.insert(*key, &bytes, false, pass)
                                    .map(|offset| (offset, tier.arena().is_pinned()))
                            });
                            match cached {
                                Some((tier_offset, true)) => {
                                    let pinned = self
                                        .tier
                                        .as_ref()
                                        .and_then(|tier| tier.arena().pinned())
                                        .expect("pinned tier arena");
                                    engine.upload_pinned(
                                        &self.arena,
                                        dst_offset,
                                        pinned,
                                        tier_offset,
                                        bytes.len(),
                                    )?;
                                }
                                _ => engine.stage_upload(&self.arena, dst_offset, &bytes)?,
                            }
                            self.stats.bytes_read += source.bytes_per_expert as u64;
                        }
                    }
                    self.stats.misses += 1;
                }
                engine.finish_pass(engine_stream)?;
            }
            None => {
                // Legacy synchronous uploads (bisection escape hatch). The
                // tier, when present, holds pageable memory here.
                for ((key, source, slot), tier_hit) in misses.iter().zip(&tier_hits) {
                    let dst_offset = slot * self.slot_bytes;
                    match tier_hit {
                        Some((tier_offset, len)) => {
                            let bytes = self
                                .tier
                                .as_ref()
                                .and_then(|tier| tier.arena().heap_slice(*tier_offset, *len))
                                .ok_or_else(|| {
                                    anyhow!("pinned RAM tier requires the async upload engine")
                                })?;
                            self.arena.copy_from_host_at(dst_offset, bytes)?;
                        }
                        None => {
                            let bytes = fetched.next().expect("disk read job completed")?;
                            self.arena.copy_from_host_at(dst_offset, &bytes)?;
                            if let Some(tier) = &mut self.tier {
                                tier.insert(*key, &bytes, false, pass);
                            }
                            self.stats.bytes_read += source.bytes_per_expert as u64;
                        }
                    }
                    self.stats.misses += 1;
                }
            }
        }
        Ok(addrs)
    }

    fn take_slot(&mut self, pass: u64) -> Result<usize> {
        // Prefer a free slot; else evict the least-recently-used unpinned one.
        if let Some(free) = self.slots.iter().position(|slot| slot.key.is_none()) {
            return Ok(free);
        }
        let victim = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.pinned_pass != pass)
            .min_by_key(|(_, slot)| slot.last_use)
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                anyhow!(
                    "expert pool has no evictable slot: every resident expert is required by the current step; raise HI_CUDA_EXPERT_POOL_BYTES"
                )
            })?;
        if let Some(old_key) = self.slots[victim].key.take() {
            self.resident.remove(&old_key);
            self.stats.evictions += 1;
        }
        Ok(victim)
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        // A GEMV-ordered upload may still be in flight when the provider
        // tears down: wait before the arena and staging are freed.
        if let Some(engine) = &mut self.engine {
            engine.drain();
        }
        if let Some(usage) = &mut self.usage {
            usage.save_if_dirty();
        }
        if self.stats.hits + self.stats.misses > 0 {
            eprintln!("hi-cuda expert pool: {}", self.stats_segment());
        }
    }
}

/// Fetch a batch of expert byte extents with up to [`READ_WORKERS`] threads
/// (results in job order). Used by miss batches and pre-warm.
fn parallel_fetch(
    fetcher: &ram_tier::ExpertFetcher,
    jobs: &[(ExpertKey, &ExpertSource)],
) -> Vec<Result<Vec<u8>>> {
    let workers = jobs.len().min(READ_WORKERS);
    if workers <= 1 {
        return jobs
            .iter()
            .map(|(key, source)| fetcher.fetch(*key, source))
            .collect();
    }
    let queue = std::sync::Mutex::new(jobs.iter().enumerate());
    let results = std::sync::Mutex::new((0..jobs.len()).map(|_| None).collect::<Vec<_>>());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let job = { queue.lock().unwrap().next() };
                    let Some((idx, (key, source))) = job else {
                        break;
                    };
                    let bytes = fetcher.fetch(*key, source);
                    results.lock().unwrap()[idx] = Some(bytes);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|entry| entry.expect("expert read job completed"))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    const FIXTURE_LAYERS: u32 = 2;
    const FIXTURE_EXPERTS: u32 = 4;
    const FIXTURE_IN: usize = 8;
    const FIXTURE_OUT: usize = 8;
    /// F16 rank-3 experts: in * out * 2 bytes per expert.
    const FIXTURE_EXPERT_BYTES: usize = FIXTURE_IN * FIXTURE_OUT * 2;

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        write_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        write_string(bytes, key);
        write_u32(bytes, 8);
        write_string(bytes, value);
    }

    fn write_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        write_string(bytes, key);
        write_u32(bytes, 4);
        write_u32(bytes, value);
    }

    struct FixtureTensor {
        name: String,
        dims: Vec<u64>,
        dtype: u32,
        bytes: Vec<u8>,
        offset: u64,
    }

    /// Deterministic per-tensor byte pattern: tests compare device / tier /
    /// O_DIRECT bytes against these exact values.
    fn pattern_bytes(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (seed as usize).wrapping_add(i.wrapping_mul(31)) as u8)
            .collect()
    }

    /// Minimal qwen3moe GGUF whose routed experts are patterned rank-3 F16
    /// tensors: exactly the layout the streaming pool serves.
    fn write_streaming_fixture(path: &Path) {
        let mut tensors = Vec::new();
        for layer in 0..FIXTURE_LAYERS {
            tensors.push(FixtureTensor {
                name: format!("blk.{layer}.ffn_gate_inp.weight"),
                dims: vec![FIXTURE_IN as u64, FIXTURE_EXPERTS as u64],
                dtype: 0, // f32
                bytes: vec![0u8; FIXTURE_IN * FIXTURE_EXPERTS as usize * 4],
                offset: 0,
            });
            for (proj_seed, proj) in ["gate", "up", "down"].iter().enumerate() {
                tensors.push(FixtureTensor {
                    name: format!("blk.{layer}.ffn_{proj}_exps.weight"),
                    dims: vec![
                        FIXTURE_IN as u64,
                        FIXTURE_OUT as u64,
                        FIXTURE_EXPERTS as u64,
                    ],
                    dtype: 1, // f16 (bytes are opaque to the pool)
                    bytes: pattern_bytes(
                        FIXTURE_EXPERT_BYTES * FIXTURE_EXPERTS as usize,
                        (layer * 40 + proj_seed as u32 * 11 + 3) as u8,
                    ),
                    offset: 0,
                });
            }
        }
        let mut data = Vec::new();
        for tensor in &mut tensors {
            while data.len() % 32 != 0 {
                data.push(0);
            }
            tensor.offset = data.len() as u64;
            data.extend_from_slice(&tensor.bytes);
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensors.len() as u64);
        write_u64(&mut bytes, 11);
        write_kv_string(&mut bytes, "general.architecture", "qwen3moe");
        write_kv_string(&mut bytes, "general.name", "expert-pool-fixture");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "qwen3moe.context_length", 64);
        write_kv_u32(&mut bytes, "qwen3moe.embedding_length", FIXTURE_IN as u32);
        write_kv_u32(
            &mut bytes,
            "qwen3moe.feed_forward_length",
            FIXTURE_OUT as u32,
        );
        write_kv_u32(&mut bytes, "qwen3moe.block_count", FIXTURE_LAYERS);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "qwen3moe.expert_count", FIXTURE_EXPERTS);
        write_kv_u32(&mut bytes, "qwen3moe.expert_used_count", 2);
        for tensor in &tensors {
            write_string(&mut bytes, &tensor.name);
            write_u32(&mut bytes, tensor.dims.len() as u32);
            for &dim in &tensor.dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor.dtype);
            write_u64(&mut bytes, tensor.offset);
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend(data);
        std::fs::write(path, bytes).unwrap();
    }

    /// Unique model dir per test: the usage file lives next to the model, so
    /// sharing a directory would leak learning-cache state between tests.
    fn fixture_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-expert-pool-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_env(ram_bytes: u64, prewarm_frac: f64, sync_upload: bool) -> ram_tier::TierEnvConfig {
        ram_tier::TierEnvConfig {
            explicit_ram_gb: Some(ram_bytes as f64 / ram_tier::GIB as f64),
            prewarm_frac,
            odirect: false,
            willneed: true,
            madvise_random: true,
            sync_upload,
            usage_save_secs: 3600,
        }
    }

    fn expected_expert_bytes(gguf: &GgufFile, key: ExpertKey) -> Vec<u8> {
        let proj = ["gate", "up", "down"][key.1 as usize];
        let name = format!("blk.{}.ffn_{proj}_exps.weight", key.0);
        let view = gguf.tensor(&name).unwrap();
        view.bytes
            [key.2 as usize * FIXTURE_EXPERT_BYTES..(key.2 as usize + 1) * FIXTURE_EXPERT_BYTES]
            .to_vec()
    }

    /// Every (layer, projection, expert) of the fixture as an ensure request.
    fn all_requests(
        sources: &BTreeMap<(u32, u8), ExpertSource>,
    ) -> Vec<(ExpertKey, &ExpertSource)> {
        sources
            .iter()
            .flat_map(|(&(layer, proj), source)| {
                (0..FIXTURE_EXPERTS).map(move |expert| ((layer, proj, expert), source))
            })
            .collect()
    }

    #[test]
    fn discover_sources_matches_streaming_layout() {
        let dir = fixture_dir("discover");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();

        let sources = ram_tier::discover_sources(&gguf);
        assert_eq!(sources.len(), (FIXTURE_LAYERS * 3) as usize);
        for layer in 0..FIXTURE_LAYERS {
            for (proj, name) in ["gate", "up", "down"].iter().enumerate() {
                let source = sources.get(&(layer, proj as u8)).unwrap();
                assert_eq!(
                    source.tensor_name,
                    format!("blk.{layer}.ffn_{name}_exps.weight")
                );
                assert_eq!(source.bytes_per_expert, FIXTURE_EXPERT_BYTES);
                assert_eq!(source.rows, FIXTURE_OUT);
                assert_eq!(source.cols, FIXTURE_IN);
                assert_eq!(source.expert_count, FIXTURE_EXPERTS as usize);
            }
        }
        assert_eq!(
            ram_tier::total_expert_bytes(&sources),
            (FIXTURE_LAYERS * 3 * FIXTURE_EXPERTS) as u64 * FIXTURE_EXPERT_BYTES as u64
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The mmap and O_DIRECT fetch paths must return identical bytes for
    /// every expert extent (skips the O_DIRECT half where unsupported).
    #[test]
    fn fetcher_mmap_and_odirect_agree_on_fixture() {
        let dir = fixture_dir("fetch");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = Arc::new(GgufFile::open(&model).unwrap());
        let sources = ram_tier::discover_sources(&gguf);

        let mmap_fetcher = ram_tier::ExpertFetcher::new(Arc::clone(&gguf), None, true);
        let direct = gguf.direct_io_reader().ok();
        if direct.is_none() {
            eprintln!("skipping O_DIRECT half: unsupported filesystem");
        }
        let odirect_fetcher = direct
            .map(|reader| ram_tier::ExpertFetcher::new(Arc::clone(&gguf), Some(reader), false));
        for (key, source) in all_requests(&sources) {
            let expected = expected_expert_bytes(&gguf, key);
            let via_mmap = mmap_fetcher.fetch(key, source).unwrap();
            assert_eq!(via_mmap, expected, "mmap fetch of {key:?}");
            if let Some(fetcher) = &odirect_fetcher {
                let via_direct = fetcher.fetch(key, source).unwrap();
                assert_eq!(via_direct, expected, "O_DIRECT fetch of {key:?}");
            }
        }
        // Out-of-range experts fail loudly.
        let (&(layer, proj), source) = sources.iter().next().unwrap();
        assert!(
            mmap_fetcher
                .fetch((layer, proj, FIXTURE_EXPERTS), source)
                .is_err()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Device bytes must be identical whether uploads go through the async
    /// copy stream (tier + staging) or the legacy synchronous path, and both
    /// must match the GGUF bytes — the staged-upload parity gate. Requires a
    /// CUDA device (runs in the native suite).
    #[test]
    fn native_cuda_async_and_sync_uploads_land_identical_device_bytes() {
        let dir = fixture_dir("parity");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);

        // Enough tier for everything; device pool big enough for one pass.
        let tier_bytes = (24 * FIXTURE_EXPERT_BYTES) as u64;
        let mut collected: Vec<Vec<Vec<u8>>> = Vec::new();
        for sync_upload in [false, true] {
            let mut pool = ExpertPool::new_with_env(
                &model,
                FIXTURE_EXPERT_BYTES,
                24 * FIXTURE_EXPERT_BYTES,
                test_env(tier_bytes, 0.0, sync_upload),
            )
            .unwrap();
            let requests: Vec<(ExpertKey, &ExpertSource)> = all_requests(&sources);
            let addrs = pool.ensure_resident(&requests).unwrap();
            let base = pool.arena.as_ptr() as u64;
            let mut device_bytes = Vec::new();
            for (idx, (key, _)) in requests.iter().enumerate() {
                let offset = usize::try_from(addrs[idx] - base).unwrap();
                let bytes = pool
                    .arena
                    .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                    .unwrap();
                assert_eq!(
                    bytes,
                    expected_expert_bytes(&gguf, *key),
                    "device bytes for {key:?} (sync_upload={sync_upload})"
                );
                device_bytes.push(bytes);
            }
            collected.push(device_bytes);
        }
        assert_eq!(collected[0], collected[1], "async vs sync upload parity");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// RAM-tier behavior end to end: repeat misses are served from the tier
    /// with ZERO additional disk bytes, device evictions stay correct, the
    /// engine-stream variant agrees, and the usage file persists at drop.
    /// Requires a CUDA device (runs in the native suite).
    #[test]
    fn native_cuda_tier_serves_repeat_misses_without_disk_and_persists_usage() {
        let dir = fixture_dir("tier");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);

        // Device pool: 3 slots (forced eviction across passes). Tier: all 24.
        let tier_bytes = (24 * FIXTURE_EXPERT_BYTES) as u64;
        let mut pool = ExpertPool::new_with_env(
            &model,
            FIXTURE_EXPERT_BYTES,
            3 * FIXTURE_EXPERT_BYTES,
            test_env(tier_bytes, 0.0, false),
        )
        .unwrap();
        let stream = Stream::create().unwrap();
        let source_of = |layer: u32, proj: u8| sources.get(&(layer, proj)).unwrap();

        // Pass 1: three misses on layer 0 gate -> disk.
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((0u32, 0u8, expert), source_of(0, 0)))
            .collect();
        pool.ensure_resident(&requests).unwrap();
        let disk_after_first = pool.stats().bytes_read;
        assert_eq!(disk_after_first, 3 * FIXTURE_EXPERT_BYTES as u64);

        // Pass 2: different experts evict the device slots (tier keeps all).
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((1u32, 2u8, expert), source_of(1, 2)))
            .collect();
        pool.ensure_resident_on(&requests, Some(&stream)).unwrap();
        stream.synchronize().unwrap();
        assert!(pool.stats().evictions >= 3);
        let disk_after_second = pool.stats().bytes_read;
        assert_eq!(
            disk_after_second,
            disk_after_first + 3 * FIXTURE_EXPERT_BYTES as u64,
            "pass 2 reads its own experts from disk"
        );

        // Pass 3: the original experts again -> device misses, tier hits, no
        // new disk bytes, and byte-exact device contents.
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((0u32, 0u8, expert), source_of(0, 0)))
            .collect();
        let addrs = pool.ensure_resident(&requests).unwrap();
        assert_eq!(
            pool.stats().bytes_read,
            disk_after_second,
            "tier hits cost no disk I/O"
        );
        let tier_stats = pool.tier.as_ref().unwrap().stats();
        assert_eq!(tier_stats.hits, 3);
        assert!(tier_stats.misses >= 6);
        let base = pool.arena.as_ptr() as u64;
        for (idx, (key, _)) in requests.iter().enumerate() {
            let offset = usize::try_from(addrs[idx] - base).unwrap();
            let bytes = pool
                .arena
                .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                .unwrap();
            assert_eq!(
                bytes,
                expected_expert_bytes(&gguf, *key),
                "tier-served {key:?}"
            );
        }
        let segment = pool.stats_segment();
        assert!(segment.contains("ram(hits=3"), "{segment}");

        // Dropping the pool persists the selection counters atomically.
        drop(pool);
        let usage_path = dir.join(".hi_expert_usage");
        let raw = std::fs::read_to_string(&usage_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["version"], 1);
        assert_eq!(parsed["model"], "model.gguf");
        assert_eq!(parsed["counts"]["0:0"], 2, "{raw}");
        assert_eq!(parsed["counts"]["1:1"], 1, "{raw}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The learning cache: a usage history pre-warms the hottest experts as
    /// sticky tier entries at construction, and routing to them afterwards
    /// costs zero disk reads. Requires a CUDA device (runs in the native
    /// suite).
    #[test]
    fn native_cuda_prewarm_pins_hottest_experts_from_usage_history() {
        let dir = fixture_dir("prewarm");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);

        // Seed a history: expert (0, 1) is hottest, then (1, 2).
        let mut usage = ram_tier::ExpertUsage::load_or_new(&model, 3600);
        for _ in 0..5 {
            usage.record_pass([(0u32, 1u32)]);
        }
        usage.record_pass([(1u32, 2u32)]);
        usage.save().unwrap();

        let tier_bytes = (24 * FIXTURE_EXPERT_BYTES) as u64;
        let mut pool = ExpertPool::new_with_env(
            &model,
            FIXTURE_EXPERT_BYTES,
            4 * FIXTURE_EXPERT_BYTES,
            test_env(tier_bytes, 1.0, false),
        )
        .unwrap();
        let tier_stats = pool.tier.as_ref().unwrap().stats();
        assert_eq!(tier_stats.prewarmed, 6, "2 hot experts x 3 projections");
        assert_eq!(pool.tier.as_ref().unwrap().sticky_count(), 6);
        let disk_after_prewarm = pool.stats().bytes_read;
        assert_eq!(disk_after_prewarm, 6 * FIXTURE_EXPERT_BYTES as u64);

        // Routing to the pre-warmed expert: all three projections hit the
        // tier; no disk bytes move.
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3u8)
            .map(|proj| ((0u32, proj, 1u32), sources.get(&(0, proj)).unwrap()))
            .collect();
        let addrs = pool.ensure_resident(&requests).unwrap();
        assert_eq!(pool.stats().bytes_read, disk_after_prewarm);
        assert_eq!(pool.tier.as_ref().unwrap().stats().hits, 3);
        let base = pool.arena.as_ptr() as u64;
        for (idx, (key, _)) in requests.iter().enumerate() {
            let offset = usize::try_from(addrs[idx] - base).unwrap();
            let bytes = pool
                .arena
                .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                .unwrap();
            assert_eq!(
                bytes,
                expected_expert_bytes(&gguf, *key),
                "pre-warmed {key:?}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
