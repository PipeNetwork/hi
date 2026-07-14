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
    /// Temporal-prefetch outcomes (`HI_CUDA_EXPERT_PREFETCH`): ensure hits on
    /// a slot whose upload was issued speculatively, and prefetched slots
    /// evicted before any ensure ever touched them.
    pub prefetch_hits: u64,
    pub prefetch_wasted: u64,
}

struct Slot {
    key: Option<ExpertKey>,
    last_use: u64,
    /// Guards a slot from eviction while the current ensure pass also needs it.
    pinned_pass: u64,
    /// Filled by a speculative prefetch and not yet hit by an ensure pass
    /// (drives the prefetch_hits / prefetch_wasted counters).
    prefetched: bool,
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
    /// io_uring reads may DMA straight into pinned tier slots: requires the
    /// ring backend, a pinned tier arena with a 4 KiB-aligned base, and the
    /// async upload engine. Decided once at construction.
    ring_slot_dma: bool,
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

        let mem_available = ram_tier::mem_available_bytes();

        // Ring selection is tri-state (`HI_CUDA_EXPERT_IOURING`): `1` forces
        // the ring, `0` forces it off, unset = AUTO — ring when the
        // streamable experts cannot be comfortably page-cache-resident and
        // are not already warm (see `load_source::auto_for_expert_stream`).
        // Then the ladder: io_uring -> O_DIRECT threads -> mmap. The ring is
        // probed with one real read at construction; any failure (kernel
        // <5.6, io_uring_disabled sysctl, seccomp/container denial,
        // O_DIRECT-less filesystem) logs and falls through — a knob can never
        // fail the model load.
        #[cfg(target_os = "linux")]
        let paths: Vec<std::path::PathBuf> = (0..gguf.shard_count())
            .filter_map(|shard| gguf.shard_path(shard).map(std::path::Path::to_path_buf))
            .collect();
        #[cfg(target_os = "linux")]
        let ring_mode: Option<&'static str> = match env.iouring {
            Some(true) => Some("forced"),
            Some(false) => None,
            None if sources.is_empty() => None,
            None => {
                let sample = sample_expert_extents(&gguf, &sources);
                let inputs = crate::load_source::AutoInputs {
                    needed_bytes: total_expert_bytes,
                    mem_available,
                    residency: crate::expert_uring::sampled_extent_residency(&paths, &sample, 64),
                };
                let decision = crate::load_source::auto_for_expert_stream(&inputs);
                eprintln!(
                    "hi-cuda expert streaming: io_uring auto -> {} ({})",
                    if decision.use_ring { "on" } else { "off" },
                    decision.why
                );
                decision.use_ring.then_some("auto")
            }
        };
        #[cfg(not(target_os = "linux"))]
        let ring_mode: Option<&'static str> = {
            if env.iouring == Some(true) {
                eprintln!(
                    "hi-cuda expert streaming: HI_CUDA_EXPERT_IOURING=1 is Linux-only; using the fallback ladder"
                );
            }
            None
        };
        #[cfg(target_os = "linux")]
        let mut uring = if ring_mode.is_some() {
            match crate::expert_uring::IoUringReader::open(&paths, env.iouring_qd) {
                Ok(reader) => {
                    for note in reader.notes() {
                        eprintln!("hi-cuda expert streaming: io_uring {note}");
                    }
                    Some(reader)
                }
                Err(err) => {
                    eprintln!(
                        "hi-cuda expert streaming: io_uring is unavailable ({err:#}); falling back to O_DIRECT thread reads"
                    );
                    None
                }
            }
        } else {
            None
        };
        #[cfg(target_os = "linux")]
        let ring_active = uring.is_some();
        #[cfg(not(target_os = "linux"))]
        let ring_active = false;

        let direct = if !ring_active && (env.odirect || ring_mode.is_some()) {
            match gguf.direct_io_reader() {
                Ok(reader) => Some(reader),
                Err(err) => {
                    eprintln!(
                        "hi-cuda expert streaming: O_DIRECT is unavailable ({err:#}); using buffered mmap reads"
                    );
                    None
                }
            }
        } else {
            None
        };
        let odirect = direct.is_some();
        let mmap_reads = !ring_active && !odirect;

        // Buffered expert faults should not drag neighboring experts into the
        // page cache: mark every expert tensor extent random-access, and rely
        // on explicit WILLNEED for exact readahead. Trunk tensors keep the
        // default readahead. Irrelevant under O_DIRECT/io_uring (no page
        // cache at all).
        if env.madvise_random && mmap_reads {
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

        // In ring mode every tier slot doubles as an O_DIRECT DMA destination:
        // widen the stride so slot bases stay 4 KiB-aligned with room for the
        // block-aligned span of a full payload (small waste, zero CPU copies).
        #[cfg(target_os = "linux")]
        let tier_stride = if ring_active {
            crate::expert_uring::tier_slot_stride(slot_bytes)
        } else {
            slot_bytes
        };
        #[cfg(not(target_os = "linux"))]
        let tier_stride = slot_bytes;

        // With a widened ring stride the arena needs `slices x stride` to
        // cache everything, not just the payload bytes on disk.
        let tier_cap_bytes = if tier_stride == slot_bytes {
            total_expert_bytes
        } else {
            ram_tier::total_expert_slices(&sources).saturating_mul(tier_stride as u64)
        };
        let plan = ram_tier::plan_tier_budget(
            env.explicit_ram_gb,
            mem_available,
            tier_stride,
            tier_cap_bytes,
            staging_bytes,
        );
        eprintln!("hi-cuda expert RAM tier: {}", plan.describe());
        let tier = if plan.enabled() {
            // Ring mode DMAs O_DIRECT reads straight into slots, so slot 0
            // must sit on a 4 KiB boundary: over-allocate by one block and
            // shift (cudaHostAlloc suballocates small buffers unaligned).
            let align_slack = if ring_active {
                ram_tier::SLOT_DMA_ALIGN
            } else {
                0
            };
            let arena = if engine.is_some() {
                match crate::runtime::PinnedBuffer::alloc(plan.budget_bytes as usize + align_slack)
                {
                    Ok(pinned) => {
                        let base = pinned.as_mut_ptr() as usize;
                        let base_offset = if ring_active {
                            base.next_multiple_of(ram_tier::SLOT_DMA_ALIGN) - base
                        } else {
                            0
                        };
                        Some((ram_tier::TierArena::Pinned(pinned), base_offset))
                    }
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
            let (arena, base_offset) = arena.unwrap_or_else(|| {
                (
                    ram_tier::TierArena::Heap(vec![0u8; plan.budget_bytes as usize]),
                    0,
                )
            });
            Some(ram_tier::RamTier::new_with_stride(
                arena,
                slot_bytes,
                tier_stride,
                plan.slots,
                base_offset,
            ))
        } else {
            None
        };

        // Ring + pinned tier: reads can DMA straight into the slots. Register
        // the slot region as fixed buffers while we are at it (best-effort:
        // most of the win is queue depth, not fixed buffers).
        #[cfg(target_os = "linux")]
        let ring_slot_dma = {
            let aligned_base = tier.as_ref().and_then(|tier| {
                tier.arena()
                    .pinned()
                    .map(|pinned| pinned.as_mut_ptr() as usize + tier.base_offset())
            });
            match (&mut uring, aligned_base) {
                (Some(reader), Some(base))
                    if base % crate::expert_uring::URING_BLOCK == 0 && engine.is_some() =>
                {
                    if let Err(err) = reader.register_arena(
                        base as *mut u8,
                        plan.budget_bytes as usize,
                        tier_stride,
                    ) {
                        eprintln!(
                            "hi-cuda expert streaming: io_uring buffer registration unavailable ({err:#}); unregistered reads on the same ring"
                        );
                    }
                    true
                }
                (Some(_), Some(_)) => {
                    eprintln!(
                        "hi-cuda expert streaming: ring slot DMA disabled ({}); ring reads stage through scratch",
                        if engine.is_none() {
                            "no async upload engine"
                        } else {
                            "pinned tier arena base did not align"
                        }
                    );
                    false
                }
                _ => false,
            }
        };
        #[cfg(not(target_os = "linux"))]
        let ring_slot_dma = false;

        #[cfg(target_os = "linux")]
        let backend = match uring {
            Some(reader) => ram_tier::FetchBackend::Uring {
                reader,
                mode: ring_mode.expect("an open ring implies a selection mode"),
            },
            None => match direct {
                Some(direct) => ram_tier::FetchBackend::Direct(direct),
                None => ram_tier::FetchBackend::Mmap {
                    willneed_inline: env.willneed,
                },
            },
        };
        #[cfg(not(target_os = "linux"))]
        let backend = match direct {
            Some(direct) => ram_tier::FetchBackend::Direct(direct),
            None => ram_tier::FetchBackend::Mmap {
                willneed_inline: env.willneed,
            },
        };
        let fetcher = ram_tier::ExpertFetcher::new(Arc::clone(&gguf), backend);

        let usage = (env.usage_save_secs > 0)
            .then(|| ram_tier::ExpertUsage::load_or_new(model_path, env.usage_save_secs));
        let willneed = (env.willneed && mmap_reads)
            .then(|| ram_tier::WillNeedThread::spawn(Arc::clone(&gguf)));

        eprintln!(
            "hi-cuda expert streaming: io={} slot_dma={} upload={} madvise_random={} willneed={} usage={} prewarm_frac={:.2}",
            fetcher.io_label(),
            ring_slot_dma,
            if engine.is_some() { "async" } else { "sync" },
            env.madvise_random && mmap_reads,
            env.willneed && mmap_reads,
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
                    prefetched: false,
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
            ring_slot_dma,
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

    /// Device-pool misses served by the pinned RAM tier so far (no disk I/O);
    /// 0 without a tier. Snapshot/delta counter for the decode timers.
    pub(crate) fn tier_hits(&self) -> u64 {
        self.tier.as_ref().map_or(0, |tier| tier.stats().hits)
    }

    /// The `/health` expert-streaming segment: device-pool counters plus the
    /// RAM-tier addition (hits/misses/evictions/pinned/budget/disk MB/s).
    pub(crate) fn stats_segment(&self) -> String {
        let stats = self.stats;
        let mut out = format!(
            "pool(hits={},misses={},evictions={},read_mb={},prefetch={}/{})",
            stats.hits,
            stats.misses,
            stats.evictions,
            stats.bytes_read / (1024 * 1024),
            stats.prefetch_hits,
            stats.prefetch_wasted,
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
                    self.fetcher.io_label(),
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
        #[cfg(target_os = "linux")]
        let (warmed, warmed_bytes) = if self.ring_slot_dma {
            self.prewarm_ring(&jobs)
        } else {
            self.prewarm_copy(&jobs)
        };
        #[cfg(not(target_os = "linux"))]
        let (warmed, warmed_bytes) = self.prewarm_copy(&jobs);
        self.read_nanos += started.elapsed().as_nanos() as u64;
        self.stats.bytes_read += warmed_bytes;
        eprintln!(
            "hi-cuda expert RAM tier: pre-warmed {warmed} expert slices ({} MiB) from {} usage entries in {:.1}s",
            warmed_bytes / (1024 * 1024),
            self.usage.as_ref().map(|usage| usage.len()).unwrap_or(0),
            started.elapsed().as_secs_f64(),
        );
    }

    /// Chunked pre-warm reads through [`parallel_fetch`] (which itself uses
    /// the ring for owned batches when active), CPU-copied into sticky tier
    /// entries. Chunking keeps the transient pageable staging small.
    fn prewarm_copy(&mut self, jobs: &[(ExpertKey, &ExpertSource)]) -> (usize, u64) {
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
        (warmed, warmed_bytes)
    }

    /// Ring pre-warm: reserve sticky tier slots and let the O_DIRECT reads
    /// DMA straight into them — the whole warm set at queue depth with zero
    /// CPU copies. Best-effort like [`ExpertPool::prewarm_copy`]: failed
    /// slices roll back and are simply not warmed.
    #[cfg(target_os = "linux")]
    fn prewarm_ring(&mut self, jobs: &[(ExpertKey, &ExpertSource)]) -> (usize, u64) {
        use crate::expert_uring::{SlotDest, UringJob, UringRead};

        let (arena_base, stride) = {
            let tier = self.tier.as_ref().expect("tier checked above");
            let pinned = tier.arena().pinned().expect("ring_slot_dma implies pinned");
            (pinned.as_mut_ptr() as usize, tier.slot_stride())
        };
        let mut ring_jobs: Vec<UringJob> = Vec::new();
        let mut slots: Vec<usize> = Vec::new();
        for (key, source) in jobs {
            let Ok((shard, offset, len)) = self.fetcher.file_extent(*key, source) else {
                continue;
            };
            let tier = self.tier.as_mut().expect("tier checked above");
            let Some((slot, slot_base)) = tier.reserve(*key, len, true, 0) else {
                continue;
            };
            slots.push(slot);
            ring_jobs.push(UringJob {
                shard,
                offset,
                len,
                dest: Some(SlotDest {
                    ptr: (arena_base + slot_base) as *mut u8,
                    cap: stride,
                }),
            });
        }
        if ring_jobs.is_empty() {
            return (0, 0);
        }
        let mut outcomes: Vec<Option<Result<UringRead>>> =
            (0..ring_jobs.len()).map(|_| None).collect();
        let reader = self.fetcher.uring().expect("ring_slot_dma implies uring");
        // SAFETY: each dest is a distinct freshly-reserved tier slot (aligned
        // base, stride-sized), untouched by anything else during the batch
        // (pre-warm runs single-threaded at construction).
        let batch =
            unsafe { reader.read_batch(&ring_jobs, |idx, outcome| outcomes[idx] = Some(outcome)) };
        if let Err(err) = batch {
            eprintln!("hi-cuda expert RAM tier: ring pre-warm aborted ({err:#})");
        }
        let tier = self.tier.as_mut().expect("tier checked above");
        let mut warmed = 0usize;
        let mut warmed_bytes = 0u64;
        for ((slot, job), outcome) in slots.iter().zip(&ring_jobs).zip(outcomes) {
            match outcome {
                Some(Ok(UringRead::InPlace { head })) => {
                    tier.commit_reserved(*slot, head);
                    warmed += 1;
                    warmed_bytes += job.len as u64;
                }
                _ => tier.abort_reserved(*slot),
            }
        }
        (warmed, warmed_bytes)
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
    ///   copies during this call; the NEXT pass's begin host-waits them
    ///   before touching tier/staging bytes.
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
        self.ensure_resident_inner(requests, engine_stream, false)
    }

    /// Evented ensure (`HI_CUDA_EXPERT_ASYNC`, the host-nonblocking decode
    /// path): uploads ride the copy stream fenced by fork/done events against
    /// `compute`, and the pass is left OPEN — the caller stages its pointer
    /// tables through [`ExpertPool::stage_table_upload`] on the same stream,
    /// then calls [`ExpertPool::finish_evented`] BEFORE launching the GEMVs
    /// that consume the returned addresses. The host never waits on the
    /// uploads: cross-pass reuse of pinned tier bytes is protected by a
    /// begin-of-pass done-wait taken only when the pass overwrites tier
    /// bytes, staging regions by their per-generation guards, and device pool
    /// slots by fork/done stream ordering (an evicted slot's new bytes land
    /// on the same copy stream AFTER the fork, i.e. after every
    /// previously-enqueued GEMV that could still read the old bytes).
    pub(crate) fn ensure_resident_evented(
        &mut self,
        requests: &[(ExpertKey, &ExpertSource)],
        compute: &Stream,
    ) -> Result<Vec<u64>> {
        if self.engine.is_none() {
            bail!("evented expert ensure requires the async upload engine");
        }
        let addrs = self.ensure_resident_inner(requests, Some(compute), true);
        if addrs.is_err()
            && let Some(engine) = &mut self.engine
        {
            engine.abort_pass();
        }
        addrs
    }

    /// Stage one device pointer-table rewrite onto the copy stream of the
    /// OPEN evented pass (bytes are consumed into pinned staging before this
    /// returns, so the caller may reuse them immediately). Ordered before the
    /// pass's `done` event like every expert upload.
    pub(crate) fn stage_table_upload(&mut self, dst: &DeviceBuffer, bytes: &[u8]) -> Result<()> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| anyhow!("pointer-table staging requires the async upload engine"))?;
        let staged = engine.stage_upload(dst, 0, bytes);
        if staged.is_err() {
            engine.abort_pass();
        }
        staged
    }

    /// Close the open evented pass: record `done` on the copy stream and make
    /// `compute` wait on it, so every GEMV launched afterwards sees completed
    /// expert slots AND pointer tables. Host-nonblocking.
    pub(crate) fn finish_evented(&mut self, compute: &Stream) -> Result<()> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| anyhow!("finish_evented requires the async upload engine"))?;
        let finished = engine.finish_pass(Some(compute));
        if finished.is_err() {
            engine.abort_pass();
        }
        finished
    }

    /// Whether the evented ensure path is available (async upload engine up).
    pub(crate) fn evented_available(&self) -> bool {
        self.engine.is_some()
    }

    fn ensure_resident_inner(
        &mut self,
        requests: &[(ExpertKey, &ExpertSource)],
        engine_stream: Option<&Stream>,
        evented: bool,
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
                if self.slots[slot].prefetched {
                    self.slots[slot].prefetched = false;
                    self.stats.prefetch_hits += 1;
                }
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
            self.slots[slot].prefetched = false;
            self.resident.insert(*key, slot);
            addrs[idx] = self.slot_device_addr(slot);
            misses.push((*key, source, slot));
        }
        if misses.is_empty() {
            // Evented callers still owe pointer-table stagings + the finish:
            // open the pass (fork ordering covers table rewrites of tables
            // whose previous-step GEMVs are still enqueued).
            if evented {
                self.engine
                    .as_mut()
                    .expect("evented ensure requires the engine")
                    .begin_pass(engine_stream, false)?;
            }
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
        // Whether this pass will WRITE pinned tier-arena bytes (disk misses
        // insert / ring reserves DMA into tier slots). Only those writes can
        // collide with a previous evented pass's in-flight DMA *reads* of the
        // tier, so only they pay the begin-of-pass done-wait in evented mode.
        let writes_pinned_tier = !disk_jobs.is_empty()
            && self
                .tier
                .as_ref()
                .is_some_and(|tier| tier.arena().is_pinned());
        let sync_done = !evented || writes_pinned_tier;
        // Ring fast path: the whole miss batch is submitted to io_uring at
        // queue depth, O_DIRECT reads DMA straight into reserved pinned tier
        // slots, and each slice's H2D is enqueued as its read completes.
        #[cfg(target_os = "linux")]
        if self.ring_slot_dma {
            return self.ring_pass(addrs, &misses, &tier_hits, engine_stream, pass, evented);
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
                engine.begin_pass(engine_stream, sync_done)?;
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
                if !evented {
                    engine.finish_pass(engine_stream)?;
                }
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

    /// The io_uring ensure pass (`ring_slot_dma`): reserve a pinned tier slot
    /// per disk miss, submit the whole batch at queue depth, and enqueue each
    /// slice's async H2D as its O_DIRECT read lands (completion-driven
    /// uploads: the copy stream drains while later reads are still in
    /// flight). Tier declines fall back to owned scratch reads staged through
    /// the pinned staging rings, exactly like the legacy path. On any error
    /// the reserved-but-unread slots are rolled back and the copy stream is
    /// host-synced before returning, so no DMA dangles into the tier arena.
    #[cfg(target_os = "linux")]
    fn ring_pass(
        &mut self,
        addrs: Vec<u64>,
        misses: &[(ExpertKey, &ExpertSource, usize)],
        tier_hits: &[Option<(usize, usize)>],
        engine_stream: Option<&Stream>,
        pass: u64,
        evented: bool,
    ) -> Result<Vec<u64>> {
        use crate::expert_uring::{SlotDest, UringJob, UringRead};

        enum Plan {
            /// Reads straight into tier slot `slot` (arena offset `slot_base`).
            Slot {
                miss_idx: usize,
                slot: usize,
                slot_base: usize,
            },
            /// Tier declined (every slot pass-pinned): owned scratch + staging.
            Owned { miss_idx: usize },
        }

        // Resolve every extent before mutating anything (fallible).
        let mut extents: Vec<Option<(usize, u64, usize)>> = Vec::with_capacity(misses.len());
        for ((key, source, _), hit) in misses.iter().zip(tier_hits) {
            if hit.is_some() {
                extents.push(None);
            } else {
                extents.push(Some(self.fetcher.file_extent(*key, source)?));
            }
        }

        // The engine pass must open BEFORE any tier-arena write: when an
        // engine stream is in play, the previous pass's uploads may still be
        // reading the very slots the ring is about to overwrite. Ring reserves
        // DMA into tier slots from the NVMe side (not event-orderable), so an
        // evented pass with disk misses always pays the done-wait.
        let sync_done = !evented || extents.iter().any(Option::is_some);
        self.engine
            .as_mut()
            .expect("ring_slot_dma implies the upload engine")
            .begin_pass(engine_stream, sync_done)?;

        // Reserve destination slots (provisional entries, committed as reads
        // land). Slots the tier declines read into owned scratch instead.
        let (arena_base, stride) = {
            let tier = self.tier.as_ref().expect("ring_slot_dma implies the tier");
            let pinned = tier.arena().pinned().expect("ring_slot_dma implies pinned");
            (pinned.as_mut_ptr() as usize, tier.slot_stride())
        };
        let mut jobs: Vec<UringJob> = Vec::new();
        let mut plans: Vec<Plan> = Vec::new();
        {
            let tier = self.tier.as_mut().expect("checked above");
            for (miss_idx, ((key, _, _), extent)) in misses.iter().zip(&extents).enumerate() {
                let Some((shard, offset, len)) = *extent else {
                    continue; // tier hit
                };
                let dest = match tier.reserve(*key, len, false, pass) {
                    Some((slot, slot_base)) => {
                        plans.push(Plan::Slot {
                            miss_idx,
                            slot,
                            slot_base,
                        });
                        Some(SlotDest {
                            ptr: (arena_base + slot_base) as *mut u8,
                            cap: stride,
                        })
                    }
                    None => {
                        plans.push(Plan::Owned { miss_idx });
                        None
                    }
                };
                jobs.push(UringJob {
                    shard,
                    offset,
                    len,
                    dest,
                });
            }
        }

        let mut first_err: Option<anyhow::Error> = None;
        {
            let engine = self.engine.as_ref().expect("checked above");
            let tier = self.tier.as_ref().expect("checked above");
            let pinned = tier.arena().pinned().expect("checked above");
            // Tier hits are ready immediately: enqueue their DMAs first so
            // the copy stream works while the NVMe reads run.
            for ((_, _, device_slot), hit) in misses.iter().zip(tier_hits) {
                let Some((tier_offset, len)) = hit else {
                    continue;
                };
                if let Err(err) = engine.upload_pinned(
                    &self.arena,
                    device_slot * self.slot_bytes,
                    pinned,
                    *tier_offset,
                    *len,
                ) {
                    first_err = Some(err);
                    break;
                }
            }
            // Drive the ring; completion-driven uploads for slot-DMA reads.
            let mut outcomes: Vec<Option<Result<UringRead>>> =
                (0..jobs.len()).map(|_| None).collect();
            if first_err.is_none() && !jobs.is_empty() {
                let reader = self
                    .fetcher
                    .uring()
                    .expect("ring_slot_dma implies the uring backend");
                let device_arena = &self.arena;
                let slot_bytes = self.slot_bytes;
                let started = Instant::now();
                // SAFETY: every Slot dest is a distinct reserved tier slot —
                // 4 KiB-aligned base (arena base alignment checked at
                // construction, stride is a 4 KiB multiple), stride-sized,
                // eviction-protected for this pass, and written by nothing
                // else until the batch returns.
                let batch = unsafe {
                    reader.read_batch(&jobs, |idx, outcome| {
                        if let (
                            Plan::Slot {
                                miss_idx,
                                slot_base,
                                ..
                            },
                            Ok(UringRead::InPlace { head }),
                        ) = (&plans[idx], &outcome)
                            && first_err.is_none()
                        {
                            let device_slot = misses[*miss_idx].2;
                            if let Err(err) = engine.upload_pinned(
                                device_arena,
                                device_slot * slot_bytes,
                                pinned,
                                slot_base + head,
                                jobs[idx].len,
                            ) {
                                first_err = Some(err);
                            }
                        }
                        outcomes[idx] = Some(outcome);
                    })
                };
                self.read_nanos += started.elapsed().as_nanos() as u64;
                if let Err(err) = batch
                    && first_err.is_none()
                {
                    first_err = Some(err);
                }
            }

            // Publish / roll back the reserved slots and collect owned bytes.
            let mut owned_uploads: Vec<(usize, Vec<u8>)> = Vec::new();
            {
                let tier = self.tier.as_mut().expect("checked above");
                for (idx, plan) in plans.iter().enumerate() {
                    let outcome = outcomes[idx].take();
                    match plan {
                        Plan::Slot { slot, .. } => match outcome {
                            Some(Ok(UringRead::InPlace { head })) => {
                                tier.commit_reserved(*slot, head);
                                self.stats.bytes_read += jobs[idx].len as u64;
                            }
                            Some(Ok(UringRead::Owned(_))) => {
                                unreachable!("slot job returned owned bytes")
                            }
                            Some(Err(err)) => {
                                tier.abort_reserved(*slot);
                                if first_err.is_none() {
                                    first_err = Some(err);
                                }
                            }
                            // Ring-level failure before this job completed.
                            None => tier.abort_reserved(*slot),
                        },
                        Plan::Owned { miss_idx } => match outcome {
                            Some(Ok(UringRead::Owned(bytes))) => {
                                owned_uploads.push((*miss_idx, bytes));
                            }
                            Some(Ok(UringRead::InPlace { .. })) => {
                                unreachable!("owned job landed in place")
                            }
                            Some(Err(err)) => {
                                if first_err.is_none() {
                                    first_err = Some(err);
                                }
                            }
                            None => {}
                        },
                    }
                }
            }
            // Owned fallbacks stage through the pinned staging rings. The
            // tier declined these this pass (all slots pass-pinned), so no
            // insert attempt either.
            let engine = self.engine.as_mut().expect("checked above");
            if first_err.is_none() {
                for (miss_idx, bytes) in &owned_uploads {
                    let device_slot = misses[*miss_idx].2;
                    if let Err(err) =
                        engine.stage_upload(&self.arena, device_slot * self.slot_bytes, bytes)
                    {
                        first_err = Some(err);
                        break;
                    }
                    self.stats.bytes_read += bytes.len() as u64;
                }
            }
        }
        self.stats.misses += misses.len() as u64;

        let engine = self.engine.as_mut().expect("checked above");
        match first_err {
            None => {
                if !evented {
                    engine.finish_pass(engine_stream)?;
                }
                Ok(addrs)
            }
            Some(err) => {
                // Host-sync so no in-flight copy still reads tier slots or
                // staging when the caller unwinds.
                let _ = engine.finish_pass(None);
                Err(err)
            }
        }
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
            if self.slots[victim].prefetched {
                self.slots[victim].prefetched = false;
                self.stats.prefetch_wasted += 1;
            }
        }
        Ok(victim)
    }

    /// Temporal prefetch (`HI_CUDA_EXPERT_PREFETCH`): speculatively upload
    /// `requests` whose bytes sit in the PINNED RAM tier and are not already
    /// device-resident, fire-and-forget on the copy stream. Must run AFTER
    /// [`ExpertPool::finish_evented`] — the copies are deliberately NOT gated
    /// into the compute stream (mispredictions must never delay this layer's
    /// GEMVs); the next ensure pass's `done` record covers them for every
    /// waiter, and [`UploadEngine::record_tail`] re-arms the drain/tier
    /// barriers immediately. Never touches disk, never evicts a slot pinned
    /// by the pass that just ran, and marks slots for the hits/wasted
    /// counters. Returns how many uploads were enqueued.
    pub(crate) fn prefetch_evented(
        &mut self,
        requests: &[(ExpertKey, &ExpertSource)],
    ) -> Result<usize> {
        if self.engine.is_none() || self.pass == 0 {
            return Ok(0);
        }
        let tier_pinned = self
            .tier
            .as_ref()
            .is_some_and(|tier| tier.arena().is_pinned());
        if !tier_pinned {
            return Ok(0);
        }
        let pass = self.pass;
        let mut enqueued = 0usize;
        for (key, source) in requests {
            if self.resident.contains_key(key) || source.bytes_per_expert > self.slot_bytes {
                continue;
            }
            let Some((tier_offset, len)) = self
                .tier
                .as_mut()
                .expect("tier checked above")
                .lookup(key, pass)
            else {
                continue;
            };
            // A fully pass-pinned pool: stop prefetching rather than error.
            let Ok(slot) = self.take_slot(pass) else {
                break;
            };
            self.tick += 1;
            self.slots[slot].key = Some(*key);
            self.slots[slot].last_use = self.tick;
            // Deliberately NOT pass-pinned: a prefetched slot ages out via
            // LRU if the prediction was wrong.
            self.slots[slot].pinned_pass = 0;
            self.slots[slot].prefetched = true;
            self.resident.insert(*key, slot);
            let dst_offset = slot * self.slot_bytes;
            let engine = self.engine.as_ref().expect("checked above");
            let pinned = self
                .tier
                .as_ref()
                .and_then(|tier| tier.arena().pinned())
                .expect("pinned tier checked above");
            if let Err(err) =
                engine.upload_pinned(&self.arena, dst_offset, pinned, tier_offset, len)
            {
                // Keep the drain/tier barriers covering whatever DID enqueue.
                let _ = self.engine.as_mut().expect("checked above").record_tail();
                return Err(err);
            }
            enqueued += 1;
        }
        if enqueued > 0 {
            self.engine.as_mut().expect("checked above").record_tail()?;
        }
        Ok(enqueued)
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

/// Up to one expert extent per (layer, projection) source, rotating the
/// expert index, for the AUTO-mode page-cache residency sample (spread over
/// layers and shards; the sampler sub-samples further).
#[cfg(target_os = "linux")]
fn sample_expert_extents(
    gguf: &GgufFile,
    sources: &BTreeMap<(u32, u8), ExpertSource>,
) -> Vec<(usize, u64, usize)> {
    let mut extents = Vec::with_capacity(sources.len());
    for (index, source) in sources.values().enumerate() {
        if source.expert_count == 0 {
            continue;
        }
        let Ok(range) = gguf.tensor_file_range(&source.tensor_name) else {
            continue;
        };
        let expert = (index * 7) % source.expert_count;
        let offset = range.file_offset + (expert * source.bytes_per_expert) as u64;
        extents.push((range.shard, offset, source.bytes_per_expert));
    }
    extents
}

/// Fetch a batch of expert byte extents (results in job order): the whole
/// batch through the io_uring backend at queue depth when active, otherwise
/// up to [`READ_WORKERS`] threads. Used by miss batches and pre-warm.
fn parallel_fetch(
    fetcher: &ram_tier::ExpertFetcher,
    jobs: &[(ExpertKey, &ExpertSource)],
) -> Vec<Result<Vec<u8>>> {
    if let Some(results) = fetcher.fetch_batch_uring(jobs) {
        return results;
    }
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
pub(crate) mod tests {
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
    pub(crate) fn write_streaming_fixture(path: &Path) {
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
    pub(crate) fn fixture_dir(name: &str) -> PathBuf {
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
            iouring: Some(false),
            iouring_qd: ram_tier::DEFAULT_IOURING_QD,
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

        let mmap_fetcher = ram_tier::ExpertFetcher::new(
            Arc::clone(&gguf),
            ram_tier::FetchBackend::Mmap {
                willneed_inline: true,
            },
        );
        let direct = gguf.direct_io_reader().ok();
        if direct.is_none() {
            eprintln!("skipping O_DIRECT half: unsupported filesystem");
        }
        let odirect_fetcher = direct.map(|reader| {
            ram_tier::ExpertFetcher::new(Arc::clone(&gguf), ram_tier::FetchBackend::Direct(reader))
        });
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

    /// Byte equivalence io_uring == mmap on the synthetic fixture GGUF: the
    /// ring backend must return exactly the mmap bytes for every expert
    /// extent, one at a time and as a whole owned batch. Skips (loudly) where
    /// io_uring or O_DIRECT is unavailable.
    #[cfg(target_os = "linux")]
    #[test]
    fn fetcher_uring_matches_mmap_on_fixture() {
        let dir = fixture_dir("uring-fetch");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = Arc::new(GgufFile::open(&model).unwrap());
        let sources = ram_tier::discover_sources(&gguf);

        let reader = match crate::expert_uring::IoUringReader::open(
            &[model.clone()],
            crate::expert_uring::DEFAULT_QD,
        ) {
            Ok(reader) => reader,
            Err(err) => {
                eprintln!("skipping io_uring fetcher test: {err:#}");
                std::fs::remove_dir_all(&dir).unwrap();
                return;
            }
        };
        let uring_fetcher = ram_tier::ExpertFetcher::new(
            Arc::clone(&gguf),
            ram_tier::FetchBackend::Uring {
                reader,
                mode: "forced",
            },
        );
        assert!(
            uring_fetcher
                .io_label()
                .starts_with("iouring(forced,qd=256")
        );
        let requests = all_requests(&sources);
        // Single fetches.
        for (key, source) in &requests {
            let expected = expected_expert_bytes(&gguf, *key);
            let via_uring = uring_fetcher.fetch(*key, source).unwrap();
            assert_eq!(via_uring, expected, "io_uring fetch of {key:?}");
        }
        // Whole batch through one drive loop.
        let batch = uring_fetcher.fetch_batch_uring(&requests).unwrap();
        for ((key, _), bytes) in requests.iter().zip(batch) {
            assert_eq!(
                bytes.unwrap(),
                expected_expert_bytes(&gguf, *key),
                "io_uring batch fetch of {key:?}"
            );
        }
        // Out-of-range experts fail loudly.
        let (&(layer, proj), source) = sources.iter().next().unwrap();
        assert!(
            uring_fetcher
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
        assert_prewarm_serves_without_disk(&mut pool, &gguf, &sources);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Shared tail of the prewarm tests: routing to the pre-warmed expert
    /// hits the tier for all three projections with zero new disk bytes and
    /// byte-exact device contents.
    fn assert_prewarm_serves_without_disk(
        pool: &mut ExpertPool,
        gguf: &GgufFile,
        sources: &BTreeMap<(u32, u8), ExpertSource>,
    ) {
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
                expected_expert_bytes(gguf, *key),
                "pre-warmed {key:?}"
            );
        }
    }

    /// The io_uring tier end to end on a CUDA device: ring-mode construction
    /// (probe, widened stride, buffer registration), zero-copy DMA into
    /// reserved tier slots, completion-driven uploads, device evictions, and
    /// repeat passes served from the tier with zero extra disk bytes — all
    /// byte-exact against the GGUF. Skips loudly where io_uring or O_DIRECT
    /// is unavailable.
    #[cfg(target_os = "linux")]
    #[test]
    fn native_cuda_uring_tier_end_to_end_bytes_and_bounds() {
        let dir = fixture_dir("uring-tier");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);

        let stride = crate::expert_uring::tier_slot_stride(FIXTURE_EXPERT_BYTES);
        let mut env = test_env((24 * stride) as u64, 0.0, false);
        env.iouring = Some(true);
        env.iouring_qd = 64;
        let mut pool =
            ExpertPool::new_with_env(&model, FIXTURE_EXPERT_BYTES, 3 * FIXTURE_EXPERT_BYTES, env)
                .unwrap();
        if !pool.fetcher.io_label().starts_with("iouring") {
            eprintln!("skipping: io_uring unavailable here (fell back cleanly)");
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        }
        assert!(
            pool.ring_slot_dma,
            "ring + pinned tier + engine must enable slot DMA"
        );
        let tier = pool.tier.as_ref().unwrap();
        assert_eq!(tier.slot_stride(), stride);
        assert_eq!(tier.slot_count(), 24, "stride-aware budget cap");
        let stream = Stream::create().unwrap();
        let source_of = |layer: u32, proj: u8| sources.get(&(layer, proj)).unwrap();

        // Pass 1: three misses -> O_DIRECT ring reads into tier slots.
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((0u32, 0u8, expert), source_of(0, 0)))
            .collect();
        let addrs = pool.ensure_resident(&requests).unwrap();
        let disk_after_first = pool.stats().bytes_read;
        assert_eq!(disk_after_first, 3 * FIXTURE_EXPERT_BYTES as u64);
        let base = pool.arena.as_ptr() as u64;
        for (idx, (key, _)) in requests.iter().enumerate() {
            let offset = usize::try_from(addrs[idx] - base).unwrap();
            let bytes = pool
                .arena
                .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                .unwrap();
            assert_eq!(bytes, expected_expert_bytes(&gguf, *key), "ring {key:?}");
        }

        // Pass 2 (engine-stream variant): different experts evict the device
        // slots; the tier keeps everything.
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((1u32, 2u8, expert), source_of(1, 2)))
            .collect();
        pool.ensure_resident_on(&requests, Some(&stream)).unwrap();
        stream.synchronize().unwrap();
        assert!(pool.stats().evictions >= 3);

        // Pass 3: the originals again -> tier hits, zero new disk bytes,
        // byte-exact device contents.
        let disk_before_repeat = pool.stats().bytes_read;
        let requests: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((0u32, 0u8, expert), source_of(0, 0)))
            .collect();
        let addrs = pool.ensure_resident(&requests).unwrap();
        assert_eq!(
            pool.stats().bytes_read,
            disk_before_repeat,
            "tier hits cost no disk I/O in ring mode"
        );
        assert_eq!(pool.tier.as_ref().unwrap().stats().hits, 3);
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
        // Tier stays bounded under pressure: hammer every expert through.
        let all = all_requests(&sources);
        pool.ensure_resident(&all).unwrap_err();
        // (24 requests through a 3-slot device pool in ONE pass must fail:
        // every slot is pass-pinned. Split per projection instead.)
        for chunk in all.chunks(3) {
            pool.ensure_resident(chunk).unwrap();
            let tier = pool.tier.as_ref().unwrap();
            assert!(tier.resident_bytes() <= tier.budget_bytes());
            assert!(tier.resident_count() <= tier.slot_count());
        }
        let segment = pool.stats_segment();
        assert!(segment.contains("io=iouring(forced,qd=64"), "{segment}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Enqueue enough device work on `stream` that copies forked after it
    /// stay QUEUED for a few milliseconds — long enough that the host-side
    /// passes below run while the previous pass's uploads are genuinely in
    /// flight (the hazard windows these tests exist to exercise).
    fn stuff_stream(stream: &Stream) -> DeviceBuffer {
        let ballast = DeviceBuffer::alloc(32 * 1024 * 1024).unwrap();
        for _ in 0..40 {
            ballast.memset_zero_async(stream).unwrap();
        }
        ballast
    }

    /// The evented-ensure slot-churn hazard, deterministically: with the
    /// compute stream stuffed, evented passes run back-to-back on a 3-slot
    /// pool WITHOUT any host wait, so pass N+1's LRU hands pass N's slots to
    /// new keys while pass N's H2D into those very slots is still queued
    /// behind the fork event. Correctness relies exactly on the protocol
    /// under test: overwrites ride the same copy stream (FIFO), readers are
    /// fenced by fork/done — after one final compute-stream sync (which
    /// waits the last `done`), the surviving keys' device bytes must equal
    /// their GGUF bytes. Requires a CUDA device (runs in the native suite).
    #[test]
    fn native_cuda_evented_ensure_survives_slot_churn_with_inflight_uploads() {
        let dir = fixture_dir("evented-churn");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);
        let source_of = |layer: u32, proj: u8| sources.get(&(layer, proj)).unwrap();
        let keys = |layer: u32, proj: u8| -> Vec<(ExpertKey, &ExpertSource)> {
            (0..3)
                .map(|expert| ((layer, proj, expert), source_of(layer, proj)))
                .collect()
        };

        // Device pool: 3 slots (every pass evicts the previous one
        // wholesale). Tier: everything, so the churn passes are pure tier
        // hits — no disk, no tier writes, no host barriers, maximum overlap.
        let tier_bytes = (24 * FIXTURE_EXPERT_BYTES) as u64;
        let mut pool = ExpertPool::new_with_env(
            &model,
            FIXTURE_EXPERT_BYTES,
            3 * FIXTURE_EXPERT_BYTES,
            test_env(tier_bytes, 0.0, false),
        )
        .unwrap();
        assert!(pool.evented_available());
        let compute = Stream::create().unwrap();

        // Seed the tier (blocking passes; each evicts the previous device
        // set out of the 3 slots).
        let set_a = keys(0, 0);
        let set_b = keys(1, 2);
        let set_c = keys(0, 1);
        pool.ensure_resident(&set_a).unwrap();
        pool.ensure_resident(&set_b).unwrap();
        pool.ensure_resident(&set_c).unwrap();
        let disk_after_seed = pool.stats().bytes_read;

        // Stuff the compute stream, then churn A -> B -> A -> B with zero
        // host waits.
        let _ballast = stuff_stream(&compute);
        let mut final_addrs = Vec::new();
        for (round, set) in [&set_a, &set_b, &set_a, &set_b].iter().enumerate() {
            let addrs = pool.ensure_resident_evented(set, &compute).unwrap();
            pool.finish_evented(&compute).unwrap();
            if round == 3 {
                final_addrs = addrs;
            }
        }
        assert_eq!(
            pool.stats().bytes_read,
            disk_after_seed,
            "churn passes must be pure tier hits"
        );

        compute.synchronize().unwrap();
        let base = pool.arena.as_ptr() as u64;
        for (idx, (key, _)) in set_b.iter().enumerate() {
            let offset = usize::try_from(final_addrs[idx] - base).unwrap();
            let bytes = pool
                .arena
                .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                .unwrap();
            assert_eq!(
                bytes,
                expected_expert_bytes(&gguf, *key),
                "churned slot for {key:?}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The evented tier-write barrier, deterministically: pass 1 uploads
    /// straight out of pinned tier slots (in flight behind a stuffed compute
    /// stream); pass 2's disk misses must recycle those very tier slots (the
    /// tier only HAS 3) — the begin-of-pass done-wait is what keeps the host
    /// from overwriting a DMA source. Both passes' device bytes must be
    /// exact. Requires a CUDA device (runs in the native suite).
    #[test]
    fn native_cuda_evented_disk_misses_wait_for_inflight_tier_reads() {
        let dir = fixture_dir("evented-tier-barrier");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);
        let source_of = |layer: u32, proj: u8| sources.get(&(layer, proj)).unwrap();

        // Tier: 3 slots only. Device pool: 6 slots (both passes coresident).
        let mut pool = ExpertPool::new_with_env(
            &model,
            FIXTURE_EXPERT_BYTES,
            6 * FIXTURE_EXPERT_BYTES,
            test_env((3 * FIXTURE_EXPERT_BYTES) as u64, 0.0, false),
        )
        .unwrap();
        assert!(pool.tier.as_ref().unwrap().arena().is_pinned());
        let compute = Stream::create().unwrap();

        let set_a: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((0u32, 0u8, expert), source_of(0, 0)))
            .collect();
        let set_b: Vec<(ExpertKey, &ExpertSource)> = (0..3)
            .map(|expert| ((1u32, 1u8, expert), source_of(1, 1)))
            .collect();
        // Pass 1: disk misses insert into the tier's only 3 slots, and the
        // uploads read those slots — queued behind the stuffed stream.
        let _ballast = stuff_stream(&compute);
        let addrs_a = pool.ensure_resident_evented(&set_a, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        // Pass 2 immediately: its tier inserts MUST evict pass 1's tier
        // slots. The begin-of-pass barrier host-waits pass 1's copies first;
        // without it these host memcpys would race the in-flight DMAs.
        let addrs_b = pool.ensure_resident_evented(&set_b, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();

        compute.synchronize().unwrap();
        let base = pool.arena.as_ptr() as u64;
        for (set, addrs) in [(&set_a, &addrs_a), (&set_b, &addrs_b)] {
            for (idx, (key, _)) in set.iter().enumerate() {
                let offset = usize::try_from(addrs[idx] - base).unwrap();
                let bytes = pool
                    .arena
                    .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                    .unwrap();
                assert_eq!(bytes, expected_expert_bytes(&gguf, *key), "{key:?}");
            }
        }
        assert_eq!(
            pool.stats().bytes_read,
            (6 * FIXTURE_EXPERT_BYTES) as u64,
            "both passes read disk exactly once per key"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The ring (io_uring slot-DMA) flavor of the evented flow — the real
    /// GLM-5.2 configuration: evented tier-hit passes churn device slots with
    /// in-flight uploads, and an evented disk-miss pass whose O_DIRECT reads
    /// DMA into recycled tier slots pays the begin-of-pass barrier. Byte
    /// exactness end to end. Requires a CUDA device; skips loudly where
    /// io_uring is unavailable.
    #[cfg(target_os = "linux")]
    #[test]
    fn native_cuda_uring_evented_ensure_bytes_exact_under_churn() {
        let dir = fixture_dir("uring-evented");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);
        let source_of = |layer: u32, proj: u8| sources.get(&(layer, proj)).unwrap();
        let keys = |layer: u32, proj: u8| -> Vec<(ExpertKey, &ExpertSource)> {
            (0..3)
                .map(|expert| ((layer, proj, expert), source_of(layer, proj)))
                .collect()
        };

        let stride = crate::expert_uring::tier_slot_stride(FIXTURE_EXPERT_BYTES);
        let mut env = test_env((3 * stride) as u64, 0.0, false);
        env.iouring = Some(true);
        env.iouring_qd = 64;
        let mut pool =
            ExpertPool::new_with_env(&model, FIXTURE_EXPERT_BYTES, 3 * FIXTURE_EXPERT_BYTES, env)
                .unwrap();
        if !pool.fetcher.io_label().starts_with("iouring") {
            eprintln!("skipping: io_uring unavailable here (fell back cleanly)");
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        }
        assert!(pool.ring_slot_dma);
        let compute = Stream::create().unwrap();
        let set_a = keys(0, 0);
        let set_b = keys(1, 2);

        // Evented passes back-to-back with the compute stream stuffed: pass 1
        // ring-reads A into the 3 tier slots and uploads from them; pass 2's
        // ring reserves MUST recycle those slots (barrier); pass 3 hits the
        // tier for B... which pass 2 just cached, then pass 4 re-reads A.
        let _ballast = stuff_stream(&compute);
        pool.ensure_resident_evented(&set_a, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        pool.ensure_resident_evented(&set_b, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        let addrs_a = pool.ensure_resident_evented(&set_a, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();

        compute.synchronize().unwrap();
        let base = pool.arena.as_ptr() as u64;
        for (idx, (key, _)) in set_a.iter().enumerate() {
            let offset = usize::try_from(addrs_a[idx] - base).unwrap();
            let bytes = pool
                .arena
                .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                .unwrap();
            assert_eq!(bytes, expected_expert_bytes(&gguf, *key), "ring {key:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Temporal prefetch: speculative uploads land real bytes, ensure passes
    /// on predicted keys count as prefetch hits with zero new disk reads,
    /// never-used predictions age out as prefetch_wasted, and keys absent
    /// from the tier are never speculatively read from disk. Requires a CUDA
    /// device (runs in the native suite).
    #[test]
    fn native_cuda_prefetch_evented_warms_predicted_keys_and_counts() {
        let dir = fixture_dir("prefetch-evented");
        let model = dir.join("model.gguf");
        write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();
        let sources = ram_tier::discover_sources(&gguf);
        let source_of = |layer: u32, proj: u8| sources.get(&(layer, proj)).unwrap();
        let keys = |layer: u32, proj: u8| -> Vec<(ExpertKey, &ExpertSource)> {
            (0..3)
                .map(|expert| ((layer, proj, expert), source_of(layer, proj)))
                .collect()
        };

        let tier_bytes = (24 * FIXTURE_EXPERT_BYTES) as u64;
        let mut pool = ExpertPool::new_with_env(
            &model,
            FIXTURE_EXPERT_BYTES,
            6 * FIXTURE_EXPERT_BYTES,
            test_env(tier_bytes, 0.0, false),
        )
        .unwrap();
        let compute = Stream::create().unwrap();
        let set_a = keys(0, 0);
        let set_b = keys(1, 2);
        let set_c = keys(0, 1);

        // Seed the tier with A and B, then push A off the device via C.
        pool.ensure_resident(&set_a).unwrap();
        pool.ensure_resident(&set_b).unwrap();
        pool.ensure_resident(&set_c).unwrap();
        assert!(pool.stats().evictions >= 3);
        let disk_after_seed = pool.stats().bytes_read;

        // An evented pass on B, then prefetch A (tier hits, not resident).
        pool.ensure_resident_evented(&set_b, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        assert_eq!(pool.prefetch_evented(&set_a).unwrap(), 3);
        // Predicted correctly: the next ensure sees pure hits and counts
        // them; bytes are exact with zero new disk reads.
        let addrs = pool.ensure_resident_evented(&set_a, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        assert_eq!(pool.stats().prefetch_hits, 3);
        assert_eq!(pool.stats().bytes_read, disk_after_seed);
        compute.synchronize().unwrap();
        let base = pool.arena.as_ptr() as u64;
        for (idx, (key, _)) in set_a.iter().enumerate() {
            let offset = usize::try_from(addrs[idx] - base).unwrap();
            let bytes = pool
                .arena
                .copy_to_host_offset::<u8>(offset, FIXTURE_EXPERT_BYTES)
                .unwrap();
            assert_eq!(
                bytes,
                expected_expert_bytes(&gguf, *key),
                "prefetched {key:?}"
            );
        }

        // Mispredicted: prefetch C (tier-resident since its seed pass), never
        // ensure it, and displace it -> prefetch_wasted. C's slots carry the
        // newest last_use, so it takes two passes (B evicts the stale A set,
        // then A evicts C as the oldest survivor) to age C out.
        assert_eq!(pool.prefetch_evented(&set_c).unwrap(), 3);
        pool.ensure_resident_evented(&set_b, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        pool.ensure_resident_evented(&set_a, &compute).unwrap();
        pool.finish_evented(&compute).unwrap();
        compute.synchronize().unwrap();
        assert_eq!(
            pool.stats().prefetch_wasted,
            3,
            "unused prefetched slots must count as wasted when evicted; stats: {:?}",
            pool.stats()
        );
        // Keys whose bytes are NOT in the tier are never speculatively read
        // from disk: prefetch declines them outright.
        let set_d = keys(1, 0);
        assert_eq!(pool.prefetch_evented(&set_d).unwrap(), 0);
        assert_eq!(pool.stats().bytes_read, disk_after_seed);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Ring pre-warm: the usage history warms sticky tier entries via
    /// zero-copy O_DIRECT DMA at construction, then serves them with no disk
    /// reads. Requires a CUDA device; skips where io_uring is unavailable.
    #[cfg(target_os = "linux")]
    #[test]
    fn native_cuda_uring_prewarm_dma_pins_hottest_experts() {
        let dir = fixture_dir("uring-prewarm");
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

        let stride = crate::expert_uring::tier_slot_stride(FIXTURE_EXPERT_BYTES);
        let mut env = test_env((24 * stride) as u64, 1.0, false);
        env.iouring = Some(true);
        let mut pool =
            ExpertPool::new_with_env(&model, FIXTURE_EXPERT_BYTES, 4 * FIXTURE_EXPERT_BYTES, env)
                .unwrap();
        if !pool.fetcher.io_label().starts_with("iouring") {
            eprintln!("skipping: io_uring unavailable here (fell back cleanly)");
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        }
        assert!(pool.ring_slot_dma);
        let tier_stats = pool.tier.as_ref().unwrap().stats();
        assert_eq!(
            tier_stats.prewarmed, 6,
            "2 hot experts x 3 projections, DMA'd sticky"
        );
        assert_prewarm_serves_without_disk(&mut pool, &gguf, &sources);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
