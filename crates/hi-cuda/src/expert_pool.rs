//! Fixed-size device pool for streamed MoE expert weights.
//!
//! Giant MoE models (GLM-5.2 class: hundreds of GB of routed experts) cannot
//! hold every expert resident. The dense trunk loads as usual; routed experts
//! stay in the (split) GGUF on disk and are paged into this pool on demand:
//! one fixed device arena divided into equal slots, an LRU over resident
//! experts, and per-(layer, projection) host mirrors of the device pointer
//! tables the grouped MoE GEMV kernels consume. Expert slices are contiguous
//! in the GGUF (rank-3 tensors, expert-major), so a miss is one mmap read +
//! one host-to-device copy into the victim slot — no byte transformation
//! (quantized matrix normalization is validation-only).
//!
//! This is the synchronous Phase-2 form of the expert-streaming plan: misses
//! block the forward. Prefetch/overlap (pinned staging, IO threads) layers on
//! top in Phase 3 without changing this interface.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::{GgufFile, GgufTensorType};

use crate::runtime::DeviceBuffer;

/// Identifies one routed expert's weights: (layer, projection, expert).
/// Projection: 0 = gate, 1 = up, 2 = down.
pub(crate) type ExpertKey = (u32, u8, u32);

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
    pub bytes_read: u64,
}

struct Slot {
    key: Option<ExpertKey>,
    last_use: u64,
    /// Guards a slot from eviction while the current ensure pass also needs it.
    pinned_pass: u64,
}

pub(crate) struct ExpertPool {
    /// Re-opened model file(s); owning it keeps the mmaps alive independent of
    /// the caller's `GgufFile` borrow during model construction.
    gguf: GgufFile,
    arena: DeviceBuffer,
    slot_bytes: usize,
    slots: Vec<Slot>,
    resident: HashMap<ExpertKey, usize>,
    tick: u64,
    pass: u64,
    stats: ExpertPoolStats,
}

impl ExpertPool {
    pub(crate) fn new(
        model_path: &std::path::Path,
        slot_bytes: usize,
        budget_bytes: usize,
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
        let gguf = GgufFile::open(model_path)
            .with_context(|| format!("re-opening {} for expert streaming", model_path.display()))?;
        Ok(Self {
            gguf,
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
        })
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

    fn slot_device_addr(&self, slot: usize) -> u64 {
        self.arena.as_ptr() as u64 + (slot * self.slot_bytes) as u64
    }

    /// Make every (key, source) pair resident, returning each expert's device
    /// address in order. One "pass" pins all requested experts so a miss never
    /// evicts another expert needed by the same forward step. Misses read from
    /// disk CONCURRENTLY (scoped threads over the mmap slices — the page
    /// faults are the expensive part, and parallel reads fill the NVMe queue),
    /// then upload serially (host-to-device runs at memory speed).
    pub(crate) fn ensure_resident(
        &mut self,
        requests: &[(ExpertKey, &ExpertSource)],
    ) -> Result<Vec<u64>> {
        self.pass += 1;
        let pass = self.pass;
        // First pass: mark hits so they cannot be evicted by this pass's misses.
        for (key, _) in requests {
            if let Some(&slot) = self.resident.get(key) {
                self.tick += 1;
                self.slots[slot].last_use = self.tick;
                self.slots[slot].pinned_pass = pass;
            }
        }
        // Assign slots for the misses (serial: eviction order stays
        // deterministic), dedup within the request list.
        let mut addrs = vec![0u64; requests.len()];
        let mut misses: Vec<(usize, ExpertKey, &ExpertSource, usize)> = Vec::new();
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
            misses.push((idx, *key, source, slot));
        }
        if misses.is_empty() {
            return Ok(addrs);
        }
        // Concurrent disk reads into per-miss staging buffers.
        let gguf = &self.gguf;
        let mut staged: Vec<Result<Vec<u8>>> = Vec::with_capacity(misses.len());
        let workers = misses.len().min(6);
        if workers <= 1 {
            for (_, key, source, _) in &misses {
                staged.push(Self::read_expert_bytes(gguf, *key, source));
            }
        } else {
            let jobs = std::sync::Mutex::new(misses.iter().enumerate());
            let results =
                std::sync::Mutex::new((0..misses.len()).map(|_| None).collect::<Vec<_>>());
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| {
                        loop {
                            let job = { jobs.lock().unwrap().next() };
                            let Some((slot_idx, (_, key, source, _))) = job else {
                                break;
                            };
                            let bytes = Self::read_expert_bytes(gguf, *key, source);
                            results.lock().unwrap()[slot_idx] = Some(bytes);
                        }
                    });
                }
            });
            staged = results
                .into_inner()
                .unwrap()
                .into_iter()
                .map(|entry| entry.expect("expert read job completed"))
                .collect();
        }
        // Serial host-to-device uploads into the assigned slots.
        for ((_, key, source, slot), bytes) in misses.iter().zip(staged) {
            let bytes = bytes?;
            if source.bytes_per_expert > self.slot_bytes {
                bail!(
                    "expert {} of {} needs {} bytes; pool slots are {} bytes",
                    key.2,
                    source.tensor_name,
                    source.bytes_per_expert,
                    self.slot_bytes
                );
            }
            self.arena
                .copy_from_host_at(slot * self.slot_bytes, &bytes)?;
            self.stats.misses += 1;
            self.stats.bytes_read += source.bytes_per_expert as u64;
        }
        Ok(addrs)
    }

    /// Copy one expert's contiguous byte range out of the (split) GGUF mmap.
    fn read_expert_bytes(
        gguf: &GgufFile,
        key: ExpertKey,
        source: &ExpertSource,
    ) -> Result<Vec<u8>> {
        let expert = key.2 as usize;
        if expert >= source.expert_count {
            bail!(
                "expert {} out of range for {} ({} experts)",
                expert,
                source.tensor_name,
                source.expert_count
            );
        }
        let view = gguf
            .tensor(&source.tensor_name)
            .ok_or_else(|| anyhow!("expert tensor {} missing from GGUF", source.tensor_name))?;
        let start = expert
            .checked_mul(source.bytes_per_expert)
            .context("expert byte offset overflows usize")?;
        let end = start
            .checked_add(source.bytes_per_expert)
            .context("expert byte range overflows usize")?;
        let bytes = view.bytes.get(start..end).ok_or_else(|| {
            anyhow!(
                "expert {} byte range {start}..{end} exceeds tensor {} ({} bytes)",
                expert,
                source.tensor_name,
                view.bytes.len()
            )
        })?;
        Ok(bytes.to_vec())
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
