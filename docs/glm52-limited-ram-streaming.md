# GLM-5.2 limited-RAM MoE streaming (colibri-informed)

Status: IMPLEMENTED + VALIDATED (2026-07-13). Highlights:
- **Rope verdict: INTERLEAVED** for glm-dsa (real-checkpoint A/B: split-half
  produced V2-Lite-grade incoherence; matches colibri's token-exact engine).
  `mla_pe_rope_interleaved()` names the families; the fixture now runs
  qk_rope=4 and pins a 6.3e-2 logit separation between the styles — a
  both-sides flip fails loudly.
- **int8 verdict: stays ON for glm-dsa** — layer-by-layer trunk divergence
  1e-6..1.3e-3 across all 78 layers with no compounding (deepseek2's
  pathology was 35% by layer 19); near-tie greedy flips only (colibri-#100
  acceptable). No float/grouped split needed; streaming deadlock avoided.
- **Loader bug fixed at the root**: glm-dsa carries the latent KV widths in
  plain `attention.key/value_length` alongside the true per-head `*_mla`
  dims; `attention_value_head_dim()` now prefers the MLA dim (mirroring the
  key twin) — this also fixes 1.5x-oversized KV pages and health reporting.
- **The bounded-RAM tier** (expert_pool.rs + expert_ram_tier.rs + hi-gguf IO
  surface): pinned-host LRU with itemized budget accounting, `.hi_expert_usage`
  learning cache with startup pre-warm, MADV_RANDOM/threaded-WILLNEED,
  O_DIRECT twin-fd path, double-buffered pinned staged async uploads.
- **Auto-enable**: when routed experts + trunk exceed free VRAM, expert
  streaming turns on by itself with an auto-sized pool (log line explains;
  `HI_CUDA_EXPERT_STREAMING=0` forces resident). `hi -p glm-5.2` needs no env.
- **Bounded-RSS proof (the acceptance test)**: GLM-5.2 REAP50 Q3_K_M (394B,
  169 GB on disk, 5 shards) served under `systemd-run MemoryMax=32G
  MemorySwapMax=0` with pool 48 GiB / RAM tier 20 GiB / O_DIRECT: ready,
  coherent 48-token greedy answer at 0.41 tok/s, **cgroup MemoryPeak 23.7 GiB,
  zero OOM kills**; the learning cache pre-warmed 1,241 expert slices (7 GiB)
  from prior-run usage history in 1.6 s.
- Full-page-cache smoke (251 GB box, GPU 0, 60 GiB pool): load ~40 s, decode
  0.31 cold → 0.49 warm tok/s at 61.7% pool hit after two requests; coherent.
- Suites: 137 default / 408 native at the workstreams' close.
- **io_uring fetch backend (2026-07-13)**: third fetcher — whole miss batches
  at queue depth, O_DIRECT DMA straight into pinned tier slots,
  completion-driven uploads. Measured: cold mmap 0.71 GiB/s → O_DIRECT-family
  6.5 GiB/s (device ceiling) on this box; the ring matches the thread pool on
  GLM-sized extents and removes the CPU copy chain. Follow-up the same day:
  `HI_CUDA_EXPERT_IOURING` became tri-state with **AUTO as the default**
  (ring when the expert set is too big to cache AND cold; mmap when warm or
  cacheable), and the ring now also serves **general model loads** behind
  `HI_CUDA_LOAD_IOURING` (gpu.rs qwen matrices + dsv4 resident uploads;
  cold trunk disk phase 2.5 → 5.6-6.1 GiB/s at the loader's serial seam).
  Full sections at the bottom of this file.

Original plan follows. Goal: users with limited RAM (24–64 GB) + NVMe (+
optionally a GPU) run GLM-5.2-class MoE through `hi -p glm-5.2` with honest
memory budgets. Reference design: `~/colibri` (744B GLM-5.2 on ~25 GB RAM,
pure C); its full internals and our gap analysis live in the session research
(2026-07-13 agent reports); key facts inlined below.

## What already exists on main (prior GLM phases 0–3a)

glm_moe_dsa loads & computes through the qwen GPU path: MLA with q_lora 2048 +
kv_lora 512 (fused kv_b synthesized from split k_b/v_b at load), sigmoid
router (`expert_gating_func=2`) with selection-only noaux_tc bias +
`expert_weights_norm` + `expert_weights_scale`, shared expert, dense-prefix
layers by tensor presence, `nextn_predict_layers` excludes the blk.78 MTP
head, split-GGUF shards, and VRAM expert streaming: `HI_CUDA_EXPERT_STREAMING=1`
+ `HI_CUDA_EXPERT_POOL_BYTES` (expert_pool.rs: single arena, HashMap LRU,
per-pass pinning, K-quant grouped dp4a GEMV + scatter-reduce, mmap→Vec→sync
H2D misses, ≤6 read threads), /health `expert-streaming=` stats. DSA indexer
tensors are ignored by construction (dense-attention path; indexer optional).

## Correctness gates before anything else (Workstream A)

1. **pe-rope style**: gpu.rs:17854 `mla_pe_rope_split_half = !arch.contains("deepseek")`
   gives GLM split-half; colibri's token-exact-validated engine says GLM-5.2 is
   INTERLEAVED (glm.c:806, applied 1459/1466). The tiny-fixture tests use
   qk_rope=2 where the styles coincide — the V2-Lite lesson. Decide on the
   real checkpoint (teacher-forcing / coherence A/B / colibri tiny-oracle
   ported into a fixture with qk_rope large enough to distinguish).
2. **int8-activation numerics**: deepseek2 needed float activations
   (`int8_activation_paths_allowed`, gpu.rs:4046); GLM-5.2 is the same
   MLA/DSv3 lineage and colibri finds int4 GLM sits near argmax ties. BUT the
   grouped MoE path (which expert streaming REQUIRES, bail at 18851) is gated
   on the same predicate. Measure first (HI_MLA_DEBUG_DUMP / parity probes /
   HI_CUDA_NO_KQUANT_DP4A); if int8 fails: split the gate per call-site
   (float trunk GEMVs, int8 grouped MoE) or add a float-activation grouped
   GEMV variant. Never gate glm into the deadlock blindly.
3. **Group-limited routing guard**: `expert_group_count>1` is not implemented;
   GLM-5.2 is n_group=1. Parse + bail if >1 (silent wrong routing otherwise).

## The limited-RAM disk tier (Workstream B — the feature)

Today the "RAM tier" is the unbounded OS page cache; a 32 GB box with a
169–466 GB GGUF thrashes. Borrow colibri's discipline:

1. **Bounded pinned-host expert cache** fronting `ExpertPool::read_expert_bytes`
   (expert_pool.rs:219), keyed like the VRAM pool (layer, proj, expert):
   - Budget: `HI_CUDA_EXPERT_RAM_GB` explicit, else auto = a fraction of
     MemAvailable measured at load MINUS itemized slack incl. a **page-cache
     reserve** (colibri measured buffered preads collapsing 800→180 MB/s when
     the cache is starved; they reserve 2.5 GB) and the working-set slab.
     Print the projected peak at startup (the "OOM-killer never fires" rule).
   - Storage: pinned host memory (PinnedBuffer, runtime.rs:823) so hits
     upload async without a staging copy; LRU with recency tickets;
     frequency counters persisted to `<model_dir>/.hi_expert_usage` and used
     to pre-pin the hottest experts at startup (colibri's "learning cache" —
     their data: profile-ranked placement 0.94 tok/s vs heat-blind same
     capacity 0.29).
2. **Page-cache policy on the GGUF mmap**: `madvise(MADV_RANDOM)` on expert
   tensor extents (kill readahead amplification), keep default readahead for
   trunk/dense; `POSIX_FADV_WILLNEED` readahead of the NEXT routed expert
   block while the current one computes (colibri glm.c:1817 — issued from a
   thread, never inline on a saturated queue: +0.5 ms/call measured);
   optional `HI_CUDA_EXPERT_ODIRECT=1` twin-fd path (their VHDX/ext4 data:
   buffered 0.8 → O_DIRECT 2.3 GB/s).
3. **Upload overlap (phase 3b)**: replace mmap→pageable Vec→sync memcpy with
   the dsv4 CopyEngine pattern (~100 lines on runtime.rs primitives:
   non-blocking copy stream + events + double-buffered pinned staging).
4. **GGUF granularity caveat**: one expert = 3 disjoint extents (gate/up/down
   rank-3 tensors) vs colibri's 1 contiguous ~19 MB pread. Start with 3-read
   fetch + WILLNEED batching; if NVMe profiling shows extent scatter binding,
   add an optional expert-major sidecar repack (one-time, derived file) — NOT
   in scope initially.
5. **Observability**: extend the /health expert-streaming segment with
   ram_tier hits/misses/bytes, pinned count, budget, and disk-read MB/s.
6. **The guarantee (colibri policy, adopted)**: placement/caching never
   changes router semantics or precision — cache pressure affects speed,
   never output. Any lossy knob (e.g. expert top-p, which colibri shows cuts
   reads 30–40%) must be explicit and warn. Byte-exactness caveats mirror
   their #100 finding: kernel-family changes can flip near-tie argmaxes —
   keep A/Bs on fixed prompts and document.

## Test/acceptance plan

- Fixture tests for the RAM tier (bounded size, eviction, pinning, stats).
- Real model: GLM-5.2-REAP50 Q3_K_M (~169 GB, 5 shards,
  ~/.hi/models/glm-5.2-reap50/): teacher-forcing/coherence gates, then
  **bounded-RSS proof**: run hi-local under `systemd-run --user -p
  MemoryMax=32G` (and 24G) on this 251 GB box — decode must work (slowly,
  honestly) with RSS+pinned under budget, zero OOM kills.
- Acceptance matrix LARGE_MODELS row; cuda_real_smoke FixtureSpec entry.
- hi client: `[profiles.glm-5.2]` in hi.toml, service unit example (GPU 0),
  docs in this file updated with measured numbers.

## Later levers (documented, not in scope)

- PILOT-style cross-layer prefetch: L+1's router on L's residual = 71.6%
  top-8 recall (colibri-measured) — feed the readahead queue.
- Device-resident MLA step + latent (576/token) KV cache — dsv4 engine is the
  blueprint; today's expanded-KV MLA path is correct but host-round-trip-bound.
- MTP (blk.78) self-speculation — dsv4 spec-decode machinery is the blueprint;
  note colibri's cold-cache finding (drafts amplify expert traffic ~1.7x) and
  their S-scaling result (verify cost ~linear in S for MoE).
- Expert-major sidecar repack for single-pread expert fetches.

## io_uring fetch backend (2026-07-13)

Third `ExpertFetcher` backend (Linux only, `io-uring = "0.7"`): the miss path
was a ≤6-thread pread pool, so the effective submission concurrency was the
worker count. The ring submits a whole ensure-pass's miss extents (a cold GLM
token is ~1,800 extents ≈ 10 GB) in as few `io_uring_enter` syscalls as
possible and reaps completions as they land.

### Env knobs

- `HI_CUDA_EXPERT_IOURING` — tri-state (since the default-on follow-up): `1`
  forces the ring, `0` forces it off, **unset = AUTO**. Auto rings only when
  the streamable expert bytes exceed 50% of `MemAvailable` AND a `mincore`
  sample of the expert extents is under 50% resident — i.e. the set can
  neither fit in nor already lives in the page cache. Warm or cacheable sets
  stay on mmap (streaming re-reads forever, so the page cache is the implicit
  tier and should be allowed to warm; this keeps the big-RAM warm-cache boxes
  exactly as before). One log line states the choice and why
  (`io_uring auto -> on/off (...)`); /health reports
  `io=iouring(auto|forced,qd=N[,regbuf])`.
- `HI_CUDA_EXPERT_IOURING_QD` — submission queue depth, default 256, clamped
  to a power of two in [8, 4096].
- `HI_CUDA_LOAD_IOURING` — the same tri-state for **general model loads**
  (see the subsection at the end of this file). Auto for loads is
  deliberately more aggressive: loads are one-shot, so it rings when the
  extents are cold (< 90% resident) OR exceed 50% of `MemAvailable`; mmap
  only when warm or unmeasurable-and-fitting.

### How it reads

- **O_DIRECT twin fds** per shard (same 4 KiB alignment rules as the
  `GgufDirectReader` path: offset rounded down, length rounded up, payload at
  the `head` fixup offset inside the destination).
- **Zero-copy into the tier**: in ring mode the pinned RAM-tier slot stride is
  widened to `round_up(slot_bytes, 4K) + 4K` and the slot region is shifted to
  a 4 KiB boundary inside the pinned allocation (`cudaHostAlloc` suballocates
  small buffers unaligned). Each miss reserves a tier slot and the NVMe DMAs
  the aligned span straight into it; the tier records the payload's `head`
  offset. End to end (disk → pinned host → device) there is no CPU memcpy —
  the old path staged through `Vec` scratch plus a `copy_in` into the tier.
- **Completion-driven uploads (phase 2)**: as each slice's read completes, its
  async H2D is enqueued on the copy stream immediately, overlapping the
  remaining NVMe reads. Tier-declined slices (all slots pass-pinned) fall back
  to owned scratch staged after the batch, exactly like the legacy path. The
  engine-stream overlap question is untouched: `ensure_resident_on`'s
  `Some(&stream)` wiring in gpu.rs stays reverted to sync pending the
  cross-test-flake root cause.
- **Registered resources**: shard fds via IORING_REGISTER_FILES; the pinned
  arena via IORING_REGISTER_BUFFERS in 1 GiB stride-multiple iovec chunks.
  Both degrade with a one-line log to the unregistered forms (most of the win
  is queue depth, not fixed buffers — confirmed by the bench).
- **Pre-warm rides the ring**: sticky slots are reserved and DMA'd the same
  way, so the startup warm set loads at device speed with zero copies.
- The `WillNeedThread` and `MADV_RANDOM` machinery are not engaged in ring
  mode (no page cache in play); `/health` reports `io=iouring(qd=N[,regbuf])`.

### Probe + fallback ladder

Construction probes the whole stack — ring setup, O_DIRECT opens, best-effort
fd registration, then **one real read byte-compared against a buffered read**
(buffer registration follows once the pinned tier arena exists) — and on any
failure logs the reason and falls back: **io_uring → O_DIRECT threads →
mmap**. A knob can never fail the model load. Failure modes caught
at load, not decode: kernel < 5.6 (no IORING_OP_READ),
`kernel.io_uring_disabled` sysctl (=2, common on hardened hosts),
seccomp/container denial (Docker's default profile blocks `io_uring_setup`),
O_DIRECT-less filesystems (tmpfs/overlay upper layers), and
EINVAL/ENOMEM/RLIMIT_MEMLOCK on buffer registration (registration is skipped,
the ring still runs; since kernel 5.12 registration charges memcg rather than
RLIMIT_MEMLOCK).

### Measured (this box: 251 GB RAM, ext4 on NVMe root, kernel 6.8)

`cargo test -p hi-cuda --release bench_glm_expert_read_backends -- --ignored
--nocapture` — no GPU needed. 600 random (layer, expert) triples × 3
projections = 1,800 extents, 10.0 GiB payload, extents 4.6–8.25 MiB, page
cache dropped per-range via `posix_fadvise(DONTNEED)` and verified 0% resident
via `mincore` (never a global drop_caches). Device ceiling reference
(sequential 16 MiB O_DIRECT, 1 thread): 6.30 GiB/s.

| backend                          | GiB/s | wall  |
|----------------------------------|------:|------:|
| mmap+willneed (6 threads, cold)  |  0.71 | 14.0s |
| O_DIRECT pread (6 threads)       |  6.49 |  1.5s |
| io_uring qd=8 unregistered       |  6.50 |  1.5s |
| io_uring qd=8 registered         |  6.47 |  1.5s |
| io_uring qd=64 unregistered      |  6.50 |  1.5s |
| io_uring qd=64 registered        |  6.47 |  1.5s |
| io_uring qd=256 unregistered     |  6.51 |  1.5s |
| io_uring qd=256 registered       |  6.46 |  1.5s |

Reading of the numbers, honestly: the 9x is **buffered mmap → O_DIRECT
family** (0.71 → 6.5 GiB/s, and 0.71 GiB/s reproduces the ~723 MB/s baseline
measured for the mmap path earlier in this file). At GLM's 4.6–8.25 MiB
extents the block layer splits every pread into ~64 device-level commands, so
6 threads already saturate this drive and **queue depth beyond 8 buys nothing
here**; registered buffers are neutral. The ring's wins over the O_DIRECT
thread pool on this hardware are therefore CPU- and pipeline-side, not
raw-bandwidth: no per-read scratch alloc + copy-out + tier `copy_in` (three
trips over ~10 GB per cold token), no read-worker threads, and per-slice
upload overlap. On drives/filesystems where extents fragment smaller (or a
future expert-major repack changes the extent mix), the QD headroom is
already in place — re-run the bench to re-evaluate.

### Tests

Default suite (no GPU, no model): alignment math, QD clamping, probe failure
modes (missing shard, tmpfs O_DIRECT denial), registration geometry, byte
equivalence ring == buffered on synthetic files (slots + owned, EOF tails),
batches ≫ QD. Native suite adds: ring == mmap on the streaming fixture GGUF
(`fetcher_uring_matches_mmap_on_fixture`), the ring tier end to end
(`native_cuda_uring_tier_end_to_end_bytes_and_bounds`: byte-exact device
contents, bounded tier, evictions, engine-stream pass) and ring pre-warm
(`native_cuda_uring_prewarm_dma_pins_hottest_experts`). Ignored real-model
checks: `real_glm_shards_uring_matches_mmap_spot_check` and the bench above.
Suites at this workstream's close: 145 default / 116 hi-gguf / 420 native
(the pre-existing GPU-contention flake — a prefill tok/s health counter under
a concurrent training job — reproduced once across runs and passed alone).

## io_uring for general model loads (2026-07-13 follow-up)

The same ring now serves resident-weight loads for ANY GGUF model, behind one
seam: `load_source::LoadByteSource` decides mmap vs ring once per load set
(tri-state `HI_CUDA_LOAD_IOURING`; auto terms above) and serves whole-tensor
bytes either as the mmap view or as an O_DIRECT bulk read
(`read_extent_chunked`: the tensor's block-aligned span split into 8 MiB
sub-reads submitted together, so one multi-hundred-MB tensor fills the queue
by itself). Wiring is deliberately narrow:

- gpu.rs `GpuMatrix::load` byte sourcing (the qwen matrix loop builds one
  `LoadByteSource` for all `matrix_specs`; the vision mmproj loop stays
  mmap-only — small side-file).
- dsv4_gpu.rs `upload_resident` → `upload_matrix`/`upload_grouped`. Bonus fix
  there: dsv4 previously ran `cudaMemcpy` STRAIGHT from file-backed mmap
  pages (the slow page-by-page path); ring bytes live in anonymous RAM, so
  the H2D DMAs at full speed.
- Ring read errors degrade per-tensor to the mmap view with a log line;
  probe failures fall back to mmap — a knob never fails a load.
- safetensors side-loads — DONE 2026-07-13: `SafetensorsFile::open` runs the
  same tri-state + auto over its data section and, when the ring wins,
  bulk-reads the whole section into owned anonymous memory behind the
  unchanged `bytes()` API (coverage justified whole-section reads: the MTP
  shard and all three DSpark shards are 100% loader-read; DFlash reads 70.6%
  at load and row-gathers the rest — `embed_tokens` — at proposal time from
  the owned buffer). Cold byte-layer A/B (fadvise + mincore 0%, file-offset
  order): MTP 4.58 -> 6.27 GiB/s, DSpark shards 4.4 -> 6.3 GiB/s, DFlash an
  honest wash (4.54 vs 4.46 GiB/s — the ring reads its whole 3.36 GiB
  section including the ~1 GiB runtime-only embed table that mmap skips at
  load).
- dsv4 expert prefill — DONE 2026-07-13: `prefill_expert_pool` computes the
  exact per-tensor expert-prefix extents the pool will consume, runs the
  load auto over just those bytes, and when the ring wins streams
  expert-aligned 128 MiB windows off a reader thread so O_DIRECT reads
  overlap the H2D copies (serial read-then-upload measured 4.1 GiB/s, LOSING
  to mmap's readahead-overlapped 4.5 — the windows restore the pipeline).
  Cold A/B on the real V4 GGUF (6 GiB pool, extent-targeted eviction,
  0% residency): 1.31 s mmap (4.6 GiB/s) vs 1.02 s ring (5.7-6.1 GiB/s),
  ~1.3x. Warm production boots keep mmap via the auto, as before.
- Still open as follow-ups: `GpuVector` loads (tiny), the mla split-kv
  synthesis reads, and a whole-set batched loader (headroom row below;
  needs windowed memory bounds).

### Measured (cold trunk = every non-expert tensor, DONTNEED + mincore = 0%)

| backend | GLM trunk 9.17 GiB | V4-Flash trunk 7.38 GiB |
|---|---:|---:|
| mmap to_vec (1 thread — the actual loader) | 2.71 GiB/s | 2.50 GiB/s |
| mmap to_vec (6 threads, context) | 5.92 | 5.80 |
| io_uring per-tensor chunked (the new loader path) | 5.61 | 6.07 |
| io_uring whole-trunk batch (headroom) | 6.49 | 6.50 |

Device ceilings 6.4 / 5.5 GiB/s. So the disk phase of a cold load improves
~2.1-2.4x at the loader's real (serial, per-tensor) seam. Two honest caveats
from measurement: (1) trunk reads are sequential, so buffered readahead is
far better here (2.5-2.7 GiB/s) than on the scattered expert extents (0.7);
(2) an end-to-end A/B on a real 7B q6_k (`real_cold_load_wall_time_ab`,
GPU 0) measured 3.01 s (mmap) vs 3.06 s (ring) — that load is bound by
quantized-matrix normalization + upload, and mmap's page faults naturally
overlap the CPU work while the ring serializes read-then-normalize. The ring
therefore wins the disk phase without regressing CPU-bound loads (ties), and
pays off most where normalization is light (dsv4's native-dtype uploads,
which also gain the anonymous-memory DMA fix) or disks are faster than CPUs.

### Follow-up validation

`facade_ring_bytes_match_mmap_view_on_fixture` (byte equivalence through the
facade), `read_extent_chunked_matches_buffered_reads` (chunk geometry incl.
EOF tails), `real_glm_bulk_chunked_reads_match_mmap_spot_check` (ignored;
744 MiB tensor byte-exact), auto truth tables for both heuristics,
`bench_cold_trunk_load_backends` (ignored; table above), and the ignored
`real_cold_load_wall_time_ab`. Suites at this follow-up's close: 146 default
/ 116 hi-gguf / 425 native (one fully green run; the pre-listed CUDA 906
stream-capture cross-test race hit individual dsv4_backend tests in two other
runs and each passes alone).

The 2026-07-13 drafter/prefill follow-up adds
`safetensors_ring_backing_matches_mmap_backing` (bytes + every typed reader
identical across backings), `dsv4_gpu_expert_pool_prefill_forced_ring_matches_mmap`
(identical pool counters and GEMV parity vs the mmap prefill, including a
partial-prefix pool), mid-tensor slice windows in the facade test, and the
ignored `safetensors_real_cold_read_ab` / `dsv4_real_cold_expert_prefill_ab`
benches. Suites at its close: 150 default (load_source's unit tests now
compile CUDA-free) / 116 hi-gguf / 427 native (one fully green run under a
~68 GB training job on GPU 0; contention flaked one unrelated backend test
in other runs — clean main flaked the same way — and it passes alone).
