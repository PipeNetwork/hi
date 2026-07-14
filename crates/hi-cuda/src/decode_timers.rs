//! Env-gated per-phase wall-clock timing for the qwen-family CUDA decode step
//! (`HI_CUDA_DECODE_TIMERS=1`), built to answer "where does each GLM-5.2
//! decode millisecond go" — expert fetch vs attention vs host round trips.
//!
//! One aggregated line is printed to stderr every `HI_CUDA_DECODE_TIMERS_EVERY`
//! decoded tokens (default 16), machine-parseable `key=value` pairs:
//!
//! ```text
//! hi-cuda decode timers[16 tok]: total=412.31ms/tok embed=0.05 \
//!   attn_qkv=38.10(mla_host=22.00) kv_write=0.80 attn=9.10 attn_out=2.10 \
//!   ffn_dense=1.20 route=2.20 route_sync=11.40 \
//!   expert_ensure=291.00(hit=1493 miss=307 ram_hit=250 disk_read=2610.0MiB) \
//!   expert_gemv=51.20 moe_shexp=2.90 logits=5.10 sample=3.50 other=9.80 \
//!   syncs/tok=812.0(dtoh=395.0 htod=402.0 stream=15.0 event=0.0)
//! ```
//!
//! Time values are host wall-clock milliseconds per token (window average, 2
//! decimals); the parenthesised expert counters are window totals; `syncs/tok`
//! are per-token averages of the host<->device synchronisation points hit
//! inside the timed spans (the existing blocking `cudaMemcpy` /
//! `cudaStreamSynchronize` / `cudaEventSynchronize` call sites in
//! `runtime.rs`, counted — never added — by this module).
//!
//! Semantics and caveats (kernel launches are asynchronous):
//! * Spans measure HOST wall time. Device work queued by a phase but not yet
//!   awaited drains at the next synchronising call, so its cost lands in the
//!   phase that owns that sync. Concretely, on the streamed-MoE path the
//!   `route_sync` readback of layer L+1 absorbs whatever device work from
//!   layer L was still in flight (expert GEMVs, attention); `sample` absorbs
//!   the tail of the step (final norm + lm-head). Phases that only launch
//!   kernels (`expert_gemv`, `attn` on a fully-async build) therefore show
//!   launch cost, not kernel cost. No new synchronisation points are added by
//!   the timers — measuring must not change the decode's overlap behaviour —
//!   so within-span device attribution stays approximate by design.
//! * On the qwen MLA path (`attention_qkv_f32_device`) the q/kv_a/kv_b host
//!   round trips are real blocking copies, so `mla_host` (nested inside
//!   `attn_qkv`) is accurate wall time including the device drain they imply.
//! * `sample` runs outside the decode-step span (the generation loop samples
//!   from the previous step's logits), so it is added to `total` separately;
//!   the one next-token selection a prefill performs is also recorded, making
//!   a window's sample count occasionally tokens+1.
//! * `other` = `total` minus every listed phase (nested `mla_host` excluded);
//!   it holds RMS norms, residual adds, buffer allocation and orchestration.
//! * Windows flush at the START of the next decode step once `EVERY` tokens
//!   have accumulated; a partial window at the end of a generation is dropped
//!   (the next generation continues filling it).
//! * Only the paged decode-step entries mark steps (single-sequence
//!   `decode_one_logits_paged_device*` and the scheduler's batched
//!   `decode_batch_logits_paged_device*`); prefill calls into the same
//!   instrumented primitives record nothing because no step is active.
//! * Under CUDA-graph decode capture the eager forward runs once inside
//!   capture (spans then measure capture cost); replayed steps never re-enter
//!   these functions, so graph-decode tokens are invisible to the timers.
//!   GLM's streamed-MoE decode is not capturable (blocking route readback),
//!   so its every token is timed.
//!
//! Cost when OFF: each instrumented site performs one thread-local flag load
//! (plus a single `OnceLock` env parse per process at the first decode step);
//! no `Instant::now`, no allocation, no formatting.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use std::time::Instant;

/// Number of `Phase` variants (accumulator array size).
pub(crate) const PHASE_COUNT: usize = 14;

/// One timed span kind inside a decode step. Indices are accumulator slots.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Phase {
    /// Token-id upload + embedding gather (`embed_tokens_device`).
    Embed = 0,
    /// Whole q/k/v production (`attention_qkv_f32_device`): projections,
    /// norms, RoPE, and (MLA) latent decompression. Contains `MlaHost`.
    AttnQkv = 1,
    /// MLA host round trips nested inside `AttnQkv`: q readback + host pe
    /// rope + re-upload, kv_a readback + latent/k_pe split, kv_b readback +
    /// host K/V assembly + re-upload.
    MlaHost = 2,
    /// Paged KV cache append for the new token.
    KvWrite = 3,
    /// Paged decode attention kernel span.
    Attn = 4,
    /// Attention output projection (+ optional gate).
    AttnOut = 5,
    /// Dense FFN body (leading dense layers of MoE models, dense models).
    FfnDense = 6,
    /// MoE router projection + top-k routing kernel launch.
    Route = 7,
    /// Streamed MoE only: the blocking route-ids readback (plus dedup) that
    /// makes routes host-visible — also the step's main device drain point.
    RouteSync = 8,
    /// Streamed MoE only: expert-pool `ensure_resident` (RAM-tier/disk fetch
    /// + H2D uploads) and device pointer-table rewrites.
    ExpertEnsure = 9,
    /// Grouped MoE compute launches: activation quantize, gate/up/down
    /// grouped GEMVs, SiLU, weighted scatter-reduce.
    ExpertGemv = 10,
    /// Shared-expert projections of MoE layers.
    MoeShexp = 11,
    /// LM-head projection (`output_logits_f32_device`).
    Logits = 12,
    /// Next-token selection (argmax / sampling), incl. its logits readback.
    Sample = 13,
}

/// Host<->device synchronisation flavours counted at the existing call sites.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SyncKind {
    /// Blocking device-to-host `cudaMemcpy` (`copy_to_host*`).
    Dtoh,
    /// Blocking host-to-device `cudaMemcpy` (`copy_from_host*`).
    Htod,
    /// `cudaStreamSynchronize`.
    Stream,
    /// `cudaEventSynchronize` (pinned-staging reuse gates in the expert pool).
    Event,
}

/// `HI_CUDA_DECODE_TIMERS` set to anything but `0`/empty enables the timers.
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HI_CUDA_DECODE_TIMERS").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// `HI_CUDA_DECODE_TIMERS_EVERY`: tokens per printed window (default 16, min 1).
fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("HI_CUDA_DECODE_TIMERS_EVERY")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(16)
            .max(1)
    })
}

/// One aggregation window's raw accumulators. Pure data + formatting so the
/// line format is unit-testable without CUDA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Window {
    pub(crate) tokens: u64,
    /// Sum of decode-step wall nanos plus sample-span wall nanos.
    pub(crate) total_nanos: u64,
    pub(crate) phase_nanos: [u64; PHASE_COUNT],
    /// Expert-pool counters over the window (device-pool hits/misses, misses
    /// served by the pinned RAM tier, bytes read from disk).
    pub(crate) expert_hits: u64,
    pub(crate) expert_misses: u64,
    pub(crate) expert_ram_hits: u64,
    pub(crate) expert_disk_bytes: u64,
    pub(crate) sync_dtoh: u64,
    pub(crate) sync_htod: u64,
    pub(crate) sync_stream: u64,
    pub(crate) sync_event: u64,
}

impl Window {
    fn ms_per_token(&self, nanos: u64) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }
        nanos as f64 / self.tokens as f64 / 1.0e6
    }

    fn per_token(&self, count: u64) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }
        count as f64 / self.tokens as f64
    }

    /// Nanos unattributed to any listed phase: total minus every phase except
    /// the nested `MlaHost` (which is already inside `AttnQkv`).
    fn other_nanos(&self) -> u64 {
        let attributed: u64 = self
            .phase_nanos
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != Phase::MlaHost as usize)
            .map(|(_, nanos)| *nanos)
            .sum();
        self.total_nanos.saturating_sub(attributed)
    }

    /// The aggregated stderr line (without trailing newline).
    pub(crate) fn format_line(&self) -> String {
        let p = |phase: Phase| self.ms_per_token(self.phase_nanos[phase as usize]);
        format!(
            "hi-cuda decode timers[{} tok]: total={:.2}ms/tok embed={:.2} \
             attn_qkv={:.2}(mla_host={:.2}) kv_write={:.2} attn={:.2} attn_out={:.2} \
             ffn_dense={:.2} route={:.2} route_sync={:.2} \
             expert_ensure={:.2}(hit={} miss={} ram_hit={} disk_read={:.1}MiB) \
             expert_gemv={:.2} moe_shexp={:.2} logits={:.2} sample={:.2} other={:.2} \
             syncs/tok={:.1}(dtoh={:.1} htod={:.1} stream={:.1} event={:.1})",
            self.tokens,
            self.ms_per_token(self.total_nanos),
            p(Phase::Embed),
            p(Phase::AttnQkv),
            p(Phase::MlaHost),
            p(Phase::KvWrite),
            p(Phase::Attn),
            p(Phase::AttnOut),
            p(Phase::FfnDense),
            p(Phase::Route),
            p(Phase::RouteSync),
            p(Phase::ExpertEnsure),
            self.expert_hits,
            self.expert_misses,
            self.expert_ram_hits,
            self.expert_disk_bytes as f64 / (1024.0 * 1024.0),
            p(Phase::ExpertGemv),
            p(Phase::MoeShexp),
            p(Phase::Logits),
            p(Phase::Sample),
            self.ms_per_token(self.other_nanos()),
            self.per_token(self.sync_dtoh + self.sync_htod + self.sync_stream + self.sync_event),
            self.per_token(self.sync_dtoh),
            self.per_token(self.sync_htod),
            self.per_token(self.sync_stream),
            self.per_token(self.sync_event),
        )
    }
}

thread_local! {
    static WINDOW: RefCell<Window> = const { RefCell::new(Window {
        tokens: 0,
        total_nanos: 0,
        phase_nanos: [0; PHASE_COUNT],
        expert_hits: 0,
        expert_misses: 0,
        expert_ram_hits: 0,
        expert_disk_bytes: 0,
        sync_dtoh: 0,
        sync_htod: 0,
        sync_stream: 0,
        sync_event: 0,
    }) };
    /// True between a decode step's begin and end: gates phase spans and the
    /// expert counters, so prefill's calls into the same primitives are free.
    static STEP_ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// True while any timed span may attribute sync counts (a decode step or
    /// a standalone sample span).
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// RAII marker for one decode step (one token's forward). Ends the step and
/// accounts its wall time on drop.
pub(crate) struct StepGuard {
    started: Instant,
}

/// Begin a decode step. `None` (and no other cost) when the timers are off or
/// a step is already active (nested/delegating decode entries count once).
/// Flushes the previous window first once it holds `EVERY` tokens, so the
/// printed line never interleaves with a step's own spans.
pub(crate) fn step_begin() -> Option<StepGuard> {
    if !enabled() || STEP_ACTIVE.get() {
        return None;
    }
    WINDOW.with(|window| {
        let mut window = window.borrow_mut();
        if window.tokens >= every() {
            eprintln!("{}", window.format_line());
            *window = Window::default();
        }
    });
    STEP_ACTIVE.set(true);
    RECORDING.set(true);
    Some(StepGuard {
        started: Instant::now(),
    })
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed().as_nanos() as u64;
        STEP_ACTIVE.set(false);
        RECORDING.set(false);
        WINDOW.with(|window| {
            let mut window = window.borrow_mut();
            window.tokens = window.tokens.saturating_add(1);
            window.total_nanos = window.total_nanos.saturating_add(elapsed);
        });
    }
}

/// RAII span accumulating into one phase slot on drop.
pub(crate) struct PhaseGuard {
    phase: Phase,
    started: Instant,
    /// Sample spans run outside the step: they own the RECORDING flag and add
    /// their wall time to the window total as well.
    owns_recording: bool,
}

/// Time a phase within the active decode step; `None` outside a step or when
/// the timers are off (one thread-local flag load).
pub(crate) fn phase(phase: Phase) -> Option<PhaseGuard> {
    if !STEP_ACTIVE.get() {
        return None;
    }
    Some(PhaseGuard {
        phase,
        started: Instant::now(),
        owns_recording: false,
    })
}

/// Time a next-token selection. Sampling happens in the generation loop
/// between decode steps, so this records whenever the timers are enabled; if
/// a step IS active (recurrent layouts select inside the forward) it behaves
/// like a regular phase span.
pub(crate) fn sample_phase() -> Option<PhaseGuard> {
    if STEP_ACTIVE.get() {
        return phase(Phase::Sample);
    }
    if !enabled() {
        return None;
    }
    RECORDING.set(true);
    Some(PhaseGuard {
        phase: Phase::Sample,
        started: Instant::now(),
        owns_recording: true,
    })
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed().as_nanos() as u64;
        if self.owns_recording {
            RECORDING.set(false);
        }
        WINDOW.with(|window| {
            let mut window = window.borrow_mut();
            let slot = &mut window.phase_nanos[self.phase as usize];
            *slot = slot.saturating_add(elapsed);
            if self.owns_recording {
                window.total_nanos = window.total_nanos.saturating_add(elapsed);
            }
        });
    }
}

/// Whether a decode step is currently being timed (gates optional bookkeeping
/// like expert-pool stat snapshots at the call site).
pub(crate) fn step_active() -> bool {
    STEP_ACTIVE.get()
}

/// Count one host<->device synchronisation point. Called from the existing
/// blocking-copy/synchronize sites in `runtime.rs`; a no-op single flag load
/// unless a timed span is active on this thread.
#[inline]
pub(crate) fn count_sync(kind: SyncKind) {
    if !RECORDING.get() {
        return;
    }
    WINDOW.with(|window| {
        let mut window = window.borrow_mut();
        let slot = match kind {
            SyncKind::Dtoh => &mut window.sync_dtoh,
            SyncKind::Htod => &mut window.sync_htod,
            SyncKind::Stream => &mut window.sync_stream,
            SyncKind::Event => &mut window.sync_event,
        };
        *slot = slot.saturating_add(1);
    });
}

/// Record one streamed-MoE ensure pass's expert-pool counter deltas.
pub(crate) fn add_expert_pass(hits: u64, misses: u64, ram_hits: u64, disk_bytes: u64) {
    if !STEP_ACTIVE.get() {
        return;
    }
    WINDOW.with(|window| {
        let mut window = window.borrow_mut();
        window.expert_hits = window.expert_hits.saturating_add(hits);
        window.expert_misses = window.expert_misses.saturating_add(misses);
        window.expert_ram_hits = window.expert_ram_hits.saturating_add(ram_hits);
        window.expert_disk_bytes = window.expert_disk_bytes.saturating_add(disk_bytes);
    });
}

/// Print a one-time decode-geometry line (KV bytes per token, attention path)
/// at the first timed decode step. The closure only runs once, and only when
/// the timers are enabled.
pub(crate) fn print_geometry_once<F: FnOnce() -> String>(describe: F) {
    static PRINTED: OnceLock<()> = OnceLock::new();
    if !enabled() {
        return;
    }
    PRINTED.get_or_init(|| {
        eprintln!("hi-cuda decode timers: {}", describe());
    });
}

#[cfg(test)]
mod tests {
    use super::{PHASE_COUNT, Phase, Window};

    fn ms(value: f64) -> u64 {
        (value * 1.0e6) as u64
    }

    #[test]
    fn window_line_formats_all_phases_and_counters() {
        let mut phase_nanos = [0u64; PHASE_COUNT];
        phase_nanos[Phase::Embed as usize] = ms(0.8);
        phase_nanos[Phase::AttnQkv as usize] = ms(609.6);
        phase_nanos[Phase::MlaHost as usize] = ms(352.0);
        phase_nanos[Phase::KvWrite as usize] = ms(12.8);
        phase_nanos[Phase::Attn as usize] = ms(145.6);
        phase_nanos[Phase::AttnOut as usize] = ms(33.6);
        phase_nanos[Phase::FfnDense as usize] = ms(19.2);
        phase_nanos[Phase::Route as usize] = ms(35.2);
        phase_nanos[Phase::RouteSync as usize] = ms(182.4);
        phase_nanos[Phase::ExpertEnsure as usize] = ms(4656.0);
        phase_nanos[Phase::ExpertGemv as usize] = ms(819.2);
        phase_nanos[Phase::MoeShexp as usize] = ms(46.4);
        phase_nanos[Phase::Logits as usize] = ms(81.6);
        phase_nanos[Phase::Sample as usize] = ms(56.0);
        let window = Window {
            tokens: 16,
            total_nanos: ms(6855.2),
            phase_nanos,
            expert_hits: 1493,
            expert_misses: 307,
            expert_ram_hits: 250,
            expert_disk_bytes: 2610 * 1024 * 1024,
            sync_dtoh: 6320,
            sync_htod: 6432,
            sync_stream: 240,
            sync_event: 16,
        };
        assert_eq!(
            window.format_line(),
            "hi-cuda decode timers[16 tok]: total=428.45ms/tok embed=0.05 \
             attn_qkv=38.10(mla_host=22.00) kv_write=0.80 attn=9.10 attn_out=2.10 \
             ffn_dense=1.20 route=2.20 route_sync=11.40 \
             expert_ensure=291.00(hit=1493 miss=307 ram_hit=250 disk_read=2610.0MiB) \
             expert_gemv=51.20 moe_shexp=2.90 logits=5.10 sample=3.50 other=9.80 \
             syncs/tok=813.0(dtoh=395.0 htod=402.0 stream=15.0 event=1.0)"
        );
    }

    #[test]
    fn window_other_excludes_nested_mla_host_and_clamps() {
        let mut phase_nanos = [0u64; PHASE_COUNT];
        phase_nanos[Phase::AttnQkv as usize] = ms(10.0);
        // Nested inside AttnQkv: must not be double-subtracted from total.
        phase_nanos[Phase::MlaHost as usize] = ms(8.0);
        let window = Window {
            tokens: 1,
            total_nanos: ms(12.0),
            phase_nanos,
            ..Window::default()
        };
        assert_eq!(window.other_nanos(), ms(2.0));

        // Phases can exceed total (sample spans add to total separately, and
        // clocks are monotonic but spans overlap-free is not guaranteed under
        // future edits): other clamps at zero instead of wrapping.
        let window = Window {
            tokens: 1,
            total_nanos: ms(5.0),
            phase_nanos,
            ..Window::default()
        };
        assert_eq!(window.other_nanos(), 0);
    }

    #[test]
    fn empty_window_formats_zeroes_without_dividing_by_zero() {
        let window = Window::default();
        let line = window.format_line();
        assert!(line.starts_with("hi-cuda decode timers[0 tok]: total=0.00ms/tok"));
        assert!(line.contains("syncs/tok=0.0(dtoh=0.0 htod=0.0 stream=0.0 event=0.0)"));
    }
}
