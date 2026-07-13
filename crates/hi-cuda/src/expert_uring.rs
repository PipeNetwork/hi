//! io_uring batch reader for streamed MoE expert extents (Linux only).
//!
//! The thread-pool miss path caps the effective NVMe queue depth at the
//! worker count (≤6); scattered 2–8 MB expert extents want QD 64–256 to reach
//! device bandwidth. This module submits a whole ensure-pass's miss extents
//! to one `io_uring` in as few `io_uring_enter` syscalls as possible and
//! reaps completions as they land, reading O_DIRECT (twin-fd, page cache
//! bypassed) either:
//!
//! * straight into caller-provided 4 KiB-aligned slots (the pinned RAM-tier
//!   arena: the NVMe DMAs into page-locked memory and the subsequent H2D DMA
//!   reads the same bytes — zero CPU memcpy end to end), or
//! * into reader-owned aligned scratch, returned as plain `Vec<u8>` (tier
//!   declined / heap tier / probe / tests).
//!
//! O_DIRECT alignment follows the hi-gguf twin-fd rules: the file offset is
//! rounded down to the 4 KiB block, the length rounded up, and the payload
//! starts `head` bytes into the destination ([`aligned_span`]). Slot
//! destinations therefore need `slot_stride`-sized regions
//! ([`tier_slot_stride`]) and the payload is reported at its `head` offset
//! rather than copied down — the caller records that offset (small waste,
//! zero copies).
//!
//! Construction is a probe: ring setup, O_DIRECT twin fds, optional
//! IORING_REGISTER_FILES / IORING_REGISTER_BUFFERS (both degrade to the
//! unregistered forms with a note), then one real read compared against a
//! buffered read of the same bytes. Any failure surfaces as an error so the
//! caller can walk its fallback ladder (io_uring → O_DIRECT threads → mmap);
//! kernels without IORING_OP_READ (<5.6), `kernel.io_uring_disabled`, and
//! seccomp/container denials all fail here, never at decode time.

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use io_uring::{IoUring, opcode, types};

/// O_DIRECT logical-block granularity (file offset, length and destination
/// address must be multiples of this). 4096 covers every NVMe/ext4/xfs
/// configuration in practice, matching hi-gguf's `DIRECT_IO_BLOCK`.
pub(crate) const URING_BLOCK: usize = 4096;

/// `HI_CUDA_EXPERT_IOURING_QD` default.
pub(crate) const DEFAULT_QD: u32 = 256;

/// Largest single registered-buffer iovec the kernel accepts (1 GiB), and the
/// historical cap on the number of iovecs.
const MAX_FIXED_CHUNK: usize = 1 << 30;
const MAX_FIXED_CHUNKS: usize = 1024;

/// Clamp a requested submission queue depth to a sane power of two the kernel
/// will accept (io_uring_setup wants power-of-two entries; IORING_SETUP_CLAMP
/// caps it at the kernel maximum).
pub(crate) fn clamp_qd(qd: u32) -> u32 {
    qd.clamp(8, 4096).next_power_of_two().min(4096)
}

/// O_DIRECT-legal span covering `len` bytes at file offset `offset`:
/// `(aligned_start, head, aligned_len)` where `aligned_start` is `offset`
/// rounded down to [`URING_BLOCK`], `head = offset - aligned_start` is where
/// the payload begins inside the destination, and `aligned_len` is the
/// rounded-up read length.
pub(crate) fn aligned_span(offset: u64, len: usize) -> (u64, usize, usize) {
    let block = URING_BLOCK as u64;
    let aligned_start = offset / block * block;
    let head = (offset - aligned_start) as usize;
    let aligned_len = (head + len).div_ceil(URING_BLOCK) * URING_BLOCK;
    (aligned_start, head, aligned_len)
}

/// RAM-tier slot stride for ring mode: every slot base must be a legal
/// O_DIRECT destination (4 KiB-aligned given a 4 KiB-aligned arena base) and
/// must hold the worst-case aligned span of a `slot_bytes` payload (up to
/// `URING_BLOCK - 1` bytes of head plus tail round-up).
pub(crate) fn tier_slot_stride(slot_bytes: usize) -> usize {
    slot_bytes.div_ceil(URING_BLOCK) * URING_BLOCK + URING_BLOCK
}

/// Caller-owned destination region for one read: 4 KiB-aligned, at least
/// `aligned_span(offset, len).2` bytes. The payload lands `head` bytes in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotDest {
    pub ptr: *mut u8,
    pub cap: usize,
}

/// One extent to read: `len` bytes at absolute file offset `offset` within
/// shard `shard`. `dest: None` reads into reader-owned aligned scratch and
/// returns the payload as a `Vec`.
#[derive(Debug)]
pub(crate) struct UringJob {
    pub shard: usize,
    pub offset: u64,
    pub len: usize,
    pub dest: Option<SlotDest>,
}

/// Where one job's payload ended up.
#[derive(Debug)]
pub(crate) enum UringRead {
    /// In the caller's slot, starting `head` bytes past `SlotDest::ptr`.
    InPlace { head: usize },
    /// Copied out of reader scratch (dest-less jobs only).
    Owned(Vec<u8>),
}

/// Registered-buffer geometry over one contiguous arena: fixed chunks of
/// `chunk_bytes` (a multiple of the slot stride, so no slot straddles a chunk
/// boundary) registered as one iovec each.
#[derive(Debug, Clone, Copy)]
struct RegisteredArena {
    base: usize,
    len: usize,
    chunk_bytes: usize,
}

impl RegisteredArena {
    /// Fixed-buffer index for a read of `len` bytes at `ptr`, or `None` when
    /// the span is not wholly inside one registered chunk.
    fn buf_index(&self, ptr: *mut u8, len: usize) -> Option<u16> {
        let addr = ptr as usize;
        let rel = addr.checked_sub(self.base)?;
        if rel.checked_add(len)? > self.len {
            return None;
        }
        let chunk = rel / self.chunk_bytes;
        if rel + len > (chunk + 1) * self.chunk_bytes {
            return None;
        }
        u16::try_from(chunk).ok()
    }
}

/// Heap allocation aligned to [`URING_BLOCK`] (O_DIRECT destination scratch
/// for dest-less jobs), mirroring hi-gguf's private `AlignedBlockBuf`.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len.max(URING_BLOCK), URING_BLOCK)
            .context("io_uring scratch layout")?;
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            bail!("allocating {len}-byte aligned io_uring scratch failed");
        }
        Ok(Self {
            ptr,
            len: len.max(URING_BLOCK),
        })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is a live allocation of exactly `len` bytes owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, URING_BLOCK)
            .expect("layout validated at construction");
        // SAFETY: allocated with the identical layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

// SAFETY: AlignedBuf is a plain owned allocation (no thread affinity).
unsafe impl Send for AlignedBuf {}

/// Per-job drive-loop state.
struct JobState {
    shard: usize,
    aligned_start: u64,
    aligned_len: usize,
    /// Payload start within the destination.
    head: usize,
    /// Bytes that must be filled before the payload is complete (`head + len`;
    /// the tail up to `aligned_len` is round-up padding an EOF may cut short).
    target: usize,
    filled: usize,
    dest_ptr: *mut u8,
    /// Owns the scratch for dest-less jobs (freed when the state drops).
    scratch: Option<AlignedBuf>,
    /// Fixed-buffer index when the destination lies in the registered arena.
    buf_index: Option<u16>,
}

/// Batch O_DIRECT reader over the shards of a (split) GGUF, one `io_uring`
/// per reader. Thread-safe (`&self`): the ring is driven under a mutex, one
/// whole batch at a time.
pub(crate) struct IoUringReader {
    ring: Mutex<IoUring>,
    files: Vec<File>,
    qd: u32,
    fixed_files: bool,
    registered: Option<RegisteredArena>,
    /// Non-fatal downgrades hit during construction (for the startup log).
    notes: Vec<String>,
}

impl IoUringReader {
    /// Build the ring, open O_DIRECT twin fds for every shard, register the
    /// fds (best-effort), then prove the stack with one real read compared
    /// against a buffered read of the same bytes. Every hard failure mode —
    /// kernel <5.6, `kernel.io_uring_disabled=2`, seccomp/container denial,
    /// an O_DIRECT-less filesystem — surfaces here as `Err`.
    pub(crate) fn open(paths: &[PathBuf], requested_qd: u32) -> Result<Self> {
        if paths.is_empty() {
            bail!("io_uring reader needs at least one shard path");
        }
        let qd = clamp_qd(requested_qd);
        let ring = IoUring::builder()
            .setup_clamp()
            .build(qd)
            .context("io_uring_setup failed (kernel <5.6, io_uring_disabled sysctl, or seccomp/container denial?)")?;
        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            use std::os::unix::fs::OpenOptionsExt;
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
                .with_context(|| {
                    format!(
                        "opening {} with O_DIRECT (unsupported on this filesystem?)",
                        path.display()
                    )
                })?;
            files.push(file);
        }
        let mut notes = Vec::new();
        let fds: Vec<_> = files.iter().map(|file| file.as_raw_fd()).collect();
        let fixed_files = match ring.submitter().register_files(&fds) {
            Ok(()) => true,
            Err(err) => {
                notes.push(format!("register_files failed ({err}); using raw fds"));
                false
            }
        };
        let reader = Self {
            ring: Mutex::new(ring),
            files,
            qd,
            fixed_files,
            registered: None,
            notes,
        };
        reader.probe(&paths[0])?;
        Ok(reader)
    }

    /// One real read through the ring (first block of shard 0), byte-compared
    /// against a buffered `pread` of the same range.
    fn probe(&self, path: &Path) -> Result<()> {
        let file_len = self.files[0]
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let len = usize::try_from(file_len.min(URING_BLOCK as u64)).expect("<= 4096");
        if len == 0 {
            bail!("probe target {} is empty", path.display());
        }
        let mut expected = vec![0u8; len];
        {
            use std::os::unix::fs::FileExt;
            let plain =
                File::open(path).with_context(|| format!("probe open {}", path.display()))?;
            plain
                .read_exact_at(&mut expected, 0)
                .with_context(|| format!("probe buffered read of {}", path.display()))?;
        }
        let mut got = self.read_owned(&[(0, 0, len)]);
        let got = got
            .pop()
            .expect("one probe job")
            .context("io_uring probe read")?;
        if got != expected {
            bail!("io_uring probe read returned different bytes than the buffered read");
        }
        Ok(())
    }

    pub(crate) fn notes(&self) -> &[String] {
        &self.notes
    }

    pub(crate) fn queue_depth(&self) -> u32 {
        self.qd
    }

    pub(crate) fn buffers_registered(&self) -> bool {
        self.registered.is_some()
    }

    /// Register `len` bytes at `base` (the pinned tier arena) as fixed
    /// buffers, chunked so no `slot_stride`-strided slot straddles an iovec.
    /// Errors leave the reader fully functional with unregistered reads (the
    /// caller logs and moves on — most of the win is queue depth, not fixed
    /// buffers).
    pub(crate) fn register_arena(
        &mut self,
        base: *mut u8,
        len: usize,
        slot_stride: usize,
    ) -> Result<()> {
        if base.is_null() || len == 0 {
            bail!("cannot register an empty arena");
        }
        if !(base as usize).is_multiple_of(URING_BLOCK) {
            bail!("arena base {base:p} is not {URING_BLOCK}-byte aligned");
        }
        if slot_stride == 0 || !slot_stride.is_multiple_of(URING_BLOCK) {
            bail!("slot stride {slot_stride} is not a multiple of {URING_BLOCK}");
        }
        let slots_per_chunk = (MAX_FIXED_CHUNK / slot_stride).max(1);
        let chunk_bytes = slots_per_chunk * slot_stride;
        let chunks = len.div_ceil(chunk_bytes);
        if chunks > MAX_FIXED_CHUNKS {
            bail!(
                "arena needs {chunks} fixed-buffer iovecs; the kernel caps at {MAX_FIXED_CHUNKS}"
            );
        }
        let iovecs: Vec<libc::iovec> = (0..chunks)
            .map(|chunk| {
                let offset = chunk * chunk_bytes;
                libc::iovec {
                    // SAFETY: offset < len, so the pointer stays inside the arena.
                    iov_base: unsafe { base.add(offset) }.cast(),
                    iov_len: chunk_bytes.min(len - offset),
                }
            })
            .collect();
        let ring = self
            .ring
            .lock()
            .map_err(|_| anyhow!("io_uring mutex poisoned"))?;
        // SAFETY: the iovecs cover a live allocation the caller keeps alive as
        // long as this reader (the tier arena outlives the fetcher in the
        // pool); registration is undone when the ring drops.
        unsafe { ring.submitter().register_buffers(&iovecs) }.with_context(|| {
            format!(
                "IORING_REGISTER_BUFFERS over {} bytes in {chunks} iovecs (RLIMIT_MEMLOCK?)",
                len
            )
        })?;
        self.registered = Some(RegisteredArena {
            base: base as usize,
            len,
            chunk_bytes,
        });
        Ok(())
    }

    /// Read `extents` (`(shard, file_offset, len)`) into fresh `Vec`s, results
    /// in job order. Convenience wrapper over [`IoUringReader::read_batch`]
    /// with no caller-owned destinations (hence fully safe).
    pub(crate) fn read_owned(&self, extents: &[(usize, u64, usize)]) -> Vec<Result<Vec<u8>>> {
        let jobs: Vec<UringJob> = extents
            .iter()
            .map(|&(shard, offset, len)| UringJob {
                shard,
                offset,
                len,
                dest: None,
            })
            .collect();
        let mut results: Vec<Option<Result<Vec<u8>>>> = (0..jobs.len()).map(|_| None).collect();
        // SAFETY: no job carries a caller destination pointer.
        let batch = unsafe {
            self.read_batch(&jobs, |idx, outcome| {
                results[idx] = Some(outcome.map(|read| match read {
                    UringRead::Owned(bytes) => bytes,
                    UringRead::InPlace { .. } => unreachable!("dest-less job returned InPlace"),
                }));
            })
        };
        match batch {
            Ok(()) => results
                .into_iter()
                .map(|slot| slot.expect("all jobs completed"))
                .collect(),
            Err(err) => {
                // Ring-level failure: every job that did not complete inherits
                // the batch error.
                let msg = format!("{err:#}");
                results
                    .into_iter()
                    .map(|slot| slot.unwrap_or_else(|| Err(anyhow!("{msg}"))))
                    .collect()
            }
        }
    }

    /// Drive one batch: submit every job's aligned read (batched enter
    /// syscalls, up to the ring's queue depth in flight), resubmit short
    /// reads, and call `on_complete(job_index, outcome)` exactly once per job
    /// as its payload lands (completion order, not submission order).
    ///
    /// An `Err` return is a ring-level failure: jobs whose `on_complete` has
    /// not fired have unspecified destination contents and must be discarded.
    ///
    /// # Safety
    ///
    /// Every `Some(dest)` must be a writable region of at least
    /// `aligned_span(offset, len).2` bytes (`cap` is checked against that),
    /// 4 KiB-aligned, disjoint from all other destinations, and unaliased for
    /// the duration of the call.
    pub(crate) unsafe fn read_batch(
        &self,
        jobs: &[UringJob],
        mut on_complete: impl FnMut(usize, Result<UringRead>),
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let mut states: Vec<Option<JobState>> = Vec::with_capacity(jobs.len());
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(jobs.len());
        let mut remaining = 0usize;
        for (idx, job) in jobs.iter().enumerate() {
            match self.prepare(job) {
                Ok(Some(state)) => {
                    states.push(Some(state));
                    queue.push_back(idx);
                    remaining += 1;
                }
                Ok(None) => {
                    // Zero-length read: complete immediately.
                    states.push(None);
                    let outcome = match job.dest {
                        Some(_) => UringRead::InPlace { head: 0 },
                        None => UringRead::Owned(Vec::new()),
                    };
                    on_complete(idx, Ok(outcome));
                }
                Err(err) => {
                    states.push(None);
                    on_complete(idx, Err(err));
                }
            }
        }
        if remaining == 0 {
            return Ok(());
        }

        let mut ring = self
            .ring
            .lock()
            .map_err(|_| anyhow!("io_uring mutex poisoned"))?;
        let (submitter, mut sq, mut cq) = ring.split();
        let mut inflight = 0usize;
        while remaining > 0 {
            // Fill the submission queue as far as it goes (one enter syscall
            // then covers the whole wavefront).
            sq.sync();
            while !sq.is_full() {
                let Some(idx) = queue.pop_front() else { break };
                let state = states[idx].as_ref().expect("queued job has state");
                let entry = self.build_sqe(state, idx);
                // SAFETY: the destination pointer is valid for aligned_len
                // bytes (caller contract for slots, owned scratch otherwise)
                // and outlives the operation (states live past the loop).
                if unsafe { sq.push(&entry) }.is_err() {
                    queue.push_front(idx);
                    break;
                }
                inflight += 1;
            }
            sq.sync();
            if inflight == 0 {
                bail!("io_uring drive loop stalled with {remaining} jobs unsubmitted");
            }
            match submitter.submit_and_wait(1) {
                Ok(_) => {}
                Err(err)
                    if matches!(
                        err.raw_os_error(),
                        Some(libc::EINTR) | Some(libc::EBUSY) | Some(libc::EAGAIN)
                    ) => {}
                Err(err) => {
                    // Hard enter failure with reads possibly in flight: the
                    // kernel may still DMA into the states' scratch and the
                    // caller's slots, so those must not be freed. Reap what
                    // was submitted; if the ring is too broken to reap,
                    // deliberately leak the scratch rather than hand the
                    // kernel dangling memory.
                    let mut stuck_spins = 0u32;
                    while inflight > 0 && stuck_spins < 1_000 {
                        cq.sync();
                        while cq.next().is_some() {
                            inflight -= 1;
                        }
                        if inflight > 0 && submitter.submit_and_wait(1).is_err() {
                            stuck_spins += 1;
                            std::thread::yield_now();
                        }
                    }
                    if inflight > 0 {
                        for state in states.iter_mut() {
                            if let Some(state) = state.take()
                                && let Some(scratch) = state.scratch
                            {
                                std::mem::forget(scratch);
                            }
                        }
                    }
                    return Err(err).context("io_uring_enter");
                }
            }
            cq.sync();
            for cqe in cq.by_ref() {
                let idx = usize::try_from(cqe.user_data()).expect("user_data is a job index");
                inflight -= 1;
                let res = cqe.result();
                let job = &jobs[idx];
                let state = states[idx].as_mut().expect("completed job has state");
                if res < 0 {
                    let err = std::io::Error::from_raw_os_error(-res);
                    states[idx] = None;
                    remaining -= 1;
                    on_complete(
                        idx,
                        Err(anyhow!(err).context(format!(
                            "io_uring O_DIRECT read of {} bytes at {} of shard {}",
                            job.len, job.offset, job.shard
                        ))),
                    );
                    continue;
                }
                state.filled += res as usize;
                if state.filled >= state.target {
                    let outcome = self.finalize(jobs, &mut states, idx);
                    remaining -= 1;
                    on_complete(idx, outcome);
                } else if res == 0 {
                    states[idx] = None;
                    remaining -= 1;
                    on_complete(
                        idx,
                        Err(anyhow!(
                            "io_uring read hit EOF: wanted {} bytes at {} of shard {}",
                            job.len,
                            job.offset,
                            job.shard
                        )),
                    );
                } else {
                    // Short read: resubmit the remainder (the continuation
                    // offset stays block-aligned except at EOF, where the
                    // target check above already exits).
                    queue.push_back(idx);
                }
            }
        }
        Ok(())
    }

    /// Validate one job and build its drive-loop state (`Ok(None)` =
    /// zero-length no-op).
    fn prepare(&self, job: &UringJob) -> Result<Option<JobState>> {
        if job.shard >= self.files.len() {
            bail!(
                "io_uring read references missing GGUF shard {} ({} shards)",
                job.shard,
                self.files.len()
            );
        }
        if job.len == 0 {
            return Ok(None);
        }
        let (aligned_start, head, aligned_len) = aligned_span(job.offset, job.len);
        let (dest_ptr, scratch, buf_index) = match job.dest {
            Some(dest) => {
                if !(dest.ptr as usize).is_multiple_of(URING_BLOCK) {
                    bail!("slot destination {:p} is not 4096-byte aligned", dest.ptr);
                }
                if dest.cap < aligned_len {
                    bail!(
                        "slot destination holds {} bytes; the aligned read needs {aligned_len}",
                        dest.cap
                    );
                }
                let buf_index = self
                    .registered
                    .as_ref()
                    .and_then(|arena| arena.buf_index(dest.ptr, aligned_len));
                (dest.ptr, None, buf_index)
            }
            None => {
                let scratch = AlignedBuf::new(aligned_len)?;
                (scratch.ptr, Some(scratch), None)
            }
        };
        Ok(Some(JobState {
            shard: job.shard,
            aligned_start,
            aligned_len,
            head,
            target: head + job.len,
            filled: 0,
            dest_ptr,
            scratch,
            buf_index,
        }))
    }

    /// SQE for a job's next read (initial or short-read continuation).
    fn build_sqe(&self, state: &JobState, idx: usize) -> io_uring::squeue::Entry {
        // SAFETY: filled < aligned_len (target <= aligned_len and the caller
        // only builds SQEs for unfinished jobs), so the pointer stays inside
        // the destination.
        let ptr = unsafe { state.dest_ptr.add(state.filled) };
        let len = (state.aligned_len - state.filled) as u32;
        let offset = state.aligned_start + state.filled as u64;
        let entry = match (self.fixed_files, state.buf_index) {
            (true, Some(buf_index)) => {
                opcode::ReadFixed::new(types::Fixed(state.shard as u32), ptr, len, buf_index)
                    .offset(offset)
                    .build()
            }
            (false, Some(buf_index)) => opcode::ReadFixed::new(
                types::Fd(self.files[state.shard].as_raw_fd()),
                ptr,
                len,
                buf_index,
            )
            .offset(offset)
            .build(),
            (true, None) => opcode::Read::new(types::Fixed(state.shard as u32), ptr, len)
                .offset(offset)
                .build(),
            (false, None) => {
                opcode::Read::new(types::Fd(self.files[state.shard].as_raw_fd()), ptr, len)
                    .offset(offset)
                    .build()
            }
        };
        entry.user_data(idx as u64)
    }

    /// Turn a filled job into its outcome and drop its state.
    fn finalize(
        &self,
        jobs: &[UringJob],
        states: &mut [Option<JobState>],
        idx: usize,
    ) -> Result<UringRead> {
        let state = states[idx].take().expect("finalizing a live job");
        match &state.scratch {
            Some(scratch) => {
                let payload = scratch
                    .as_slice()
                    .get(state.head..state.target)
                    .ok_or_else(|| anyhow!("scratch smaller than the payload span"))?
                    .to_vec();
                debug_assert_eq!(payload.len(), jobs[idx].len);
                Ok(UringRead::Owned(payload))
            }
            None => Ok(UringRead::InPlace { head: state.head }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-expert-uring-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Deterministic patterned file (odd length so EOF is block-unaligned).
    fn write_pattern_file(path: &Path, len: usize) -> Vec<u8> {
        let bytes: Vec<u8> = (0..len)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8)
            .collect();
        std::fs::write(path, &bytes).unwrap();
        bytes
    }

    fn open_or_skip(paths: &[PathBuf], qd: u32) -> Option<IoUringReader> {
        match IoUringReader::open(paths, qd) {
            Ok(reader) => Some(reader),
            Err(err) => {
                eprintln!("skipping io_uring test: {err:#}");
                None
            }
        }
    }

    #[test]
    fn aligned_span_math_covers_heads_tails_and_exact_blocks() {
        // Block-aligned offset and length: no head, no widening.
        assert_eq!(aligned_span(0, 4096), (0, 0, 4096));
        assert_eq!(aligned_span(8192, 8192), (8192, 0, 8192));
        // Unaligned offset: rounded down, payload at the head offset.
        assert_eq!(aligned_span(5000, 100), (4096, 904, 4096));
        // Head + len crossing a block boundary widens the tail.
        assert_eq!(aligned_span(4095, 2), (0, 4095, 8192));
        // GGUF-typical: 32-byte-aligned tensor offsets.
        let (start, head, len) = aligned_span(1_000_032, 3_500_000);
        assert_eq!(start % 4096, 0);
        assert_eq!(start + head as u64, 1_000_032);
        assert!(len >= head + 3_500_000);
        assert_eq!(len % 4096, 0);
        assert!(len - (head + 3_500_000) < 4096);
    }

    #[test]
    fn tier_slot_stride_holds_any_aligned_span_of_a_slot_payload() {
        for slot_bytes in [1usize, 100, 4096, 4097, 3_500_000, 8 << 20] {
            let stride = tier_slot_stride(slot_bytes);
            assert_eq!(stride % URING_BLOCK, 0, "stride must stay 4K-aligned");
            // Worst case: maximal head with a full-slot payload.
            let (_, head, aligned_len) = aligned_span(URING_BLOCK as u64 - 1, slot_bytes);
            assert_eq!(head, URING_BLOCK - 1);
            assert!(
                aligned_len <= stride,
                "slot_bytes={slot_bytes}: aligned_len {aligned_len} > stride {stride}"
            );
        }
    }

    #[test]
    fn clamp_qd_is_a_sane_power_of_two() {
        assert_eq!(clamp_qd(0), 8);
        assert_eq!(clamp_qd(1), 8);
        assert_eq!(clamp_qd(8), 8);
        assert_eq!(clamp_qd(200), 256);
        assert_eq!(clamp_qd(256), 256);
        assert_eq!(clamp_qd(100_000), 4096);
        assert_eq!(clamp_qd(DEFAULT_QD), 256);
    }

    #[test]
    fn open_fails_cleanly_on_missing_shard() {
        let missing = PathBuf::from("/nonexistent/hi-expert-uring/shard.gguf");
        let err = IoUringReader::open(&[missing], DEFAULT_QD)
            .err()
            .expect("open of a missing shard must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("O_DIRECT") || msg.contains("No such file"),
            "{msg}"
        );
    }

    /// tmpfs rejects O_DIRECT, exercising the twin-fd leg of the probe ladder
    /// (the same failure a container with an overlay tmpfs would produce).
    #[test]
    fn open_fails_cleanly_where_o_direct_is_unsupported() {
        let dir = Path::new("/dev/shm");
        if !dir.is_dir() {
            eprintln!("skipping: /dev/shm not present");
            return;
        }
        let path = dir.join(format!(
            "hi-expert-uring-odirect-probe-{}",
            std::process::id()
        ));
        std::fs::write(&path, vec![7u8; 8192]).unwrap();
        let result = IoUringReader::open(&[path.clone()], DEFAULT_QD);
        std::fs::remove_file(&path).unwrap();
        let Err(err) = result else {
            eprintln!("skipping: this tmpfs accepts O_DIRECT");
            return;
        };
        assert!(format!("{err:#}").contains("O_DIRECT"), "{err:#}");
    }

    #[test]
    fn register_arena_validates_alignment_and_geometry() {
        let dir = scratch_dir("register");
        let path = dir.join("shard.bin");
        write_pattern_file(&path, 64 * 1024);
        // Declared before the reader so the registered memory outlives the ring.
        let arena = AlignedBuf::new(4 * tier_slot_stride(1000)).unwrap();
        let Some(mut reader) = open_or_skip(&[path], 8) else {
            return;
        };
        // Unaligned base is rejected before touching the kernel.
        let unaligned = unsafe { arena.ptr.add(1) };
        assert!(
            reader
                .register_arena(unaligned, 4096, tier_slot_stride(1000))
                .is_err()
        );
        // Stride must be block-aligned.
        assert!(reader.register_arena(arena.ptr, arena.len, 1000).is_err());
        assert!(!reader.buffers_registered());
        // A legal registration sticks (anonymous memory registers fine).
        match reader.register_arena(arena.ptr, arena.len, tier_slot_stride(1000)) {
            Ok(()) => assert!(reader.buffers_registered()),
            Err(err) => eprintln!("registration unavailable here ({err:#}); unregistered mode"),
        }
    }

    /// Byte equivalence against buffered reads on a synthetic file: owned
    /// scratch reads, in-place slot reads (registered and not), odd offsets,
    /// block-straddling extents, and the unaligned EOF tail.
    #[test]
    fn uring_reads_match_buffered_reads_incl_slots_and_eof_tail() {
        let dir = scratch_dir("equiv");
        let path = dir.join("shard.bin");
        let file_len = 256 * 1024 + 1234; // deliberately not block-aligned
        let expected = write_pattern_file(&path, file_len);
        // Declared before the reader so registered memory outlives the ring.
        let stride = tier_slot_stride(65536);
        let arena = AlignedBuf::new(7 * stride).unwrap();
        let Some(mut reader) = open_or_skip(&[path], 16) else {
            return;
        };
        let extents: Vec<(usize, u64, usize)> = vec![
            (0, 0, 4096),                      // exact first block
            (0, 32, 100),                      // GGUF-style 32-byte alignment
            (0, 5000, 10_000),                 // straddles blocks
            (0, 4095, 2),                      // head at block edge
            (0, (file_len - 700) as u64, 700), // unaligned EOF tail
            (0, 8192, 65536),                  // multi-block aligned
            (0, 123_457, 54_321),              // arbitrary
        ];
        // Owned reads.
        let owned = reader.read_owned(&extents);
        for (&(_, offset, len), result) in extents.iter().zip(&owned) {
            let bytes = result.as_ref().unwrap();
            assert_eq!(
                bytes.as_slice(),
                &expected[offset as usize..offset as usize + len],
                "owned read at {offset}+{len}"
            );
        }
        // Slot reads, unregistered then registered.
        assert!(arena.len >= extents.len() * stride);
        for registered in [false, true] {
            if registered && let Err(err) = reader.register_arena(arena.ptr, arena.len, stride) {
                eprintln!("skipping registered half: {err:#}");
                continue;
            }
            assert_eq!(reader.buffers_registered(), registered);
            let jobs: Vec<UringJob> = extents
                .iter()
                .enumerate()
                .map(|(slot, &(shard, offset, len))| UringJob {
                    shard,
                    offset,
                    len,
                    dest: Some(SlotDest {
                        // SAFETY: slot regions are disjoint stride-sized
                        // pieces of one live arena.
                        ptr: unsafe { arena.ptr.add(slot * stride) },
                        cap: stride,
                    }),
                })
                .collect();
            let mut heads: Vec<Option<usize>> = vec![None; jobs.len()];
            // SAFETY: destinations are 4K-aligned, disjoint, and sized to the
            // stride (>= any aligned span of these extents).
            unsafe {
                reader.read_batch(&jobs, |idx, outcome| match outcome.unwrap() {
                    UringRead::InPlace { head } => heads[idx] = Some(head),
                    UringRead::Owned(_) => panic!("slot job returned owned bytes"),
                })
            }
            .unwrap();
            for (slot, (&(_, offset, len), head)) in extents.iter().zip(&heads).enumerate() {
                let head = head.expect("slot job completed");
                assert_eq!(head, offset as usize % URING_BLOCK);
                let got = &arena.as_slice()[slot * stride + head..slot * stride + head + len];
                assert_eq!(
                    got,
                    &expected[offset as usize..offset as usize + len],
                    "slot read at {offset}+{len} (registered={registered})"
                );
            }
        }
        // Reads past EOF fail loudly rather than padding silently.
        let past_eof = reader.read_owned(&[(0, file_len as u64 - 100, 200)]);
        assert!(past_eof[0].is_err(), "EOF-crossing read must error");
        // Bad shard index errors per job without poisoning the batch.
        let mixed = reader.read_owned(&[(9, 0, 100), (0, 0, 100)]);
        assert!(mixed[0].is_err());
        assert_eq!(mixed[1].as_ref().unwrap().as_slice(), &expected[0..100]);
    }

    // -----------------------------------------------------------------------
    // Real-shard tests (ignored: need ~/.hi/models/glm-5.2-reap50/, 5 shards)
    // -----------------------------------------------------------------------

    const GLM_FIRST_SHARD: &str =
        ".hi/models/glm-5.2-reap50/GLM-5.2-REAP50-Q3_K_M-00001-of-00005.gguf";

    fn glm_model_path() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home).join(GLM_FIRST_SHARD);
        path.exists().then_some(path)
    }

    /// One expert-projection slice of the real model.
    #[derive(Clone)]
    struct BenchJob {
        /// Routed-expert tensor holding the slice.
        tensor: String,
        /// Byte offset of the slice within that tensor.
        rel: u64,
        /// Absolute location for direct/ring reads.
        shard: usize,
        abs: u64,
        len: usize,
    }

    /// A realistic cold ensure-pass: `triples` random (layer, expert) picks x
    /// 3 projections, in shuffled (scattered) order. Deterministic seed.
    fn glm_cold_pass_jobs(gguf: &hi_gguf::GgufFile, triples: usize) -> Vec<BenchJob> {
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};
        let config = gguf.qwen_config().expect("GLM qwen config");
        let experts = config.expert_count.expect("MoE expert count") as u64;
        let mut layers: Vec<[(String, hi_gguf::TensorFileRange, usize); 3]> = Vec::new();
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            let mut projections = Vec::new();
            for proj in ["gate", "up", "down"] {
                let Some((name, info)) =
                    hi_gguf::qwen_moe_packed_expert_weight_names(&prefix, proj)
                        .into_iter()
                        .find_map(|name| gguf.tensor_info(&name).cloned().map(|info| (name, info)))
                else {
                    break;
                };
                if info.dimensions.len() != 3 || info.dimensions[2] != experts {
                    break;
                }
                let per_expert = info
                    .dtype
                    .byte_len(info.dimensions[0] * info.dimensions[1])
                    .unwrap() as usize;
                let range = gguf.tensor_file_range(&name).unwrap();
                projections.push((name, range, per_expert));
            }
            if let Ok(projections) = <[_; 3]>::try_from(projections) {
                layers.push(projections);
            }
        }
        assert!(!layers.is_empty(), "no routed-expert layers found");
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x91A6);
        let mut jobs = Vec::with_capacity(triples * 3);
        for _ in 0..triples {
            let layer = &layers[rng.gen_range(0..layers.len())];
            let expert = rng.gen_range(0..experts);
            for (name, range, per_expert) in layer {
                let rel = expert * *per_expert as u64;
                jobs.push(BenchJob {
                    tensor: name.clone(),
                    rel,
                    shard: range.shard,
                    abs: range.file_offset + rel,
                    len: *per_expert,
                });
            }
        }
        jobs.shuffle(&mut rng);
        jobs
    }

    fn shard_paths(gguf: &hi_gguf::GgufFile) -> Vec<PathBuf> {
        (0..gguf.shard_count())
            .map(|shard| gguf.shard_path(shard).unwrap().to_path_buf())
            .collect()
    }

    /// Drop exactly the jobs' page-cache ranges (page-widened), never the
    /// whole system. Call with no mmap of the shards open in this process,
    /// or mapped pages survive the fadvise. Pages mapped by ANOTHER process
    /// (a running hi-local serving this model) survive too — measure with
    /// [`sampled_residency`] rather than assuming cold.
    fn drop_job_ranges_from_page_cache(paths: &[PathBuf], jobs: &[BenchJob]) {
        let files: Vec<File> = paths.iter().map(|path| File::open(path).unwrap()).collect();
        for job in jobs {
            let start = job.abs / 4096 * 4096;
            let end = (job.abs + job.len as u64).div_ceil(4096) * 4096;
            let ret = unsafe {
                libc::posix_fadvise(
                    files[job.shard].as_raw_fd(),
                    start as libc::off_t,
                    (end - start) as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
            assert_eq!(ret, 0, "posix_fadvise(DONTNEED) failed");
        }
    }

    /// Fraction of the jobs' pages resident in the page cache, sampled over
    /// up to `sample` extents via `mincore` (a query: faults nothing in).
    fn sampled_residency(paths: &[PathBuf], jobs: &[BenchJob], sample: usize) -> f64 {
        let files: Vec<File> = paths.iter().map(|path| File::open(path).unwrap()).collect();
        let mut resident = 0usize;
        let mut total = 0usize;
        for job in jobs.iter().step_by((jobs.len() / sample).max(1)) {
            let start = job.abs / 4096 * 4096;
            let end = (job.abs + job.len as u64).div_ceil(4096) * 4096;
            let len = (end - start) as usize;
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    files[job.shard].as_raw_fd(),
                    start as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                continue;
            }
            let pages = len / 4096;
            let mut vec = vec![0u8; pages];
            if unsafe { libc::mincore(ptr, len, vec.as_mut_ptr()) } == 0 {
                resident += vec.iter().filter(|byte| **byte & 1 != 0).count();
                total += pages;
            }
            unsafe { libc::munmap(ptr, len) };
        }
        if total == 0 {
            0.0
        } else {
            resident as f64 / total as f64
        }
    }

    /// Ignored real-shard spot check: io_uring reads must match the mmap view
    /// byte for byte on a couple dozen randomly-picked expert slices spread
    /// across all 5 shards.
    #[test]
    #[ignore = "needs ~/.hi/models/glm-5.2-reap50/ (169 GB, 5 shards)"]
    fn real_glm_shards_uring_matches_mmap_spot_check() {
        let Some(model) = glm_model_path() else {
            eprintln!("skipping: {GLM_FIRST_SHARD} not present");
            return;
        };
        let gguf = hi_gguf::GgufFile::open(&model).unwrap();
        assert_eq!(gguf.shard_count(), 5, "expected the 5-shard split");
        let jobs = glm_cold_pass_jobs(&gguf, 8); // 24 extents
        let reader = IoUringReader::open(&shard_paths(&gguf), DEFAULT_QD).unwrap();
        let extents: Vec<(usize, u64, usize)> = jobs
            .iter()
            .map(|job| (job.shard, job.abs, job.len))
            .collect();
        let shards_hit: std::collections::BTreeSet<usize> =
            jobs.iter().map(|job| job.shard).collect();
        eprintln!(
            "spot check: {} extents over shards {shards_hit:?}",
            jobs.len()
        );
        for (job, result) in jobs.iter().zip(reader.read_owned(&extents)) {
            let via_ring = result.unwrap();
            let view = gguf.tensor(&job.tensor).unwrap();
            let expected = &view.bytes[job.rel as usize..job.rel as usize + job.len];
            assert_eq!(
                via_ring.as_slice(),
                expected,
                "ring vs mmap for {} @{}",
                job.tensor,
                job.rel
            );
        }
    }

    /// The IO benchmark (no GPU): a realistic cold ensure-pass worth of
    /// scattered expert extents through every backend. Run alone:
    /// `cargo test -p hi-cuda --release bench_glm_expert_read_backends -- --ignored --nocapture`
    #[test]
    #[ignore = "IO benchmark; needs the real GLM shards and reads ~10 GB per backend"]
    fn bench_glm_expert_read_backends() {
        let Some(model) = glm_model_path() else {
            eprintln!("skipping: {GLM_FIRST_SHARD} not present");
            return;
        };
        let triples: usize = std::env::var("HI_BENCH_TRIPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600);
        let threads = 6; // READ_WORKERS in the production pool

        // Plan the pass with a short-lived GgufFile, then drop it so its
        // mmaps cannot keep the benchmark ranges cached.
        let (jobs, paths) = {
            let gguf = hi_gguf::GgufFile::open(&model).unwrap();
            (glm_cold_pass_jobs(&gguf, triples), shard_paths(&gguf))
        };
        let payload: u64 = jobs.iter().map(|job| job.len as u64).sum();
        let max_len = jobs.iter().map(|job| job.len).max().unwrap();
        let stride = tier_slot_stride(max_len);
        println!(
            "GLM-5.2 cold ensure-pass: {} triples x 3 = {} extents, {:.2} GiB payload, max extent {:.2} MiB, slot stride {:.2} MiB",
            triples,
            jobs.len(),
            payload as f64 / (1u64 << 30) as f64,
            max_len as f64 / (1 << 20) as f64,
            stride as f64 / (1 << 20) as f64,
        );
        let mut rows: Vec<(String, f64)> = Vec::new();
        let gibs = |secs: f64| payload as f64 / (1u64 << 30) as f64 / secs;

        // Device ceiling reference: one thread, sequential 16 MiB O_DIRECT
        // reads (what dd iflag=direct measures).
        {
            use std::os::unix::fs::{FileExt, OpenOptionsExt};
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&paths[0])
                .unwrap();
            let chunk = 16 << 20;
            let mut buf = AlignedBuf::new(chunk).unwrap();
            let started = std::time::Instant::now();
            let mut read = 0u64;
            for i in 0..256u64 {
                // SAFETY: buf is an exclusive, live chunk-sized allocation.
                let slice = unsafe { std::slice::from_raw_parts_mut(buf.ptr, buf.len) };
                read += file.read_at(slice, i * chunk as u64).unwrap() as u64;
            }
            let secs = started.elapsed().as_secs_f64();
            println!(
                "device ceiling reference (sequential 16 MiB O_DIRECT, 1 thread): {:.2} GiB/s",
                read as f64 / (1u64 << 30) as f64 / secs
            );
            let _ = &mut buf;
        }

        // (a) The mmap path as the pool runs it: MADV_RANDOM on the expert
        // tensors, per-extent WILLNEED, 6 copy threads, page cache dropped
        // per-range first (mincore-verified: pages another process maps
        // survive the fadvise, so the row label carries the real residency).
        {
            drop_job_ranges_from_page_cache(&paths, &jobs);
            let residency = sampled_residency(&paths, &jobs, 64);
            println!(
                "page-cache residency of the extents after per-range DONTNEED: {:.0}%",
                residency * 100.0
            );
            let gguf = hi_gguf::GgufFile::open(&model).unwrap();
            let tensors: std::collections::BTreeSet<&str> =
                jobs.iter().map(|job| job.tensor.as_str()).collect();
            for tensor in &tensors {
                gguf.advise_tensor(tensor, hi_gguf::GgufMemoryAdvice::Random)
                    .unwrap();
            }
            let queue = Mutex::new(jobs.iter());
            let started = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        loop {
                            let Some(job) = queue.lock().unwrap().next() else {
                                break;
                            };
                            let _ = gguf.advise_tensor_range(
                                &job.tensor,
                                job.rel,
                                job.len as u64,
                                hi_gguf::GgufMemoryAdvice::WillNeed,
                            );
                            let view = gguf.tensor(&job.tensor).unwrap();
                            let bytes =
                                view.bytes[job.rel as usize..job.rel as usize + job.len].to_vec();
                            // Defeat dead-copy elimination: without this the
                            // optimizer removes the memcpy and no page is
                            // ever faulted (a 256 GiB/s "read").
                            std::hint::black_box(&bytes);
                            assert_eq!(bytes.len(), job.len);
                        }
                    });
                }
            });
            let secs = started.elapsed().as_secs_f64();
            rows.push((
                format!(
                    "mmap+willneed ({threads} threads, {})",
                    if residency < 0.05 {
                        "cold".to_string()
                    } else {
                        format!("{:.0}% cached", residency * 100.0)
                    }
                ),
                secs,
            ));
            // Leave no benchmark residue in the page cache for later runs.
            drop(gguf);
            drop_job_ranges_from_page_cache(&paths, &jobs);
        }

        // (b) The O_DIRECT twin-fd thread pool as the pool runs it today.
        {
            let gguf = hi_gguf::GgufFile::open(&model).unwrap();
            let direct = gguf.direct_io_reader().unwrap();
            let queue = Mutex::new(jobs.iter());
            let started = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        loop {
                            let Some(job) = queue.lock().unwrap().next() else {
                                break;
                            };
                            let bytes = direct.read_range(job.shard, job.abs, job.len).unwrap();
                            std::hint::black_box(&bytes);
                            assert_eq!(bytes.len(), job.len);
                        }
                    });
                }
            });
            let secs = started.elapsed().as_secs_f64();
            rows.push((format!("O_DIRECT pread ({threads} threads)"), secs));
        }

        // (c) io_uring at QD 8/64/256, unregistered vs registered buffers,
        // zero-copy into stride-aligned slots (the ring_slot_dma layout).
        let arena = AlignedBuf::new(jobs.len() * stride).unwrap();
        println!(
            "slot arena: {:.2} GiB ({} slots)",
            arena.len as f64 / (1u64 << 30) as f64,
            jobs.len()
        );
        for qd in [8u32, 64, 256] {
            for registered in [false, true] {
                let mut reader = IoUringReader::open(&paths, qd).unwrap();
                assert_eq!(reader.queue_depth(), qd);
                let label = if registered {
                    if let Err(err) = reader.register_arena(arena.ptr, arena.len, stride) {
                        println!("io_uring qd={qd} registered: n/a ({err:#})");
                        continue;
                    }
                    "registered  "
                } else {
                    "unregistered"
                };
                let ring_jobs: Vec<UringJob> = jobs
                    .iter()
                    .enumerate()
                    .map(|(slot, job)| UringJob {
                        shard: job.shard,
                        offset: job.abs,
                        len: job.len,
                        dest: Some(SlotDest {
                            // SAFETY: disjoint stride-sized slots of one arena.
                            ptr: unsafe { arena.ptr.add(slot * stride) },
                            cap: stride,
                        }),
                    })
                    .collect();
                let started = std::time::Instant::now();
                let mut completed = 0usize;
                // SAFETY: 4 KiB-aligned disjoint slots, each >= the aligned
                // span of its extent (stride covers the largest extent).
                unsafe {
                    reader.read_batch(&ring_jobs, |_, outcome| {
                        outcome.unwrap();
                        completed += 1;
                    })
                }
                .unwrap();
                assert_eq!(completed, ring_jobs.len());
                let secs = started.elapsed().as_secs_f64();
                rows.push((format!("io_uring qd={qd:<3} {label}"), secs));
            }
        }

        println!("\n{:<38} {:>8} {:>9}", "backend", "GiB/s", "wall");
        for (label, secs) in &rows {
            println!("{label:<38} {:>8.2} {:>8.1}s", gibs(*secs), secs);
        }
    }

    /// A batch far larger than the queue depth completes correctly (the drive
    /// loop refills the SQ as completions land).
    #[test]
    fn batches_larger_than_queue_depth_complete() {
        let dir = scratch_dir("depth");
        let path = dir.join("shard.bin");
        let file_len = 512 * 1024;
        let expected = write_pattern_file(&path, file_len);
        let Some(reader) = open_or_skip(&[path], 8) else {
            return;
        };
        assert_eq!(reader.queue_depth(), 8);
        let extents: Vec<(usize, u64, usize)> = (0..100)
            .map(|i| (0usize, (i * 4700) as u64, 1500usize))
            .collect();
        let results = reader.read_owned(&extents);
        for (&(_, offset, len), result) in extents.iter().zip(&results) {
            assert_eq!(
                result.as_ref().unwrap().as_slice(),
                &expected[offset as usize..offset as usize + len],
                "read at {offset}"
            );
        }
    }
}
