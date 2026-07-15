//! On-demand expert slab loader for MLX MoE expert streaming.
//!
//! When the planning layer (`expert_stream`) decides to stream, the routed-expert
//! tensors stay out of the resident `HashMap<String, Array>`. Instead, this
//! module reads individual expert slabs from the safetensors shards on demand
//! and caches them in a bounded LRU pool.
//!
//! On Apple Silicon unified memory, a slab read from disk lands in host RAM
//! that the GPU can access directly — no H2D copy needed (unlike CUDA's pinned
//! arena → DMA pipeline). We use MLX's copying `from_raw_data` constructor
//! (`mlx_array_new_data`) to turn slab bytes into a temporary `Array`; the copy
//! stays within shared RAM and the source `Vec<u8>` is freed immediately after.
//! This avoids the lazy-evaluation use-after-free hazard that a zero-copy
//! `mlx_array_new_data_managed` approach would introduce (MLX defers execution
//! until `eval`, so a managed buffer must outlive any pending graph — hard to
//! guarantee with an LRU that may evict before eval).
//!
//! The pool is keyed by `(layer, projection, expert_idx)` and capped to a
//! byte budget. On a miss, the slab is read from disk; on eviction, the oldest
//! slab is dropped. Pool health (hits/misses/evictions) is tracked for the
//! `/health` endpoint and load-time logging.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use mlx_rs::{Array, Dtype};

use crate::expert_stream::ExpertStreamPlan;

// ─── POSIX AIO (macOS) ──────────────────────────────────────────────────────
// macOS doesn't have io_uring, but it does have POSIX AIO (aio_read,
// aio_suspend, lio_listio). We use lio_listio to submit a batch of reads
// and wait for all to complete — exactly the "prefetch all 8 experts for
// this layer" pattern. The libc crate (already a dependency) exposes all
// the symbols we need on macOS.

#[repr(C)]
#[derive(Clone, Copy)]
struct AioCb {
    aio_fildes: libc::c_int,
    aio_offset: libc::off_t,
    aio_buf: *mut libc::c_void,
    aio_nbytes: libc::size_t,
    aio_reqprio: libc::c_int,
    aio_sigevent: libc::sigevent,
    aio_lio_opcode: libc::c_int,
}

unsafe extern "C" {
    fn lio_listio(
        mode: libc::c_int,
        list: *const *mut AioCb,
        nent: libc::c_int,
        sig: *mut libc::sigevent,
    ) -> libc::c_int;
    fn aio_return(aiocbp: *mut AioCb) -> libc::ssize_t;
    fn aio_error(aiocbp: *const AioCb) -> libc::c_int;
    // `aio_suspend` blocks until at least one of the listed AIO requests
    // completes (or the timeout fires). On x86_64 macOS the symbol is
    // `aio_suspend$UNIX2003`; arm64 uses the plain name. We declare it against
    // our local `AioCb` (layout-compatible with `libc::aiocb`) so the existing
    // externs stay self-consistent.
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"), link_name = "aio_suspend$UNIX2003")]
    fn aio_suspend(
        list: *const *const AioCb,
        nent: libc::c_int,
        timeout: *const libc::timespec,
    ) -> libc::c_int;
}

const LIO_WAIT: libc::c_int = 2; // LIO_WAIT — block until all complete
const LIO_NOWAIT: libc::c_int = 1; // LIO_NOWAIT — return immediately
const LIO_READ: libc::c_int = 0; // LIO_READ

/// A pending POSIX AIO read request.
struct AioRequest {
    aiocb: AioCb,
    buffer: Vec<u8>,
}

// SAFETY: AioRequest contains raw pointers (in AioCb/sigevent) for the kernel's
// use. We only access them from the thread that holds the pool lock, and the
// kernel completes AIO independently of any thread. The pointers point to the
// request's own buffer (owned by the same struct), so there are no cross-thread
// aliasing issues.
unsafe impl Send for AioRequest {}

impl AioRequest {
    /// Create an async read of `len` bytes from `fd` at `offset`.
    fn new(fd: libc::c_int, offset: u64, len: usize) -> Self {
        let mut buffer = vec![0u8; len];
        let aiocb = AioCb {
            aio_fildes: fd,
            aio_offset: offset as libc::off_t,
            aio_buf: buffer.as_mut_ptr() as *mut libc::c_void,
            aio_nbytes: len,
            aio_reqprio: 0,
            aio_sigevent: unsafe { std::mem::zeroed() },
            aio_lio_opcode: LIO_READ,
        };
        AioRequest { aiocb, buffer }
    }

    /// Take the buffer (valid only after the read completes).
    #[allow(dead_code)] // kept for future direct-buffer consumers
    fn into_buffer(self) -> Vec<u8> {
        // Ensure the AIO request is complete before taking the buffer.
        self.buffer
    }
}

/// Submit a batch of AIO reads and wait for all to complete. Returns the
/// buffers in the same order as the requests. Falls back to synchronous
/// reads if AIO fails (e.g. on older macOS or certain filesystems).
fn aio_batch_read(requests: &mut [AioRequest]) -> Result<Vec<Vec<u8>>> {
    aio_batch_read_impl(requests, true)
}

/// Submit a batch of AIO reads without waiting (LIO_NOWAIT). The caller must
/// later call `aio_wait` on the returned handles to collect the buffers.
/// This enables cross-layer pipelining: submit reads for layer N, do layer
/// N-1's compute, then wait for layer N's reads.
fn aio_batch_read_async(requests: &mut [AioRequest]) -> Result<()> {
    aio_batch_read_impl(requests, false)?;
    Ok(())
}

fn aio_batch_read_impl(requests: &mut [AioRequest], wait: bool) -> Result<Vec<Vec<u8>>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let ptrs: Vec<*mut AioCb> = requests.iter_mut().map(|r| &mut r.aiocb as *mut AioCb).collect();

    let mode = if wait { LIO_WAIT } else { LIO_NOWAIT };
    let rc = unsafe {
        lio_listio(
            mode,
            ptrs.as_ptr(),
            ptrs.len() as libc::c_int,
            std::ptr::null_mut(),
        )
    };

    if rc != 0 {
        // AIO failed — fall back to synchronous pread per request.
        return sync_fallback(requests);
    }

    if !wait {
        // Non-blocking: return empty (caller will wait later via aio_wait).
        return Ok(Vec::new());
    }

    // Check each request for errors and collect buffers.
    let mut results = Vec::with_capacity(requests.len());
    for req in requests.iter_mut() {
        let err = unsafe { aio_error(&req.aiocb as *const AioCb) };
        if err != 0 {
            return sync_fallback(requests);
        }
        let n = unsafe { aio_return(&mut req.aiocb as *mut AioCb) };
        if n < 0 || n as usize != req.buffer.len() {
            return sync_fallback(requests);
        }
    }
    for req in requests.iter_mut() {
        results.push(std::mem::take(&mut req.buffer));
    }
    Ok(results)
}

/// Wait for a batch of async AIO reads to complete and collect their buffers.
///
/// Uses `aio_suspend` to block the calling thread until at least one request
/// in the batch completes, then reaps all finished requests and repeats until
/// every request is done. This is the proper blocking wait (the macOS
/// equivalent of io_uring's completion reaping) — the previous implementation
/// busy-polled `aio_error` with `thread::yield_now()`, burning a whole core.
/// `aio_suspend` lets the kernel park the thread until I/O lands, so the CPU
/// is free for the GPU compute that overlaps with the prefetch.
fn aio_wait(requests: &mut [AioRequest]) -> Result<Vec<Vec<u8>>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    // Block on aio_suspend until every request has left EINPROGRESS. We rebuild
    // the pending-pointer list each iteration (completed requests drop out).
    // A 100 ms timeout guards against a stalled request hanging decode forever;
    // on timeout we loop and re-check, so a slow-but-progressing batch still
    // completes normally.
    let timeout = libc::timespec {
        tv_sec: 0,
        tv_nsec: 100_000_000, // 100 ms
    };
    loop {
        // Collect pointers to still-in-flight requests.
        let pending: Vec<*const AioCb> = requests
            .iter()
            .filter(|r| unsafe { aio_error(&r.aiocb as *const AioCb) } == libc::EINPROGRESS)
            .map(|r| &r.aiocb as *const AioCb)
            .collect();
        if pending.is_empty() {
            break;
        }
        // Block until at least one of the pending requests completes (or the
        // timeout fires). EINTR is benign — just loop and re-check.
        let rc = unsafe {
            aio_suspend(
                pending.as_ptr(),
                pending.len() as libc::c_int,
                &timeout as *const libc::timespec,
            )
        };
        if rc != 0 {
            let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if err != libc::EINTR && err != libc::EAGAIN {
                // Unexpected failure — fall back to sync reads for the whole batch.
                return sync_fallback(requests);
            }
        }
    }

    // Collect buffers, checking for errors.
    let mut results = Vec::with_capacity(requests.len());
    for req in requests.iter_mut() {
        let err = unsafe { aio_error(&req.aiocb as *const AioCb) };
        if err != 0 {
            // Fall back to synchronous read for this one.
            let mut buf = vec![0u8; req.aiocb.aio_nbytes];
            let n = unsafe {
                libc::pread(
                    req.aiocb.aio_fildes,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    req.aiocb.aio_nbytes,
                    req.aiocb.aio_offset,
                )
            };
            if n < 0 || n as usize != buf.len() {
                bail!("aio_wait fallback pread failed");
            }
            results.push(buf);
        } else {
            let n = unsafe { aio_return(&mut req.aiocb as *mut AioCb) };
            if n < 0 || n as usize != req.buffer.len() {
                bail!("aio_wait: short read");
            }
            results.push(std::mem::take(&mut req.buffer));
        }
    }
    Ok(results)
}

/// Synchronous fallback: pread each request one-by-one.
fn sync_fallback(requests: &mut [AioRequest]) -> Result<Vec<Vec<u8>>> {
    let mut results = Vec::with_capacity(requests.len());
    for req in requests.iter_mut() {
        let mut buf = vec![0u8; req.aiocb.aio_nbytes];
        let fd = req.aiocb.aio_fildes;
        let offset = req.aiocb.aio_offset;
        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                req.aiocb.aio_nbytes,
                offset,
            )
        };
        if n < 0 || n as usize != buf.len() {
            bail!(
                "synchronous pread failed: fd={fd} offset={offset} len={} rc={n}",
                buf.len()
            );
        }
        results.push(buf);
    }
    Ok(results)
}

// ─── Mmap shard ─────────────────────────────────────────────────────────────
// Each shard file is memory-mapped once (for layout reads + fallback) AND
// opened with F_NOCACHE (macOS's O_DIRECT equivalent) for the AIO slab reads.
// F_NOCACHE tells the kernel to not pollute the page cache with the slab
// bytes — for a 390 GB MoE on 64 GB RAM, page-cache thrash is the dominant
// cost. The mmap is kept for madvise hints and the sync fallback; the AIO
// path reads through the F_NOCACHE fd so the DMA bypasses the cache.
//
// `HI_MLX_EXPERT_NOCACHE=0` forces the old mmap-buffered path (useful for
// warm page-cache benchmarks or filesystems where F_NOCACHE isn't supported).

struct MmapShard {
    addr: *const u8,
    len: usize,
    fd: libc::c_int,
    /// A second fd opened with F_NOCACHE for direct (page-cache-bypassing)
    /// slab reads. `None` if F_NOCACHE is unavailable or disabled by env.
    direct_fd: Option<libc::c_int>,
}

unsafe impl Send for MmapShard {}
unsafe impl Sync for MmapShard {}

impl MmapShard {
    fn open(path: &std::path::Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening shard {}", path.display()))?;
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        let len = file.metadata()?.len() as usize;

        // F_NOCACHE on the mmap fd: don't pollute the kernel page cache from
        // mmap faults either — we have our own pool.
        unsafe {
            let nocache: libc::c_int = 1;
            libc::fcntl(fd, libc::F_NOCACHE, &nocache as *const libc::c_int as *mut libc::c_void);
        }

        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            bail!(
                "mmap failed for {} (len={len}): {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }

        // Leak the File — its fd stays open for the lifetime of the mapping.
        std::mem::forget(file);

        // Open a second fd with F_NOCACHE for direct pread/AIO slab reads.
        // This is the macOS equivalent of Linux O_DIRECT: the kernel won't
        // cache the read pages, so a 390 GB MoE doesn't evict the working set.
        // Disabled via HI_MLX_EXPERT_NOCACHE=0 (e.g. for warm-cache benchmarks).
        let direct_fd = if std::env::var("HI_MLX_EXPERT_NOCACHE").as_deref() == Ok("0") {
            None
        } else {
            match File::open(path) {
                Ok(dfile) => {
                    let dfd = std::os::unix::io::AsRawFd::as_raw_fd(&dfile);
                    unsafe {
                        let nocache: libc::c_int = 1;
                        libc::fcntl(
                            dfd,
                            libc::F_NOCACHE,
                            &nocache as *const libc::c_int as *mut libc::c_void,
                        );
                    }
                    // Leak this fd too — it lives as long as the shard.
                    std::mem::forget(dfile);
                    Some(dfd)
                }
                Err(_) => None, // non-fatal: fall back to the mmap fd
            }
        };

        Ok(MmapShard {
            addr: addr as *const u8,
            len,
            fd,
            direct_fd,
        })
    }

    /// Read `len` bytes at `offset` from the mapped region. Returns a slice
    /// that's valid for the lifetime of this MmapShard.
    fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let start = offset as usize;
        let end = start + len;
        if end > self.len {
            bail!(
                "mmap read out of bounds: offset={start} len={len} map_len={}",
                self.len
            );
        }
        // Copy from the mapped region into a Vec — the pool owns the bytes
        // and may evict them later.
        unsafe {
            let src = std::slice::from_raw_parts(self.addr.add(start), len);
            Ok(src.to_vec())
        }
    }

    /// Hint the kernel to prefetch pages for this byte range.
    fn prefetch(&self, offset: u64, len: usize) {
        let start = offset as usize;
        let end = (start + len).min(self.len);
        if start >= end {
            return;
        }
        unsafe {
            libc::madvise(
                self.addr.add(start) as *mut _,
                end - start,
                libc::MADV_WILLNEED,
            );
        }
    }

    /// Get the raw fd for POSIX AIO / pread. Prefers the F_NOCACHE direct fd
    /// (page-cache bypass) when available; falls back to the mmap fd.
    fn raw_fd(&self) -> libc::c_int {
        self.direct_fd.unwrap_or(self.fd)
    }

    /// Whether slab reads through `raw_fd()` bypass the page cache.
    fn is_direct(&self) -> bool {
        self.direct_fd.is_some()
    }
}

impl Drop for MmapShard {
    fn drop(&mut self) {
        if !self.addr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.addr as *mut _, self.len);
            }
        }
        // Close the direct fd (the mmap fd is closed by the kernel when the
        // leaked File's fd is munmap'd — actually both fds are leaked; close
        // the direct one explicitly here, the mmap fd is closed at process
        // exit. This is fine for a long-lived server.)
        if let Some(dfd) = self.direct_fd {
            unsafe {
                libc::close(dfd);
            }
        }
    }
}

/// A tensor's on-disk layout within a safetensors shard: the shard file path,
/// the byte offset of the tensor's data region within the file (past the
/// 8-byte length prefix + header), the full tensor shape, and its dtype.
#[derive(Clone, Debug)]
struct TensorLayout {
    shard_path: PathBuf,
    data_offset: u64,
    shape: Vec<i32>,
    dtype: Dtype,
    /// Total bytes of the tensor (`product(shape) × itemsize`).
    nbytes: u64,
}

/// The slab reader: holds mmap'd shard files and a map from tensor name to
/// its on-disk layout. Reading an expert slab is a memcpy from the mapped
/// region. Batch prefetch uses POSIX AIO (lio_listio) to issue all reads
/// in parallel, overlapping I/O with compute.
pub struct ExpertSlabReader {
    /// Mmap'd shard files, keyed by shard path.
    shards: HashMap<PathBuf, MmapShard>,
    /// Tensor name → layout.
    layouts: HashMap<String, TensorLayout>,
}

impl ExpertSlabReader {
    /// Build the reader from the stream plan's expert sources and the model
    /// root. Reads each shard's safetensors header once to extract per-tensor
    /// shape, dtype, and data offset. Each shard is mmap'd once and kept open.
    pub fn new(plan: &ExpertStreamPlan, model_root: &std::path::Path) -> Result<Self> {
        let mut layouts = HashMap::new();
        let mut shard_paths: std::collections::BTreeSet<PathBuf> = std::collections::BTreeSet::new();

        for src in &plan.sources {
            // Each tensor (weight/scales/biases) may live in a different shard.
            // Build (name, shard_file) pairs using the per-tensor shard info.
            let entries: Vec<(&str, &str)> = [
                (src.weight_name.as_str(), src.shard_file.as_str()),
            ]
            .into_iter()
            .chain(
                src.scales_name
                    .iter()
                    .zip(src.scales_shard.iter())
                    .map(|(n, s)| (n.as_str(), s.as_str())),
            )
            .chain(
                src.biases_name
                    .iter()
                    .zip(src.biases_shard.iter())
                    .map(|(n, s)| (n.as_str(), s.as_str())),
            )
            .collect();
            for (name, shard_file) in entries {
                if name.is_empty() || layouts.contains_key(name) {
                    continue;
                }
                let shard_path = model_root.join(shard_file);
                shard_paths.insert(shard_path.clone());
                let layout = read_tensor_layout(&shard_path, name)
                    .with_context(|| format!("reading layout for tensor {name}"))?;
                layouts.insert(name.to_string(), layout);
            }
        }

        // mmap all shard files once.
        let mut shards = HashMap::new();
        for path in shard_paths {
            let mmap = MmapShard::open(&path)
                .with_context(|| format!("mmap'ing shard {}", path.display()))?;
            shards.insert(path, mmap);
        }

        Ok(Self { shards, layouts })
    }

    /// Read the byte slab for expert `expert_idx` of the tensor `tensor_name`.
    /// The slab is the contiguous bytes of that expert's slice within the
    /// stacked `[num_experts, ...]` tensor. Returns the raw bytes plus the
    /// per-expert shape (the tensor shape with the leading `num_experts` dim
    /// removed) and the dtype.
    fn read_slab(&mut self, tensor_name: &str, expert_idx: u32) -> Result<SlabData> {
        let layout = self
            .layouts
            .get(tensor_name)
            .ok_or_else(|| anyhow::anyhow!("no layout for tensor {tensor_name}"))?;
        let slab = self.read_slab_from_layout(layout, expert_idx)?;
        Ok(slab)
    }

    /// Look up the per-expert shape + dtype for a tensor without reading bytes.
    /// Used by the RAM-tier promotion path (we already have the bytes from the
    /// tier; we just need the metadata to build a PoolEntry).
    fn layout_for(&self, tensor_name: &str, expert_idx: u32) -> Result<(Vec<i32>, Dtype)> {
        let layout = self
            .layouts
            .get(tensor_name)
            .ok_or_else(|| anyhow::anyhow!("no layout for tensor {tensor_name}"))?;
        if layout.shape.is_empty() {
            bail!("tensor has empty shape (scalar?)");
        }
        let num_experts = layout.shape[0] as u64;
        if expert_idx as u64 >= num_experts {
            bail!("expert {expert_idx} out of range (num_experts={num_experts})");
        }
        Ok((layout.shape[1..].to_vec(), layout.dtype))
    }

    /// Read a slab given a pre-resolved layout (used by batch prefetch).
    fn read_slab_from_layout(&self, layout: &TensorLayout, expert_idx: u32) -> Result<SlabData> {
        if layout.shape.is_empty() {
            bail!("tensor has empty shape (scalar?)");
        }
        let num_experts = layout.shape[0] as u64;
        if expert_idx as u64 >= num_experts {
            bail!(
                "expert {expert_idx} out of range (num_experts={num_experts})"
            );
        }
        let per_expert_shape: Vec<i32> = layout.shape[1..].to_vec();
        let per_expert_elems: u64 = per_expert_shape.iter().map(|&d| d as u64).product();
        let itemsize = dtype_itemsize(layout.dtype);
        let per_expert_bytes = per_expert_elems * itemsize;
        let slab_offset = layout.data_offset + (expert_idx as u64) * per_expert_bytes;

        let shard = self
            .shards
            .get(&layout.shard_path)
            .ok_or_else(|| anyhow::anyhow!("shard not open: {}", layout.shard_path.display()))?;
        let bytes = shard.read(slab_offset, per_expert_bytes as usize)?;

        Ok(SlabData {
            bytes,
            shape: per_expert_shape,
            dtype: layout.dtype,
        })
    }

    /// Prefetch a batch of slabs using POSIX AIO (lio_listio). All reads are
    /// submitted in parallel and we block until all complete. This is the
    /// key optimization: instead of 24 sequential reads per layer (8 experts
    /// × 3 projections), we issue all 24 in one batch and the SSD services
    /// them concurrently via its NVMe queue.
    ///
    /// Returns the slab data in the same order as the requests. Falls back
    /// to synchronous mmap reads if AIO is unavailable.
    fn prefetch_slabs(
        &self,
        requests: &[(String, u32)], // (tensor_name, expert_idx)
    ) -> Result<Vec<SlabData>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve layouts and compute read parameters.
        let mut layouts: Vec<&TensorLayout> = Vec::with_capacity(requests.len());
        let mut params: Vec<(libc::c_int, u64, usize)> = Vec::with_capacity(requests.len()); // (fd, offset, len)

        for (name, expert) in requests {
            let layout = self
                .layouts
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("no layout for tensor {name}"))?;
            let num_experts = layout.shape[0] as u64;
            if *expert as u64 >= num_experts {
                bail!("expert {expert} out of range for {name}");
            }
            let per_expert_shape: Vec<i32> = layout.shape[1..].to_vec();
            let per_expert_elems: u64 = per_expert_shape.iter().map(|&d| d as u64).product();
            let itemsize = dtype_itemsize(layout.dtype);
            let per_expert_bytes = per_expert_elems * itemsize;
            let slab_offset = layout.data_offset + (*expert as u64) * per_expert_bytes;

            let shard = self
                .shards
                .get(&layout.shard_path)
                .ok_or_else(|| anyhow::anyhow!("shard not open: {}", layout.shard_path.display()))?;

            // Issue madvise prefetch hint first (overlaps with AIO submission).
            shard.prefetch(slab_offset, per_expert_bytes as usize);

            layouts.push(layout);
            params.push((shard.raw_fd(), slab_offset, per_expert_bytes as usize));
        }

        // Build AIO requests.
        let mut aio_reqs: Vec<AioRequest> = params
            .iter()
            .map(|&(fd, off, len)| AioRequest::new(fd, off, len))
            .collect();

        // Submit batch and wait.
        let buffers = aio_batch_read(&mut aio_reqs)?;

        // Assemble SlabData from buffers.
        let mut results = Vec::with_capacity(requests.len());
        for (i, layout) in layouts.iter().enumerate() {
            let per_expert_shape: Vec<i32> = layout.shape[1..].to_vec();
            results.push(SlabData {
                bytes: buffers[i].clone(),
                shape: per_expert_shape,
                dtype: layout.dtype,
            });
        }
        Ok(results)
    }

    /// Submit a batch of slab reads asynchronously (LIO_NOWAIT). Returns the
    /// AIO requests + metadata needed to later collect the results via
    /// `prefetch_slabs_wait`. The caller should do other work (e.g. GPU
    /// matmuls) between submit and wait to overlap I/O with compute.
    fn prefetch_slabs_async(
        &self,
        requests: &[(String, u32)], // (tensor_name, expert_idx)
    ) -> Result<(Vec<AioRequest>, Vec<Vec<i32>>, Vec<Dtype>)> {
        if requests.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let mut layouts: Vec<&TensorLayout> = Vec::with_capacity(requests.len());
        let mut params: Vec<(libc::c_int, u64, usize)> = Vec::with_capacity(requests.len());
        let mut shapes: Vec<Vec<i32>> = Vec::with_capacity(requests.len());
        let mut dtypes: Vec<Dtype> = Vec::with_capacity(requests.len());

        for (name, expert) in requests {
            let layout = self
                .layouts
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("no layout for tensor {name}"))?;
            let num_experts = layout.shape[0] as u64;
            if *expert as u64 >= num_experts {
                bail!("expert {expert} out of range for {name}");
            }
            let per_expert_shape: Vec<i32> = layout.shape[1..].to_vec();
            let per_expert_elems: u64 = per_expert_shape.iter().map(|&d| d as u64).product();
            let itemsize = dtype_itemsize(layout.dtype);
            let per_expert_bytes = per_expert_elems * itemsize;
            let slab_offset = layout.data_offset + (*expert as u64) * per_expert_bytes;

            let shard = self
                .shards
                .get(&layout.shard_path)
                .ok_or_else(|| anyhow::anyhow!("shard not open: {}", layout.shard_path.display()))?;

            shard.prefetch(slab_offset, per_expert_bytes as usize);

            layouts.push(layout);
            shapes.push(per_expert_shape);
            dtypes.push(layout.dtype);
            params.push((shard.raw_fd(), slab_offset, per_expert_bytes as usize));
        }

        let mut aio_reqs: Vec<AioRequest> = params
            .iter()
            .map(|&(fd, off, len)| AioRequest::new(fd, off, len))
            .collect();

        // Submit non-blocking.
        aio_batch_read_async(&mut aio_reqs)?;

        Ok((aio_reqs, shapes, dtypes))
    }

    /// Wait for a previously submitted async prefetch to complete and collect
    /// the slab data.
    fn prefetch_slabs_wait(
        &self,
        aio_reqs: &mut Vec<AioRequest>,
        shapes: Vec<Vec<i32>>,
        dtypes: Vec<Dtype>,
    ) -> Result<Vec<SlabData>> {
        let buffers = aio_wait(aio_reqs)?;
        let mut results = Vec::with_capacity(buffers.len());
        for (i, buf) in buffers.into_iter().enumerate() {
            results.push(SlabData {
                bytes: buf,
                shape: shapes[i].clone(),
                dtype: dtypes[i],
            });
        }
        Ok(results)
    }
}

/// Raw bytes + shape + dtype for one expert slab, ready to be turned into an
/// MLX `Array`.
struct SlabData {
    bytes: Vec<u8>,
    shape: Vec<i32>,
    dtype: Dtype,
}

impl SlabData {
    /// Create an MLX `Array` from this slab's bytes. Uses `from_raw_data` which
    /// copies the bytes into MLX-owned memory — safe because the source `Vec`
    /// is consumed and the copy outlives it.
    fn to_array(&self) -> Array {
        // SAFETY: `from_raw_data` copies the data (via `mlx_array_new_data` →
        // `mlx_array_set_data`), so the source buffer only needs to live for
        // the duration of this call. We pass a pointer to our `Vec`'s buffer,
        // which is valid for the call's duration.
        unsafe {
            Array::from_raw_data(
                self.bytes.as_ptr() as *const std::ffi::c_void,
                &self.shape,
                self.dtype,
            )
        }
    }
}

/// A cache entry: the slab bytes for one (layer, projection, expert) triple.
struct PoolEntry {
    bytes: Vec<u8>,
    shape: Vec<i32>,
    dtype: Dtype,
    /// Byte size of this entry (for budget accounting).
    size: u64,
}

/// The LRU pool of expert slabs. Caches slab bytes keyed by
/// `(layer, projection, expert_idx)`. On a miss, reads from disk via the
/// `ExpertSlabReader`; on eviction, drops the oldest entry to stay within the
/// byte budget.
/// A pending async prefetch: the AIO requests have been submitted but not
/// yet collected. `prefetch_batch_wait` collects the results and inserts
/// them into the pool. This enables cross-layer pipelining.
struct PendingPrefetch {
    /// AIO requests in flight (buffers will be filled by the kernel).
    aio_requests: Vec<AioRequest>,
    /// Pool keys for each request, in the same order.
    keys: Vec<PoolKey>,
    /// Per-expert shape for each request (from the tensor layout).
    shapes: Vec<Vec<i32>>,
    /// Per-expert dtype for each request.
    dtypes: Vec<Dtype>,
}

// ─── Bounded pinned-RAM LRU tier ────────────────────────────────────────────
// A middle cache between disk and the ExpertPool. On a pool miss, we check
// this tier first: a hit returns the slab bytes with zero disk I/O. The tier
// is a simple LRU byte-budgeted HashMap. On Apple Silicon the "pinned" RAM is
// just host RAM (unified memory — the GPU reads it directly), so unlike CUDA
// there's no mlock/pin needed; the bytes just stay in a Vec that the pool
// copies into MLX Arrays.
//
// The tier absorbs the working-set reuse that the pool (which is smaller and
// also holds MLX Array copies) can't. Env: `HI_MLX_EXPERT_RAM_GB` overrides
// the auto budget (default: ~25% of the pool budget, clamped to 1–8 GiB).

struct RamTier {
    entries: HashMap<PoolKey, Vec<u8>>,
    lru_order: Vec<PoolKey>,
    used_bytes: u64,
    budget_bytes: u64,
    hits: u64,
    misses: u64,
}

impl RamTier {
    fn new(budget_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: Vec::new(),
            used_bytes: 0,
            budget_bytes,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a slab in the tier. Returns the bytes on a hit (and touches LRU).
    fn get(&mut self, key: &PoolKey) -> Option<Vec<u8>> {
        if self.entries.contains_key(key) {
            self.hits += 1;
            self.lru_order.retain(|k| k != key);
            self.lru_order.push(key.clone());
            self.entries.get(key).cloned()
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a slab into the tier, evicting oldest entries as needed.
    fn insert(&mut self, key: PoolKey, bytes: Vec<u8>) {
        if self.budget_bytes == 0 {
            return;
        }
        let size = bytes.len() as u64;
        // Evict oldest until it fits.
        while self.used_bytes + size > self.budget_bytes && !self.lru_order.is_empty() {
            let oldest = self.lru_order.remove(0);
            if let Some(entry) = self.entries.remove(&oldest) {
                self.used_bytes -= entry.len() as u64;
            }
        }
        self.used_bytes += size;
        self.entries.insert(key.clone(), bytes);
        self.lru_order.push(key);
    }

    /// Pre-warm a slab into the tier as a sticky entry (not evicted by LRU
    /// pressure during pre-warm; it becomes a normal entry after).
    fn prewarm(&mut self, key: PoolKey, bytes: Vec<u8>) {
        self.insert(key, bytes);
    }

    fn stats(&self) -> (u64, u64, u64, u64) {
        (self.hits, self.misses, self.used_bytes, self.budget_bytes)
    }
}

// ─── Expert usage learning ──────────────────────────────────────────────────
// Per-(layer, expert) selection counters persisted to
// `<model_dir>/.hi_expert_usage`. At construction the hottest experts are
// pre-warmed into the RAM tier. Profile-ranked placement beats heat-blind
// placement ~3x at equal capacity (measured on the CUDA side).

const USAGE_FILE_NAME: &str = ".hi_expert_usage";

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct ExpertUsage {
    /// (layer, expert) → selection count. Stored as a Vec of pairs because
    /// serde_json requires map keys to be strings (tuples can't be keys).
    counts: Vec<((u32, u32), u64)>,
    /// Number of decode passes recorded (for relative frequency).
    passes: u64,
}

impl ExpertUsage {
    fn load_or_new(model_path: &std::path::Path, _max_entries: usize) -> Self {
        let path = model_path.join(USAGE_FILE_NAME);
        match std::fs::read(&path) {
            Ok(data) => serde_json::from_slice(&data).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    fn save(&self, model_path: &std::path::Path) {
        let path = model_path.join(USAGE_FILE_NAME);
        if let Ok(data) = serde_json::to_vec(&self) {
            let _ = std::fs::write(&path, data);
        }
    }

    fn record_pass(&mut self, experts: impl IntoIterator<Item = (u32, u32)>) {
        self.passes += 1;
        for (layer, expert) in experts {
            if let Some(entry) = self.counts.iter_mut().find(|(k, _)| *k == (layer, expert)) {
                entry.1 += 1;
            } else {
                self.counts.push(((layer, expert), 1));
            }
        }
    }

    /// Hottest (layer, expert) first, ties broken by key order.
    fn ranked(&self) -> Vec<(u32, u32)> {
        let mut v: Vec<_> = self.counts.iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.into_iter().map(|(k, _)| *k).collect()
    }

    fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    fn len(&self) -> usize {
        self.counts.len()
    }
}

pub struct ExpertPool {
    reader: ExpertSlabReader,
    /// The byte budget. Zero means unbounded (for tests / forced-on with no
    /// explicit pool size).
    budget_bytes: u64,
    /// Current total bytes held in the pool.
    used_bytes: u64,
    /// LRU entries: key → entry. Insertion order = recency (oldest first).
    /// We use a `Vec` of keys alongside the map for simple LRU eviction.
    entries: HashMap<PoolKey, PoolEntry>,
    lru_order: Vec<PoolKey>,
    /// Health counters.
    hits: u64,
    misses: u64,
    evictions: u64,
    /// Pending async prefetches (up to `PIPELINE_DEPTH` in flight at once).
    /// This enables 2-deep cross-layer pipelining: layer N+1's reads are
    /// submitted while layer N is still computing.
    pending: VecDeque<PendingPrefetch>,
    /// Bounded pinned-RAM LRU tier (middle cache between disk and the pool).
    ram_tier: Option<RamTier>,
    /// Expert selection frequency history (persisted to `.hi_expert_usage`).
    usage: Option<ExpertUsage>,
    /// Model directory (for persisting usage stats).
    model_path: Option<PathBuf>,
}

/// How many async prefetch batches can be in flight simultaneously. 2 means
/// layer N+1's reads are submitted while layer N is still computing, hiding
/// disk latency behind GPU compute.
const PIPELINE_DEPTH: usize = 2;

/// The pool key: which expert slab we're caching.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    layer: u32,
    projection: &'static str,
    expert: u32,
    /// Which tensor within the projection group: "weight", "scales", or "biases".
    /// Each is cached independently.
    tensor_kind: &'static str,
}

impl ExpertPool {
    /// Create a new pool with the given byte budget. `budget_bytes = 0` means
    /// unbounded (no eviction). No RAM tier, no usage learning (for tests).
    pub fn new(reader: ExpertSlabReader, budget_bytes: u64) -> Self {
        Self {
            reader,
            budget_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru_order: Vec::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            pending: VecDeque::new(),
            ram_tier: None,
            usage: None,
            model_path: None,
        }
    }

    /// Create a new pool with a RAM tier and expert usage learning.
    /// `model_path` is the model directory (for `.hi_expert_usage` persistence).
    /// `ram_tier_bytes` of 0 disables the tier.
    pub fn new_with_tier(
        reader: ExpertSlabReader,
        budget_bytes: u64,
        ram_tier_bytes: u64,
        model_path: PathBuf,
    ) -> Self {
        let usage = ExpertUsage::load_or_new(&model_path, 4096);
        let ram_tier = if ram_tier_bytes > 0 {
            Some(RamTier::new(ram_tier_bytes))
        } else {
            None
        };
        Self {
            reader,
            budget_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru_order: Vec::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            pending: VecDeque::new(),
            ram_tier,
            usage: Some(usage),
            model_path: Some(model_path),
        }
    }

    /// Prefetch a batch of expert slabs in parallel using POSIX AIO. After
    /// this call returns, all requested slabs are in the pool (cache hits on
    /// subsequent `weight_array`/`scales_array`/`biases_array` calls).
    ///
    /// This is the core optimization: instead of fetching 24 slabs one-by-one
    /// (8 experts × 3 projections per layer), we issue all reads in one
    /// `lio_listio` batch so the SSD services them concurrently.
    ///
    /// `requests` is a list of `(layer, projection, expert, tensor_kind,
    /// tensor_name)` tuples. Slabs already in the pool are skipped (cache
    /// hits). Only misses are fetched via AIO.
    pub fn prefetch_batch(
        &mut self,
        requests: &[(u32, &'static str, u32, &'static str, String)],
    ) -> Result<()> {
        self.prefetch_batch_impl(requests, true)
    }

    /// Async variant: submits the AIO batch with `LIO_NOWAIT` and returns
    /// immediately. The reads complete in the background while the caller
    /// does other work (e.g. the previous layer's matmuls). The next call to
    /// `prefetch_batch` or `get_array` for any of these slabs will find them
    /// already cached (or will briefly wait if still in flight).
    ///
    /// This enables cross-layer pipelining: layer N issues its prefetch async,
    /// then does its compute; by the time layer N+1 runs, layer N's reads
    /// have completed in the background.
    pub fn prefetch_batch_async(
        &mut self,
        requests: &[(u32, &'static str, u32, &'static str, String)],
    ) -> Result<()> {
        self.prefetch_batch_impl(requests, false)
    }

    fn prefetch_batch_impl(
        &mut self,
        requests: &[(u32, &'static str, u32, &'static str, String)],
        wait: bool,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        // If the pipeline is full (PIPELINE_DEPTH pending), wait for the
        // oldest one to complete before submitting more.
        while self.pending.len() >= PIPELINE_DEPTH {
            self.prefetch_batch_wait_one()?;
        }

        // Partition into hits (pool or RAM tier) and misses (need to fetch).
        let mut misses: Vec<(String, u32)> = Vec::new();
        let mut miss_keys: Vec<PoolKey> = Vec::new();

        for (layer, projection, expert, tensor_kind, tensor_name) in requests {
            let key = PoolKey {
                layer: *layer,
                projection,
                expert: *expert,
                tensor_kind,
            };
            if self.entries.contains_key(&key) {
                self.hits += 1;
                self.lru_order.retain(|k| k != &key);
                self.lru_order.push(key);
            } else if let Some(tier) = &mut self.ram_tier {
                // Pool miss → check the RAM tier.
                if let Some(bytes) = tier.get(&key) {
                    self.hits += 1;
                    // Promote to the pool. We need the shape/dtype from the
                    // reader's layout (no disk read — metadata only).
                    match self.reader.layout_for(tensor_name, *expert) {
                        Ok((shape, dtype)) => {
                            let size = bytes.len() as u64;
                            self.evict_if_needed(size);
                            self.used_bytes += size;
                            self.entries.insert(
                                key.clone(),
                                PoolEntry {
                                    bytes,
                                    shape,
                                    dtype,
                                    size,
                                },
                            );
                            self.lru_order.push(key);
                        }
                        Err(_) => {
                            // Layout lookup failed — treat as a miss.
                            self.misses += 1;
                            misses.push((tensor_name.clone(), *expert));
                            miss_keys.push(key);
                        }
                    }
                } else {
                    self.misses += 1;
                    misses.push((tensor_name.clone(), *expert));
                    miss_keys.push(key);
                }
            } else {
                self.misses += 1;
                misses.push((tensor_name.clone(), *expert));
                miss_keys.push(key);
            }
        }

        if misses.is_empty() {
            return Ok(());
        }

        if wait {
            // Synchronous: submit + wait + insert.
            let slabs = self.reader.prefetch_slabs(&misses)?;
            for (key, slab) in miss_keys.into_iter().zip(slabs.into_iter()) {
                let size = slab.bytes.len() as u64;
                // Insert into the RAM tier (middle cache).
                if let Some(tier) = &mut self.ram_tier {
                    tier.insert(key.clone(), slab.bytes.clone());
                }
                self.evict_if_needed(size);
                self.used_bytes += size;
                self.entries.insert(
                    key.clone(),
                    PoolEntry {
                        bytes: slab.bytes,
                        shape: slab.shape,
                        dtype: slab.dtype,
                        size,
                    },
                );
                self.lru_order.push(key);
            }
        } else {
            // Async: submit (LIO_NOWAIT) and store pending state in the
            // pipeline queue (up to PIPELINE_DEPTH deep).
            let (aio_reqs, shapes, dtypes) = self.reader.prefetch_slabs_async(&misses)?;
            self.pending.push_back(PendingPrefetch {
                aio_requests: aio_reqs,
                keys: miss_keys,
                shapes,
                dtypes,
            });
        }

        Ok(())
    }

    /// Wait for ALL pending async prefetches to complete and insert their
    /// results into the pool. No-op if none are pending.
    pub fn prefetch_batch_wait(&mut self) -> Result<()> {
        while !self.pending.is_empty() {
            self.prefetch_batch_wait_one()?;
        }
        Ok(())
    }

    /// Wait for the OLDEST pending async prefetch to complete and insert its
    /// results into the pool. No-op if none are pending.
    fn prefetch_batch_wait_one(&mut self) -> Result<()> {
        let pending = self.pending.pop_front();
        if let Some(mut pending) = pending {
            let slabs = self.reader.prefetch_slabs_wait(
                &mut pending.aio_requests,
                pending.shapes,
                pending.dtypes,
            )?;
            for (key, slab) in pending.keys.into_iter().zip(slabs.into_iter()) {
                let size = slab.bytes.len() as u64;
                // Insert into the RAM tier (middle cache).
                if let Some(tier) = &mut self.ram_tier {
                    tier.insert(key.clone(), slab.bytes.clone());
                }
                self.evict_if_needed(size);
                self.used_bytes += size;
                self.entries.insert(
                    key.clone(),
                    PoolEntry {
                        bytes: slab.bytes,
                        shape: slab.shape,
                        dtype: slab.dtype,
                        size,
                    },
                );
                self.lru_order.push(key);
            }
        }
        Ok(())
    }

    /// Get the weight `Array` for expert `expert` of `(layer, projection)`.
    /// On a pool hit, returns the cached slab as an `Array` (a copy via
    /// `from_raw_data`). On a miss, reads from disk and caches.
    pub fn weight_array(&mut self, layer: u32, projection: &'static str, expert: u32, weight_name: &str) -> Result<Array> {
        self.get_array(layer, projection, expert, "weight", weight_name)
    }

    /// Get the scales `Array` for expert `expert` of `(layer, projection)`.
    pub fn scales_array(&mut self, layer: u32, projection: &'static str, expert: u32, scales_name: &str) -> Result<Array> {
        self.get_array(layer, projection, expert, "scales", scales_name)
    }

    /// Get the biases `Array` for expert `expert` of `(layer, projection)`.
    pub fn biases_array(&mut self, layer: u32, projection: &'static str, expert: u32, biases_name: &str) -> Result<Array> {
        self.get_array(layer, projection, expert, "biases", biases_name)
    }

    fn get_array(
        &mut self,
        layer: u32,
        projection: &'static str,
        expert: u32,
        tensor_kind: &'static str,
        tensor_name: &str,
    ) -> Result<Array> {
        let key = PoolKey {
            layer,
            projection,
            expert,
            tensor_kind,
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
            // Move to most-recently-used.
            self.lru_order.retain(|k| k != &key);
            self.lru_order.push(key.clone());
            let entry = &self.entries[&key];
            let slab = SlabData {
                bytes: entry.bytes.clone(),
                shape: entry.shape.clone(),
                dtype: entry.dtype,
            };
            return Ok(slab.to_array());
        }

        // Miss: check if there's a pending async prefetch that may contain
        // this slab. If so, wait for all pending and retry.
        if !self.pending.is_empty() {
            self.prefetch_batch_wait()?;
            if self.entries.contains_key(&key) {
                self.hits += 1;
                self.lru_order.retain(|k| k != &key);
                self.lru_order.push(key.clone());
                let entry = &self.entries[&key];
                let slab = SlabData {
                    bytes: entry.bytes.clone(),
                    shape: entry.shape.clone(),
                    dtype: entry.dtype,
                };
                return Ok(slab.to_array());
            }
        }

        // Miss: check the RAM tier before hitting disk.
        if let Some(tier) = &mut self.ram_tier {
            if let Some(bytes) = tier.get(&key) {
                self.hits += 1;
                if let Ok((shape, dtype)) = self.reader.layout_for(tensor_name, expert) {
                    let size = bytes.len() as u64;
                    self.evict_if_needed(size);
                    self.used_bytes += size;
                    self.entries.insert(
                        key.clone(),
                        PoolEntry {
                            bytes,
                            shape,
                            dtype,
                            size,
                        },
                    );
                    self.lru_order.push(key.clone());
                    let entry = &self.entries[&key];
                    let slab = SlabData {
                        bytes: entry.bytes.clone(),
                        shape: entry.shape.clone(),
                        dtype: entry.dtype,
                    };
                    return Ok(slab.to_array());
                }
            }
        }

        // Miss: read from disk.
        self.misses += 1;
        let slab = self.reader.read_slab(tensor_name, expert)?;
        let size = slab.bytes.len() as u64;
        // Insert into the RAM tier (middle cache).
        if let Some(tier) = &mut self.ram_tier {
            tier.insert(key.clone(), slab.bytes.clone());
        }
        self.evict_if_needed(size);
        self.used_bytes += size;
        self.entries.insert(
            key.clone(),
            PoolEntry {
                bytes: slab.bytes.clone(),
                shape: slab.shape.clone(),
                dtype: slab.dtype,
                size,
            },
        );
        self.lru_order.push(key);
        Ok(slab.to_array())
    }

    /// Evict oldest entries until `used_bytes + incoming <= budget_bytes`.
    /// No-op if budget is 0 (unbounded).
    fn evict_if_needed(&mut self, incoming: u64) {
        if self.budget_bytes == 0 {
            return;
        }
        while self.used_bytes + incoming > self.budget_bytes && !self.lru_order.is_empty() {
            let oldest = self.lru_order.remove(0);
            if let Some(entry) = self.entries.remove(&oldest) {
                self.used_bytes -= entry.size;
                self.evictions += 1;
            }
        }
    }

    /// Pool health counters: `(hits, misses, evictions, used_bytes, budget_bytes)`.
    pub fn health(&self) -> (u64, u64, u64, u64, u64) {
        (self.hits, self.misses, self.evictions, self.used_bytes, self.budget_bytes)
    }

    /// RAM tier stats: `(hits, misses, used_bytes, budget_bytes)`. None if no tier.
    pub fn ram_tier_stats(&self) -> Option<(u64, u64, u64, u64)> {
        self.ram_tier.as_ref().map(|t| t.stats())
    }

    /// Record which experts were selected in this decode pass, for usage
    /// learning. Call once per generated token (or per layer batch).
    pub fn record_expert_usage(&mut self, experts: impl IntoIterator<Item = (u32, u32)>) {
        if let Some(usage) = &mut self.usage {
            usage.record_pass(experts);
        }
    }

    /// Persist the expert usage history to `<model_dir>/.hi_expert_usage`.
    pub fn save_usage(&self) {
        if let (Some(usage), Some(path)) = (&self.usage, &self.model_path) {
            usage.save(path);
        }
    }

    /// Pre-warm the RAM tier with the hottest experts from the usage history.
    /// Reads their slabs from disk now (before the first token) so the first
    /// decode pass finds them in the tier. Returns the number of slabs warmed.
    pub fn prewarm_from_usage(&mut self, max_slabs: usize) -> Result<usize> {
        let ranked = match &self.usage {
            Some(u) if !u.is_empty() => u.ranked(),
            _ => return Ok(0),
        };
        let tier = match &mut self.ram_tier {
            Some(t) => t,
            None => return Ok(0),
        };
        let mut warmed = 0;
        for (layer, expert) in ranked.into_iter().take(max_slabs) {
            // We need the tensor name to read the slab. Try all 3 projections.
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let tensor_name = format!("model.layers.{layer}.mlp.switch_mlp.{proj}.weight");
                if let Ok(slab) = self.reader.read_slab(&tensor_name, expert) {
                    let key = PoolKey {
                        layer,
                        projection: proj_static(proj),
                        expert,
                        tensor_kind: "weight",
                    };
                    tier.prewarm(key, slab.bytes);
                    warmed += 1;
                }
            }
        }
        Ok(warmed)
    }
}

/// Convert a projection string slice to the corresponding `&'static str`.
fn proj_static(s: &str) -> &'static str {
    match s {
        "gate_proj" => "gate_proj",
        "up_proj" => "up_proj",
        "down_proj" => "down_proj",
        _ => "gate_proj", // unreachable in practice
    }
}

/// Read a single tensor's layout (data offset, shape, dtype) from a safetensors
/// shard header.
fn read_tensor_layout(shard_path: &std::path::Path, tensor_name: &str) -> Result<TensorLayout> {
    let mut file = File::open(shard_path)
        .with_context(|| format!("opening {}", shard_path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len);
    let header_len = usize::try_from(header_len).context("safetensors header too large")?;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parsing safetensors header {}", shard_path.display()))?;
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header is not an object"))?;
    let info = obj
        .get(tensor_name)
        .ok_or_else(|| anyhow::anyhow!("tensor {tensor_name} not found in {}", shard_path.display()))?;
    let dtype_str = info
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("tensor {tensor_name} missing dtype"))?;
    let shape_arr = info
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("tensor {tensor_name} missing shape"))?;
    let offsets = info
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("tensor {tensor_name} missing data_offsets"))?;
    if offsets.len() != 2 {
        bail!("tensor {tensor_name} has malformed data_offsets");
    }
    let start = offsets[0].as_u64().unwrap_or(0);
    let end = offsets[1].as_u64().unwrap_or(0);
    let nbytes = end.saturating_sub(start);

    let dtype = parse_safetensors_dtype(dtype_str)
        .with_context(|| format!("tensor {tensor_name} has unsupported dtype {dtype_str}"))?;
    let shape: Vec<i32> = shape_arr
        .iter()
        .map(|v| {
            v.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .unwrap_or(0)
        })
        .collect();

    // The data offset within the file is: 8 (length prefix) + header_len + start.
    let data_offset = 8 + header_len as u64 + start;

    Ok(TensorLayout {
        shard_path: shard_path.to_path_buf(),
        data_offset,
        shape,
        dtype,
        nbytes,
    })
}

/// Map a safetensors dtype string to an MLX `Dtype`.
fn parse_safetensors_dtype(s: &str) -> Result<Dtype> {
    Ok(match s {
        "F64" => Dtype::Float64,
        "F32" => Dtype::Float32,
        "F16" => Dtype::Float16,
        "BF16" => Dtype::Bfloat16,
        "I64" => Dtype::Int64,
        "I32" => Dtype::Int32,
        "I16" => Dtype::Int16,
        "I8" => Dtype::Int8,
        "U64" => Dtype::Uint64,
        "U32" => Dtype::Uint32,
        "U16" => Dtype::Uint16,
        "U8" => Dtype::Uint8,
        "BOOL" => Dtype::Bool,
        other => bail!("unsupported safetensors dtype: {other}"),
    })
}

/// Bytes per element for an MLX dtype.
fn dtype_itemsize(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Bool => 1,
        Dtype::Uint8 => 1,
        Dtype::Int8 => 1,
        Dtype::Uint16 => 2,
        Dtype::Int16 => 2,
        Dtype::Float16 => 2,
        Dtype::Bfloat16 => 2,
        Dtype::Uint32 => 4,
        Dtype::Int32 => 4,
        Dtype::Float32 => 4,
        Dtype::Uint64 => 8,
        Dtype::Int64 => 8,
        Dtype::Float64 => 8,
        Dtype::Complex64 => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_stream::ExpertSource;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mlx-expert-pool-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    /// Write a safetensors file with a stacked expert weight tensor
    /// `[num_experts, out, in]` of f32 data, where expert `e`'s slab is filled
    /// with the value `e as f32` so we can distinguish experts.
    fn write_stacked_expert_safetensors(
        path: &Path,
        num_experts: u32,
        out: u32,
        in_: u32,
        tensor_name: &str,
    ) {
        let per_expert_elems = (out as u64) * (in_ as u64);
        let total_elems = per_expert_elems * (num_experts as u64);
        let total_bytes = total_elems * 4;
        let shape = format!("[{num_experts},{out},{in_}]");
        let header = format!(
            r#"{{"__metadata__":{{"format":"pt"}},"{tensor_name}":{{"dtype":"F32","shape":{shape},"data_offsets":[0,{total_bytes}]}}}}"#
        );
        let header_bytes = header.as_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        // Fill expert slabs: expert e → all bytes = e as f32 (repeated).
        for e in 0..num_experts {
            let val = e as f32;
            for _ in 0..per_expert_elems {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        fs::write(path, data).unwrap();
    }

    fn make_plan_with_one_expert_group(
        dir: &Path,
        num_experts: u32,
        out: u32,
        in_: u32,
    ) -> ExpertStreamPlan {
        let tensor_name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        write_stacked_expert_safetensors(
            &dir.join("model.safetensors"),
            num_experts,
            out,
            in_,
            tensor_name,
        );
        // Write the index.
        let index = format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{"{tensor_name}":"model.safetensors"}}}}"#
        );
        fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
        // Build the plan manually (bypass WeightCatalog since we need a real
        // data-backed safetensors, not the zero-data test helper).
        let shard_path = dir.join("model.safetensors");
        let layout = read_tensor_layout(&shard_path, tensor_name).unwrap();
        let nbytes = layout.nbytes;
        ExpertStreamPlan {
            trunk_bytes: 0,
            expert_bytes: nbytes,
            sources: vec![ExpertSource {
                layer: 0,
                projection: "gate_proj",
                weight_name: tensor_name.to_string(),
                scales_name: None,
                biases_name: None,
                shard_file: "model.safetensors".to_string(),
                scales_shard: None,
                biases_shard: None,
                bytes: nbytes,
            }],
            moe_layers: 1,
        }
    }

    #[test]
    fn slab_reader_reads_correct_expert_bytes() {
        let dir = tempfile_path("slab-reader");
        fs::create_dir_all(&dir).unwrap();
        // 3 experts, 2×4 weight, f32 → 3×2×4×4 = 96 bytes.
        let plan = make_plan_with_one_expert_group(&dir, 3, 2, 4);
        let mut reader = ExpertSlabReader::new(&plan, &dir).unwrap();
        let slab = reader.read_slab("model.layers.0.mlp.switch_mlp.gate_proj.weight", 1).unwrap();
        assert_eq!(slab.shape, vec![2, 4]);
        assert_eq!(slab.dtype, Dtype::Float32);
        assert_eq!(slab.bytes.len(), 32); // 2×4×4
        // Expert 1 → all f32 values should be 1.0.
        let vals: Vec<f32> = slab
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(vals.iter().all(|&v| v == 1.0), "expert 1 should be all 1.0, got {vals:?}");
    }

    #[test]
    fn pool_caches_slabs_and_reports_hits_misses() {
        let dir = tempfile_path("pool-hits");
        fs::create_dir_all(&dir).unwrap();
        let plan = make_plan_with_one_expert_group(&dir, 4, 2, 4);
        let reader = ExpertSlabReader::new(&plan, &dir).unwrap();
        let mut pool = ExpertPool::new(reader, 0); // unbounded

        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        // First access: miss.
        let _a1 = pool.weight_array(0, "gate_proj", 2, name).unwrap();
        // Second access to same expert: hit.
        let _a2 = pool.weight_array(0, "gate_proj", 2, name).unwrap();
        // Different expert: miss.
        let _a3 = pool.weight_array(0, "gate_proj", 0, name).unwrap();

        let (hits, misses, evictions, _, _) = pool.health();
        assert_eq!(hits, 1, "one hit (second access to expert 2)");
        assert_eq!(misses, 2, "two misses (expert 2 then expert 0)");
        assert_eq!(evictions, 0, "unbounded pool → no evictions");
    }

    #[test]
    fn pool_evicts_oldest_when_over_budget() {
        let dir = tempfile_path("pool-evict");
        fs::create_dir_all(&dir).unwrap();
        // 4 experts × 2×4 f32 = 32 bytes each.
        let plan = make_plan_with_one_expert_group(&dir, 4, 2, 4);
        let reader = ExpertSlabReader::new(&plan, &dir).unwrap();
        // Budget: 64 bytes = 2 slabs. Third access should evict the first.
        let mut pool = ExpertPool::new(reader, 64);

        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        let _a = pool.weight_array(0, "gate_proj", 0, name).unwrap(); // miss, 32B used
        let _b = pool.weight_array(0, "gate_proj", 1, name).unwrap(); // miss, 64B used
        let _c = pool.weight_array(0, "gate_proj", 2, name).unwrap(); // miss, evict expert 0, 64B used

        let (hits, misses, evictions, used, budget) = pool.health();
        assert_eq!(misses, 3);
        assert_eq!(evictions, 1, "should evict expert 0 to make room for expert 2");
        assert_eq!(used, 64, "two slabs × 32 bytes");
        assert_eq!(budget, 64);
        assert_eq!(hits, 0);
    }

    #[test]
    fn pool_array_has_correct_shape_and_values() {
        let dir = tempfile_path("pool-array");
        fs::create_dir_all(&dir).unwrap();
        let plan = make_plan_with_one_expert_group(&dir, 3, 2, 4);
        let reader = ExpertSlabReader::new(&plan, &dir).unwrap();
        let mut pool = ExpertPool::new(reader, 0);

        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        let arr = pool.weight_array(0, "gate_proj", 2, name).unwrap();
        assert_eq!(arr.shape(), &[2, 4]);
        assert_eq!(arr.dtype(), Dtype::Float32);
        // Expert 2 → all values should be 2.0.
        let slice = arr.as_slice::<f32>();
        assert!(slice.iter().all(|&v| v == 2.0), "expert 2 should be all 2.0, got {slice:?}");
    }

    #[test]
    fn parse_safetensors_dtype_covers_common_types() {
        assert_eq!(parse_safetensors_dtype("F32").unwrap(), Dtype::Float32);
        assert_eq!(parse_safetensors_dtype("F16").unwrap(), Dtype::Float16);
        assert_eq!(parse_safetensors_dtype("BF16").unwrap(), Dtype::Bfloat16);
        assert_eq!(parse_safetensors_dtype("U8").unwrap(), Dtype::Uint8);
        assert_eq!(parse_safetensors_dtype("I8").unwrap(), Dtype::Int8);
        assert!(parse_safetensors_dtype("UNKNOWN").is_err());
    }

    #[test]
    fn dtype_itemsize_is_correct() {
        assert_eq!(dtype_itemsize(Dtype::Float32), 4);
        assert_eq!(dtype_itemsize(Dtype::Float16), 2);
        assert_eq!(dtype_itemsize(Dtype::Bfloat16), 2);
        assert_eq!(dtype_itemsize(Dtype::Uint8), 1);
        assert_eq!(dtype_itemsize(Dtype::Int64), 8);
    }

    #[test]
    fn ram_tier_absorbs_pool_evictions() {
        let dir = tempfile_path("ram-tier");
        fs::create_dir_all(&dir).unwrap();
        // 4 experts, 2×4 f32 → 32 bytes per slab. Pool budget 32 (1 slab),
        // RAM tier budget 64 (2 slabs).
        let plan = make_plan_with_one_expert_group(&dir, 4, 2, 4);
        let reader = ExpertSlabReader::new(&plan, &dir).unwrap();
        let mut pool = ExpertPool::new_with_tier(reader, 32, 64, dir.clone());

        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        // Access expert 0 → pool miss, disk read, enters pool + tier.
        pool.weight_array(0, "gate_proj", 0, name).unwrap();
        // Access expert 1 → pool miss (evicts 0 from pool), enters pool + tier.
        pool.weight_array(0, "gate_proj", 1, name).unwrap();
        // Access expert 0 again → pool miss, but RAM tier hit (no disk read).
        pool.weight_array(0, "gate_proj", 0, name).unwrap();
        let (hits, misses, _, _, _) = pool.health();
        eprintln!("DEBUG: hits={hits} misses={misses} tier_stats={:?}", pool.ram_tier_stats());
        assert!(hits >= 1, "should have at least 1 hit");
        // The tier should have recorded a hit for expert 0's promotion.
        let tier_stats = pool.ram_tier_stats().unwrap();
        assert!(tier_stats.0 >= 1, "RAM tier should have at least 1 hit, got {tier_stats:?}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn expert_usage_persists_and_ranks() {
        let dir = tempfile_path("usage");
        fs::create_dir_all(&dir).unwrap();
        let model_dir = dir.join("model-dir");
        fs::create_dir_all(&model_dir).unwrap();

        let mut usage = ExpertUsage::load_or_new(&model_dir, 60);
        usage.record_pass([(0u32, 1u32), (0, 2)]);
        usage.record_pass([(0, 1), (1, 3)]);
        usage.save(&model_dir);

        let reloaded = ExpertUsage::load_or_new(&model_dir, 60);
        assert_eq!(reloaded.len(), 3);
        assert_eq!(reloaded.passes, 2);
        // (0,1) selected twice → hottest.
        assert_eq!(reloaded.ranked().first(), Some(&(0, 1)));

        // Corrupt file is ignored, not fatal.
        fs::write(model_dir.join(USAGE_FILE_NAME), b"{not json").unwrap();
        let corrupt = ExpertUsage::load_or_new(&model_dir, 60);
        assert!(corrupt.is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pipeline_allows_two_pending_prefetches() {
        let dir = tempfile_path("pipeline");
        fs::create_dir_all(&dir).unwrap();
        // 6 experts so we can submit two distinct batches.
        let plan = make_plan_with_one_expert_group(&dir, 6, 2, 4);
        let reader = ExpertSlabReader::new(&plan, &dir).unwrap();
        let mut pool = ExpertPool::new(reader, 0); // unbounded

        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        // Submit batch 1 (experts 0,1) async.
        let reqs1 = vec![(0u32, "gate_proj", 0u32, "weight", name.to_string()),
                         (0, "gate_proj", 1, "weight", name.to_string())];
        pool.prefetch_batch_async(&reqs1).unwrap();
        assert_eq!(pool.pending.len(), 1);

        // Submit batch 2 (experts 2,3) async — should coexist (pipeline depth 2).
        let reqs2 = vec![(0, "gate_proj", 2, "weight", name.to_string()),
                         (0, "gate_proj", 3, "weight", name.to_string())];
        pool.prefetch_batch_async(&reqs2).unwrap();
        assert_eq!(pool.pending.len(), 2, "2-deep pipeline should hold both");

        // Wait for all — both batches should be in the pool.
        pool.prefetch_batch_wait().unwrap();
        assert_eq!(pool.pending.len(), 0);
        assert_eq!(pool.entries.len(), 4, "all 4 experts should be cached in the pool");

        // Now accessing them should be hits.
        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
        pool.weight_array(0, "gate_proj", 0, name).unwrap();
        pool.weight_array(0, "gate_proj", 3, name).unwrap();
        let (hits, _, _, _, _) = pool.health();
        assert!(hits >= 2, "cached experts should be hits, got {hits}");

        fs::remove_dir_all(&dir).unwrap();
    }
}
