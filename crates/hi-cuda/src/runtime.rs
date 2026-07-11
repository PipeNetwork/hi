use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct CudaRuntimeInfo {
    pub device_count: i32,
    pub runtime_version: i32,
    pub driver_version: i32,
}

#[derive(Debug)]
pub struct CudaRuntime {
    info: CudaRuntimeInfo,
}

impl CudaRuntime {
    pub fn probe() -> Result<Self> {
        imp::probe()
    }

    pub fn info(&self) -> &CudaRuntimeInfo {
        &self.info
    }

    #[cfg(feature = "native-cuda")]
    fn new(info: CudaRuntimeInfo) -> Self {
        Self { info }
    }
}

#[cfg(feature = "native-cuda")]
pub use imp::{
    Cublas, CublasLt, CudaGraph, DeviceBuffer, GemmDType, GraphExec, Stream, check_last_error,
    free_memory_bytes, multiprocessor_count, set_capture_active,
};

#[cfg(not(feature = "native-cuda"))]
mod imp {
    use anyhow::{Result, bail};

    use super::CudaRuntime;

    pub fn probe() -> Result<CudaRuntime> {
        bail!(
            "hi-cuda was built without native-cuda support; rebuild with the hi-cuda/native-cuda feature and a CUDA Toolkit installation"
        )
    }
}

#[cfg(feature = "native-cuda")]
mod imp {
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_ulonglong, c_void};
    use std::ptr;
    use std::sync::OnceLock;

    use anyhow::{Result, anyhow, bail};

    use super::{CudaRuntime, CudaRuntimeInfo};

    type CudaError = c_int;
    type CublasStatus = c_int;
    type CudaStream = *mut c_void;
    type CudaGraphRaw = *mut c_void;
    type CudaGraphExecRaw = *mut c_void;
    type CublasHandle = *mut c_void;
    type CublasLtHandle = *mut c_void;
    type CudaDataType = c_int;
    type CublasComputeType = c_int;
    type CublasGemmAlgo = c_int;

    #[link(name = "cudart")]
    unsafe extern "C" {
        fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
        fn cudaRuntimeGetVersion(version: *mut c_int) -> CudaError;
        fn cudaDriverGetVersion(version: *mut c_int) -> CudaError;
        fn cudaGetErrorString(error: CudaError) -> *const c_char;
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
        fn cudaFree(ptr: *mut c_void) -> CudaError;
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> CudaError;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int)
        -> CudaError;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: c_int,
            stream: CudaStream,
        ) -> CudaError;
        fn cudaMemsetAsync(
            dst: *mut c_void,
            value: c_int,
            count: usize,
            stream: CudaStream,
        ) -> CudaError;
        fn cudaGetLastError() -> CudaError;
        fn cudaStreamCreate(stream: *mut CudaStream) -> CudaError;
        fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
        fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
        fn cudaStreamBeginCapture(stream: CudaStream, mode: c_int) -> CudaError;
        fn cudaStreamEndCapture(stream: CudaStream, graph: *mut CudaGraphRaw) -> CudaError;
        fn cudaGraphInstantiate(
            exec: *mut CudaGraphExecRaw,
            graph: CudaGraphRaw,
            flags: c_ulonglong,
        ) -> CudaError;
        fn cudaGraphLaunch(exec: CudaGraphExecRaw, stream: CudaStream) -> CudaError;
        fn cudaGraphExecDestroy(exec: CudaGraphExecRaw) -> CudaError;
        fn cudaGraphDestroy(graph: CudaGraphRaw) -> CudaError;
    }

    #[link(name = "cublas")]
    unsafe extern "C" {
        fn cublasCreate_v2(handle: *mut CublasHandle) -> CublasStatus;
        fn cublasDestroy_v2(handle: CublasHandle) -> CublasStatus;
        fn cublasSetStream_v2(handle: CublasHandle, stream: CudaStream) -> CublasStatus;
        fn cublasSgemm_v2(
            handle: CublasHandle,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: *const f32,
            c: *mut f32,
            ldc: c_int,
        ) -> CublasStatus;
        fn cublasGemmEx(
            handle: CublasHandle,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const c_void,
            a: *const c_void,
            a_type: CudaDataType,
            lda: c_int,
            b: *const c_void,
            b_type: CudaDataType,
            ldb: c_int,
            beta: *const c_void,
            c: *mut c_void,
            c_type: CudaDataType,
            ldc: c_int,
            compute_type: CublasComputeType,
            algo: CublasGemmAlgo,
        ) -> CublasStatus;
    }

    #[link(name = "cublasLt")]
    unsafe extern "C" {
        fn cublasLtCreate(handle: *mut CublasLtHandle) -> CublasStatus;
        fn cublasLtDestroy(handle: CublasLtHandle) -> CublasStatus;
    }

    pub fn probe() -> Result<CudaRuntime> {
        let mut device_count = 0;
        cuda_check(
            unsafe { cudaGetDeviceCount(&mut device_count) },
            "cudaGetDeviceCount",
        )?;
        if device_count <= 0 {
            bail!("no CUDA devices reported by cudaGetDeviceCount");
        }
        let mut runtime_version = 0;
        cuda_check(
            unsafe { cudaRuntimeGetVersion(&mut runtime_version) },
            "cudaRuntimeGetVersion",
        )?;
        let mut driver_version = 0;
        cuda_check(
            unsafe { cudaDriverGetVersion(&mut driver_version) },
            "cudaDriverGetVersion",
        )?;
        Ok(CudaRuntime::new(CudaRuntimeInfo {
            device_count,
            runtime_version,
            driver_version,
        }))
    }

    // Caching device allocator for the decode hot path. `cudaFree` is a
    // *synchronizing* call, so the per-op alloc/free of transient buffers
    // (~1.3k free() per decoded token on a 3B model) drains the stream at every
    // op boundary and leaves the GPU idle ~44% of decode wall — nsys measured
    // ~22s in cudaFree over a decode trace. Instead of freeing a small buffer we
    // return it to a size-keyed free list and reuse it on the next same-size
    // alloc, eliminating both the cudaMalloc and the synchronizing cudaFree.
    //
    // Safe because hi runs all device ops on the model's single stream: a reused
    // buffer's next use is enqueued after its previous use on that stream, so
    // stream ordering already guarantees the prior op completes first (the
    // cudaFree sync was redundant for correctness, only load-bearing for freeing
    // memory). Only buffers <= POOL_MAX_BYTES are pooled so large, rare prefill
    // temporaries (dequantized weights, seq*hidden activations) still return
    // their memory immediately and can't bloat the resident set on an 8GB card.
    // Opt out with HI_CUDA_NO_BUF_POOL (falls back to plain cudaMalloc/cudaFree).
    //
    // The pool is THREAD-LOCAL, which is what makes the single-stream argument
    // hold: each model's ops are enqueued from one thread onto that model's
    // stream, so recycling within a thread stays within one stream's ordering
    // domain. A process-global pool broke this — with several models on separate
    // threads (the test suite; any future multi-model server), thread B could
    // recycle-and-write a buffer whose reads were still queued on thread A's
    // independent stream. Streams have no cross-ordering, so B's memset/kernel
    // writes raced A's in-flight work; with an otherwise-idle GPU the in-flight
    // window is microseconds and the race almost never fired, but with any
    // co-tenant GPU load our kernels queue for milliseconds and it fired
    // constantly (nondeterministic wrong values across the parity suite).
    // Residual caveat: a buffer whose last use was enqueued from thread A but
    // that is dropped on thread B recycles into B's pool without ordering — no
    // current code path moves buffers between threads mid-flight (models are
    // built, used, and dropped on their owning thread).
    const POOL_MAX_BYTES: usize = 4 * 1024 * 1024;

    fn buffer_pool_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("HI_CUDA_NO_BUF_POOL").is_err())
    }

    // During CUDA graph capture, cudaMalloc/cudaFree are illegal on the captured stream.
    // While this flag is set the pool never falls through to cudaMalloc (alloc errors
    // instead — the pool must be pre-warmed) and never cudaFrees on drop (every freed
    // buffer is retained in the pool regardless of size). Set via `set_capture_active`
    // around a capture window.
    fn capture_flag() -> &'static std::sync::atomic::AtomicBool {
        static FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        &FLAG
    }

    pub fn set_capture_active(active: bool) {
        capture_flag().store(active, std::sync::atomic::Ordering::SeqCst);
    }

    fn capture_active() -> bool {
        capture_flag().load(std::sync::atomic::Ordering::SeqCst)
    }

    // Frees every pooled pointer when its thread exits so short-lived threads
    // (tests, one-off workers) return their device memory.
    struct ThreadBufferPool(std::cell::RefCell<HashMap<usize, Vec<usize>>>);

    impl Drop for ThreadBufferPool {
        fn drop(&mut self) {
            for (_, slots) in self.0.borrow_mut().drain() {
                for addr in slots {
                    // Ignore errors: at process exit the context may already be gone.
                    let _ = unsafe { cudaFree(addr as *mut c_void) };
                }
            }
        }
    }

    thread_local! {
        static BUFFER_POOL: ThreadBufferPool =
            ThreadBufferPool(std::cell::RefCell::new(HashMap::new()));
    }

    fn buffer_pool_take(bytes: usize) -> Option<*mut c_void> {
        // During capture, pool any size (a matching buffer may have been retained on a
        // capture-time free); otherwise only pool <= POOL_MAX_BYTES.
        if bytes == 0 || (bytes > POOL_MAX_BYTES && !capture_active()) || !buffer_pool_enabled() {
            return None;
        }
        BUFFER_POOL
            .try_with(|pool| {
                pool.0
                    .borrow_mut()
                    .get_mut(&bytes)
                    .and_then(|slots| slots.pop())
                    .map(|addr| addr as *mut c_void)
            })
            .ok()
            .flatten()
    }

    fn buffer_pool_return(ptr: *mut c_void, bytes: usize) -> bool {
        // Retain every buffer (any size) while capturing so drop never cudaFrees.
        if bytes == 0 || (bytes > POOL_MAX_BYTES && !capture_active()) || !buffer_pool_enabled() {
            return false;
        }
        // try_with: if this drop runs during the thread's TLS teardown the pool is
        // gone — report un-pooled so the caller cudaFrees directly.
        BUFFER_POOL
            .try_with(|pool| {
                pool.0
                    .borrow_mut()
                    .entry(bytes)
                    .or_default()
                    .push(ptr as usize);
            })
            .is_ok()
    }

    pub struct DeviceBuffer {
        ptr: *mut c_void,
        bytes: usize,
    }

    impl DeviceBuffer {
        pub fn alloc(bytes: usize) -> Result<Self> {
            if let Some(ptr) = buffer_pool_take(bytes) {
                return Ok(Self { ptr, bytes });
            }
            if capture_active() {
                // cudaMalloc is illegal during graph capture. The pool wasn't warmed for
                // this size — surface it so the caller falls back to eager execution.
                bail!(
                    "CUDA graph capture needs a {bytes}-byte buffer the pool has not warmed; \
                     cannot cudaMalloc during capture"
                );
            }
            let mut ptr = ptr::null_mut();
            cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
            Ok(Self { ptr, bytes })
        }

        pub fn as_mut_ptr(&self) -> *mut c_void {
            self.ptr
        }

        pub fn as_ptr(&self) -> *const c_void {
            self.ptr.cast_const()
        }

        pub fn bytes(&self) -> usize {
            self.bytes
        }

        pub fn copy_from_host<T>(&self, data: &[T]) -> Result<()> {
            let bytes = checked_slice_bytes(data)?;
            self.require_capacity(bytes)?;
            cuda_check(
                unsafe {
                    cudaMemcpy(
                        self.ptr,
                        data.as_ptr().cast(),
                        bytes,
                        CudaMemcpyKind::HostToDevice as c_int,
                    )
                },
                "cudaMemcpy(host_to_device)",
            )
        }

        /// Async device-to-device copy of `len` bytes from `src[src_offset..]` into
        /// `self[dst_offset..]` on `stream`. Used to slice/scatter row ranges of an
        /// activation tensor for chunked processing.
        pub fn copy_device_range(
            &self,
            dst_offset: usize,
            src: &DeviceBuffer,
            src_offset: usize,
            len: usize,
            stream: &Stream,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            self.require_capacity(dst_offset.saturating_add(len))?;
            src.require_capacity(src_offset.saturating_add(len))?;
            cuda_check(
                unsafe {
                    cudaMemcpyAsync(
                        (self.ptr as *mut u8).add(dst_offset).cast(),
                        (src.ptr as *const u8).add(src_offset).cast(),
                        len,
                        CudaMemcpyKind::DeviceToDevice as c_int,
                        stream.raw,
                    )
                },
                "cudaMemcpyAsync(device_to_device)",
            )
        }

        pub fn copy_from_host_async<T>(&self, data: &[T], stream: &Stream) -> Result<()> {
            let bytes = checked_slice_bytes(data)?;
            self.require_capacity(bytes)?;
            cuda_check(
                unsafe {
                    cudaMemcpyAsync(
                        self.ptr,
                        data.as_ptr().cast(),
                        bytes,
                        CudaMemcpyKind::HostToDevice as c_int,
                        stream.raw,
                    )
                },
                "cudaMemcpyAsync(host_to_device)",
            )
        }

        pub fn copy_to_host<T: Default + Copy>(&self, len: usize) -> Result<Vec<T>> {
            let bytes = len
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| anyhow!("host copy byte length overflows usize"))?;
            self.require_capacity(bytes)?;
            let mut out = vec![T::default(); len];
            cuda_check(
                unsafe {
                    cudaMemcpy(
                        out.as_mut_ptr().cast(),
                        self.ptr.cast_const(),
                        bytes,
                        CudaMemcpyKind::DeviceToHost as c_int,
                    )
                },
                "cudaMemcpy(device_to_host)",
            )?;
            Ok(out)
        }

        pub fn copy_to_host_offset<T: Default + Copy>(
            &self,
            offset: usize,
            len: usize,
        ) -> Result<Vec<T>> {
            let element_size = std::mem::size_of::<T>();
            let byte_offset = offset
                .checked_mul(element_size)
                .ok_or_else(|| anyhow!("host copy byte offset overflows usize"))?;
            let bytes = len
                .checked_mul(element_size)
                .ok_or_else(|| anyhow!("host copy byte length overflows usize"))?;
            let end = byte_offset
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("host copy byte range overflows usize"))?;
            self.require_capacity(end)?;
            let mut out = vec![T::default(); len];
            cuda_check(
                unsafe {
                    cudaMemcpy(
                        out.as_mut_ptr().cast(),
                        self.ptr.cast::<u8>().add(byte_offset).cast_const().cast(),
                        bytes,
                        CudaMemcpyKind::DeviceToHost as c_int,
                    )
                },
                "cudaMemcpy(device_to_host_offset)",
            )?;
            Ok(out)
        }

        pub fn copy_to_host_async<T: Default + Copy>(
            &self,
            len: usize,
            stream: &Stream,
        ) -> Result<Vec<T>> {
            let bytes = len
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| anyhow!("host copy byte length overflows usize"))?;
            self.require_capacity(bytes)?;
            let mut out = vec![T::default(); len];
            cuda_check(
                unsafe {
                    cudaMemcpyAsync(
                        out.as_mut_ptr().cast(),
                        self.ptr.cast_const(),
                        bytes,
                        CudaMemcpyKind::DeviceToHost as c_int,
                        stream.raw,
                    )
                },
                "cudaMemcpyAsync(device_to_host)",
            )?;
            stream.synchronize()?;
            Ok(out)
        }

        pub fn memset_zero_async(&self, stream: &Stream) -> Result<()> {
            cuda_check(
                unsafe { cudaMemsetAsync(self.ptr, 0, self.bytes, stream.raw) },
                "cudaMemsetAsync",
            )
        }

        fn require_capacity(&self, bytes: usize) -> Result<()> {
            if bytes > self.bytes {
                bail!(
                    "CUDA device buffer copy of {bytes} bytes exceeds allocation of {} bytes",
                    self.bytes
                );
            }
            Ok(())
        }
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                if buffer_pool_return(self.ptr, self.bytes) {
                    return;
                }
                let _ = unsafe { cudaFree(self.ptr) };
            }
        }
    }

    pub struct Stream {
        raw: CudaStream,
    }

    impl Stream {
        pub fn create() -> Result<Self> {
            let mut raw = ptr::null_mut();
            cuda_check(unsafe { cudaStreamCreate(&mut raw) }, "cudaStreamCreate")?;
            Ok(Self { raw })
        }

        pub fn synchronize(&self) -> Result<()> {
            cuda_check(
                unsafe { cudaStreamSynchronize(self.raw) },
                "cudaStreamSynchronize",
            )
        }

        pub fn as_raw(&self) -> *mut c_void {
            self.raw
        }

        /// Begin capturing every op enqueued on this stream into a graph instead of
        /// executing them. ThreadLocal mode records only this thread's ops. During capture
        /// no synchronizing CUDA call (cudaMalloc/cudaFree/cudaMemcpy/stream sync) may be
        /// issued on this stream, so the buffer pool must be pre-warmed (see
        /// `set_capture_active`).
        pub fn begin_capture(&self) -> Result<()> {
            const CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
            cuda_check(
                unsafe { cudaStreamBeginCapture(self.raw, CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
                "cudaStreamBeginCapture",
            )
        }

        /// Finish capture and return the recorded graph.
        pub fn end_capture(&self) -> Result<CudaGraph> {
            let mut graph: CudaGraphRaw = ptr::null_mut();
            cuda_check(
                unsafe { cudaStreamEndCapture(self.raw, &mut graph) },
                "cudaStreamEndCapture",
            )?;
            Ok(CudaGraph { raw: graph })
        }

        /// Begin capture and return a guard that MUST be `end()`ed to obtain the graph. If the
        /// guard is dropped without `end()` — e.g. an operation between begin and end returned an
        /// error via `?` — its `Drop` calls `end_capture` so the stream leaves capture mode.
        /// Otherwise a mid-capture failure (like the capture-safe pool refusing to grow) would
        /// leave the stream capturing, and every later launch on it (including the eager fallback)
        /// would be silently recorded instead of run.
        pub fn begin_capture_scoped(&self) -> Result<CaptureGuard<'_>> {
            self.begin_capture()?;
            Ok(CaptureGuard {
                stream: self,
                armed: true,
            })
        }
    }

    /// RAII guard for a stream capture (see `Stream::begin_capture_scoped`).
    pub struct CaptureGuard<'a> {
        stream: &'a Stream,
        armed: bool,
    }

    impl CaptureGuard<'_> {
        /// Finish the capture and return the recorded graph.
        pub fn end(mut self) -> Result<CudaGraph> {
            self.armed = false;
            self.stream.end_capture()
        }
    }

    impl Drop for CaptureGuard<'_> {
        fn drop(&mut self) {
            if self.armed {
                // Restore the stream to non-capturing on an early/error exit, and consume any
                // pending capture error (an illegal in-capture op invalidates the capture and
                // leaves the error latched) so the eager fallback's next launch isn't poisoned.
                let _ = self.stream.end_capture();
                clear_last_error();
            }
        }
    }

    /// A recorded (but not yet executable) CUDA graph.
    pub struct CudaGraph {
        raw: CudaGraphRaw,
    }

    impl CudaGraph {
        /// Instantiate into an executable graph that can be replayed cheaply.
        pub fn instantiate(&self) -> Result<GraphExec> {
            let mut exec: CudaGraphExecRaw = ptr::null_mut();
            cuda_check(
                unsafe { cudaGraphInstantiate(&mut exec, self.raw, 0) },
                "cudaGraphInstantiate",
            )?;
            Ok(GraphExec { raw: exec })
        }
    }

    impl Drop for CudaGraph {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                let _ = unsafe { cudaGraphDestroy(self.raw) };
            }
        }
    }

    /// An instantiated, replayable CUDA graph.
    pub struct GraphExec {
        raw: CudaGraphExecRaw,
    }

    impl GraphExec {
        /// Replay the whole captured op sequence on `stream` as a single launch.
        pub fn launch(&self, stream: &Stream) -> Result<()> {
            cuda_check(
                unsafe { cudaGraphLaunch(self.raw, stream.raw) },
                "cudaGraphLaunch",
            )
        }
    }

    impl Drop for GraphExec {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                let _ = unsafe { cudaGraphExecDestroy(self.raw) };
            }
        }
    }

    impl Drop for Stream {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                let _ = unsafe { cudaStreamDestroy(self.raw) };
            }
        }
    }

    pub struct Cublas {
        raw: CublasHandle,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum GemmDType {
        F32,
        F16,
        BF16,
    }

    impl GemmDType {
        fn element_size(self) -> usize {
            match self {
                Self::F32 => 4,
                Self::F16 | Self::BF16 => 2,
            }
        }

        fn cuda_data_type(self) -> CudaDataType {
            match self {
                Self::F32 => CudaDataTypeKind::R32F as CudaDataType,
                Self::F16 => CudaDataTypeKind::R16F as CudaDataType,
                Self::BF16 => CudaDataTypeKind::R16BF as CudaDataType,
            }
        }
    }

    impl Cublas {
        pub fn create() -> Result<Self> {
            let mut raw = ptr::null_mut();
            cublas_check(unsafe { cublasCreate_v2(&mut raw) }, "cublasCreate")?;
            Ok(Self { raw })
        }

        pub fn set_stream(&self, stream: &Stream) -> Result<()> {
            cublas_check(
                unsafe { cublasSetStream_v2(self.raw, stream.raw) },
                "cublasSetStream",
            )
        }

        pub fn as_raw(&self) -> *mut c_void {
            self.raw
        }

        pub fn matmul_f32_row_major(
            &self,
            a: &DeviceBuffer,
            b: &DeviceBuffer,
            out: &DeviceBuffer,
            rows: usize,
            cols: usize,
            inner: usize,
        ) -> Result<()> {
            if rows == 0 || cols == 0 || inner == 0 {
                bail!("cuBLAS matmul dimensions must be non-zero");
            }
            a.require_capacity(checked_bytes::<f32>(rows, inner)?)?;
            b.require_capacity(checked_bytes::<f32>(inner, cols)?)?;
            out.require_capacity(checked_bytes::<f32>(rows, cols)?)?;
            let rows = checked_cublas_dim(rows, "rows")?;
            let cols = checked_cublas_dim(cols, "cols")?;
            let inner = checked_cublas_dim(inner, "inner")?;
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS is column-major. Row-major C[M,N] = A[M,K] * B[K,N]
            // is equivalent to column-major C^T[N,M] = B^T[N,K] * A^T[K,M].
            cublas_check(
                unsafe {
                    cublasSgemm_v2(
                        self.raw,
                        CublasOperation::None as c_int,
                        CublasOperation::None as c_int,
                        cols,
                        rows,
                        inner,
                        &alpha,
                        b.as_ptr().cast(),
                        cols,
                        a.as_ptr().cast(),
                        inner,
                        &beta,
                        out.as_mut_ptr().cast(),
                        cols,
                    )
                },
                "cublasSgemm(row_major)",
            )
        }

        pub fn matmul_mixed_row_major(
            &self,
            a: &DeviceBuffer,
            b: &DeviceBuffer,
            out: &DeviceBuffer,
            rows: usize,
            cols: usize,
            inner: usize,
            input_dtype: GemmDType,
        ) -> Result<()> {
            if rows == 0 || cols == 0 || inner == 0 {
                bail!("cuBLAS matmul dimensions must be non-zero");
            }
            a.require_capacity(checked_matrix_bytes(
                rows,
                inner,
                input_dtype.element_size(),
            )?)?;
            b.require_capacity(checked_matrix_bytes(
                inner,
                cols,
                input_dtype.element_size(),
            )?)?;
            out.require_capacity(checked_bytes::<f32>(rows, cols)?)?;
            let rows = checked_cublas_dim(rows, "rows")?;
            let cols = checked_cublas_dim(cols, "cols")?;
            let inner = checked_cublas_dim(inner, "inner")?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let input_type = input_dtype.cuda_data_type();

            cublas_check(
                unsafe {
                    cublasGemmEx(
                        self.raw,
                        CublasOperation::None as c_int,
                        CublasOperation::None as c_int,
                        cols,
                        rows,
                        inner,
                        (&alpha as *const f32).cast(),
                        b.as_ptr(),
                        input_type,
                        cols,
                        a.as_ptr(),
                        input_type,
                        inner,
                        (&beta as *const f32).cast(),
                        out.as_mut_ptr(),
                        CudaDataTypeKind::R32F as CudaDataType,
                        cols,
                        CublasComputeTypeKind::F32 as CublasComputeType,
                        CublasGemmAlgoKind::DefaultTensorOp as CublasGemmAlgo,
                    )
                },
                "cublasGemmEx(mixed_row_major)",
            )
        }

        pub fn matmul_mixed_rhs_transposed_row_major(
            &self,
            lhs: &DeviceBuffer,
            rhs: &DeviceBuffer,
            out: &DeviceBuffer,
            rows: usize,
            cols: usize,
            inner: usize,
            lhs_dtype: GemmDType,
            rhs_dtype: GemmDType,
        ) -> Result<()> {
            if rows == 0 || cols == 0 || inner == 0 {
                bail!("cuBLAS matmul dimensions must be non-zero");
            }
            if lhs_dtype != rhs_dtype {
                bail!(
                    "cuBLAS projection GEMM currently requires matching lhs/rhs dtypes, got {lhs_dtype:?} and {rhs_dtype:?}"
                );
            }
            lhs.require_capacity(checked_matrix_bytes(rows, inner, lhs_dtype.element_size())?)?;
            rhs.require_capacity(checked_matrix_bytes(cols, inner, rhs_dtype.element_size())?)?;
            out.require_capacity(checked_bytes::<f32>(rows, cols)?)?;
            let rows = checked_cublas_dim(rows, "rows")?;
            let cols = checked_cublas_dim(cols, "cols")?;
            let inner = checked_cublas_dim(inner, "inner")?;
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // Computes row-major C[M,N] = lhs[M,K] * rhs[N,K]^T.
            // In cuBLAS column-major form this is C^T[N,M] = rhs[N,K] * lhs^T[K,M].
            cublas_check(
                unsafe {
                    cublasGemmEx(
                        self.raw,
                        CublasOperation::Transpose as c_int,
                        CublasOperation::None as c_int,
                        cols,
                        rows,
                        inner,
                        (&alpha as *const f32).cast(),
                        rhs.as_ptr(),
                        rhs_dtype.cuda_data_type(),
                        inner,
                        lhs.as_ptr(),
                        lhs_dtype.cuda_data_type(),
                        inner,
                        (&beta as *const f32).cast(),
                        out.as_mut_ptr(),
                        CudaDataTypeKind::R32F as CudaDataType,
                        cols,
                        CublasComputeTypeKind::F32 as CublasComputeType,
                        CublasGemmAlgoKind::DefaultTensorOp as CublasGemmAlgo,
                    )
                },
                "cublasGemmEx(mixed_rhs_transposed_row_major)",
            )
        }

        pub fn matmul_f32_rhs_transposed_row_major(
            &self,
            lhs: &DeviceBuffer,
            rhs: &DeviceBuffer,
            out: &DeviceBuffer,
            rows: usize,
            cols: usize,
            inner: usize,
        ) -> Result<()> {
            if rows == 0 || cols == 0 || inner == 0 {
                bail!("cuBLAS matmul dimensions must be non-zero");
            }
            lhs.require_capacity(checked_bytes::<f32>(rows, inner)?)?;
            rhs.require_capacity(checked_bytes::<f32>(cols, inner)?)?;
            out.require_capacity(checked_bytes::<f32>(rows, cols)?)?;
            let rows = checked_cublas_dim(rows, "rows")?;
            let cols = checked_cublas_dim(cols, "cols")?;
            let inner = checked_cublas_dim(inner, "inner")?;
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // Computes row-major C[M,N] = lhs[M,K] * rhs[N,K]^T.
            cublas_check(
                unsafe {
                    cublasSgemm_v2(
                        self.raw,
                        CublasOperation::Transpose as c_int,
                        CublasOperation::None as c_int,
                        cols,
                        rows,
                        inner,
                        &alpha,
                        rhs.as_ptr().cast(),
                        inner,
                        lhs.as_ptr().cast(),
                        inner,
                        &beta,
                        out.as_mut_ptr().cast(),
                        cols,
                    )
                },
                "cublasSgemm(f32_rhs_transposed_row_major)",
            )
        }
    }

    impl Drop for Cublas {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                let _ = unsafe { cublasDestroy_v2(self.raw) };
            }
        }
    }

    pub struct CublasLt {
        raw: CublasLtHandle,
    }

    impl CublasLt {
        pub fn create() -> Result<Self> {
            let mut raw = ptr::null_mut();
            cublas_check(unsafe { cublasLtCreate(&mut raw) }, "cublasLtCreate")?;
            Ok(Self { raw })
        }

        pub fn as_raw(&self) -> *mut c_void {
            self.raw
        }
    }

    impl Drop for CublasLt {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                let _ = unsafe { cublasLtDestroy(self.raw) };
            }
        }
    }

    /// Streaming-multiprocessor count of the active device, cached after the first
    /// query. Used to size grid splits (e.g. flash-decoding split-KV). Falls back
    /// to the GB10's 48 if the attribute query fails.
    pub fn multiprocessor_count() -> usize {
        use std::sync::OnceLock;
        static COUNT: OnceLock<usize> = OnceLock::new();
        *COUNT.get_or_init(|| {
            // cudaDevAttrMultiProcessorCount == 16.
            let mut value: c_int = 0;
            let code = unsafe { cudaDeviceGetAttribute(&mut value, 16, 0) };
            if code == 0 && value > 0 {
                value as usize
            } else {
                48
            }
        })
    }

    /// Free device memory (bytes) currently available on the active CUDA device.
    /// Used to decide whether an FP16 weight copy fits before converting.
    pub fn free_memory_bytes() -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        cuda_check(
            unsafe { cudaMemGetInfo(&mut free, &mut total) },
            "cudaMemGetInfo",
        )?;
        Ok(free)
    }

    fn cuda_check(code: CudaError, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                "unknown CUDA error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(anyhow!(
            "{operation} failed with CUDA error {code}: {message}"
        ))
    }

    fn cublas_check(code: CublasStatus, operation: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(anyhow!("{operation} failed with cuBLAS status {code}"))
        }
    }

    #[repr(i32)]
    enum CudaMemcpyKind {
        HostToDevice = 1,
        DeviceToHost = 2,
        DeviceToDevice = 3,
    }

    #[repr(i32)]
    enum CublasOperation {
        None = 0,
        Transpose = 1,
    }

    #[repr(i32)]
    enum CudaDataTypeKind {
        R32F = 0,
        R16F = 2,
        R16BF = 14,
    }

    #[repr(i32)]
    enum CublasComputeTypeKind {
        F32 = 68,
    }

    #[repr(i32)]
    enum CublasGemmAlgoKind {
        DefaultTensorOp = 99,
    }

    fn checked_slice_bytes<T>(data: &[T]) -> Result<usize> {
        data.len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow!("slice byte length overflows usize"))
    }

    fn checked_bytes<T>(rows: usize, cols: usize) -> Result<usize> {
        checked_matrix_bytes(rows, cols, std::mem::size_of::<T>())
    }

    fn checked_matrix_bytes(rows: usize, cols: usize, element_size: usize) -> Result<usize> {
        rows.checked_mul(cols)
            .and_then(|elements| elements.checked_mul(element_size))
            .ok_or_else(|| anyhow!("matrix byte length overflows usize"))
    }

    fn checked_cublas_dim(value: usize, label: &str) -> Result<c_int> {
        c_int::try_from(value)
            .map_err(|_| anyhow!("cuBLAS {label} dimension {value} exceeds c_int"))
    }

    pub fn check_last_error(operation: &str) -> Result<()> {
        cuda_check(unsafe { cudaGetLastError() }, operation)
    }

    /// Consume and discard the pending CUDA error (`cudaGetLastError` clears it). Used after a
    /// failed graph capture: an illegal in-capture op (e.g. a blocking device->host copy on a
    /// model whose decode isn't fully device-side) leaves `cudaErrorStreamCaptureInvalidated`
    /// pending, which would otherwise fail the very next launch on the eager fallback path.
    pub fn clear_last_error() {
        unsafe {
            cudaGetLastError();
        }
    }
}
