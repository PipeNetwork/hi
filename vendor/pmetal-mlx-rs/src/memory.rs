//! Metal memory management for MLX.
//!
//! MLX uses a caching allocator for Metal buffers. When arrays are freed,
//! their underlying buffers are retained in a cache for reuse rather than
//! being returned to the system. This module exposes controls over that cache
//! and provides visibility into memory usage.
//!
//! # Memory Model
//!
//! - **Active memory**: Buffers currently held by live [`Array`](crate::Array) objects.
//! - **Cache memory**: Freed buffers retained for reuse (not returned to OS).
//! - **Peak memory**: High-water mark since process start or last [`reset_peak_memory`].
//! - **Memory limit**: Soft cap that triggers backpressure during graph evaluation.
//!   When active memory exceeds this limit, MLX blocks and waits for in-flight
//!   GPU operations to complete before scheduling more work.
//! - **Cache limit**: Maximum size of the buffer cache. Excess freed buffers are
//!   returned to the system immediately.
//!
//! # Example
//!
//! ```rust,ignore
//! use mlx_rs::memory;
//!
//! // Check current usage
//! let active = memory::get_active_memory();
//! let cached = memory::get_cache_memory();
//! println!("Active: {} bytes, Cached: {} bytes", active, cached);
//!
//! // Clear the buffer cache to free memory
//! memory::clear_cache();
//!
//! // Threshold-based clearing (like mlx-lm)
//! if memory::get_cache_memory() > 2 * 1024 * 1024 * 1024 {
//!     memory::clear_cache();
//! }
//! ```

/// Get the number of bytes currently allocated by MLX's Metal allocator.
///
/// This is "active" memory — buffers held by live arrays. Does **not** include
/// cached (freed but retained) buffers.
pub fn get_active_memory() -> usize {
    let mut res: usize = 0;
    // SAFETY: mlx_get_active_memory writes a single size_t through a valid pointer.
    unsafe { mlx_sys::mlx_get_active_memory(&mut res) };
    res
}

/// Get the peak memory usage since process start or last [`reset_peak_memory`].
pub fn get_peak_memory() -> usize {
    let mut res: usize = 0;
    // SAFETY: mlx_get_peak_memory writes a single size_t through a valid pointer.
    unsafe { mlx_sys::mlx_get_peak_memory(&mut res) };
    res
}

/// Get the number of bytes held in the buffer cache.
///
/// These are freed buffers retained for reuse. They count toward process RSS
/// but are available for reallocation without a system call.
pub fn get_cache_memory() -> usize {
    let mut res: usize = 0;
    // SAFETY: mlx_get_cache_memory writes a single size_t through a valid pointer.
    unsafe { mlx_sys::mlx_get_cache_memory(&mut res) };
    res
}

/// Get the current memory limit.
///
/// During graph evaluation, if active memory exceeds this limit, MLX blocks
/// and waits for in-flight GPU operations to complete before scheduling more
/// work. Default is 1.5× the device's recommended working set size.
pub fn get_memory_limit() -> usize {
    let mut res: usize = 0;
    // SAFETY: mlx_get_memory_limit writes a single size_t through a valid pointer.
    unsafe { mlx_sys::mlx_get_memory_limit(&mut res) };
    res
}

/// Set the memory limit for MLX's backpressure mechanism.
///
/// Returns the previous limit. Setting to 0 disables the limit.
pub fn set_memory_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    // SAFETY: mlx_set_memory_limit writes the old limit and sets the new one.
    unsafe { mlx_sys::mlx_set_memory_limit(&mut prev, limit) };
    prev
}

/// Set the maximum size of the buffer cache.
///
/// Freed buffers beyond this limit are returned to the system immediately.
/// Returns the previous cache limit. Setting to 0 disables caching entirely.
pub fn set_cache_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    // SAFETY: mlx_set_cache_limit writes the old limit and sets the new one.
    unsafe { mlx_sys::mlx_set_cache_limit(&mut prev, limit) };
    prev
}

/// Set the wired memory limit (macOS 15.0+).
///
/// Wired buffers are kept resident in GPU memory and not paged out.
/// Returns the previous wired limit. Setting to 0 (default) disables
/// residency tracking.
pub fn set_wired_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    // SAFETY: mlx_set_wired_limit writes the old limit and sets the new one.
    unsafe { mlx_sys::mlx_set_wired_limit(&mut prev, limit) };
    prev
}

/// Clear the Metal buffer cache, returning all cached buffers to the system.
///
/// This frees buffers that were retained for reuse after their owning arrays
/// were dropped. It does **not** affect buffers held by live arrays.
///
/// **When to call:**
/// - After a failed initialization (e.g., ANE fallback) before loading a new model
/// - After training completes to release memory
/// - When cache memory exceeds a threshold (like mlx-lm's `_clear_cache`)
///
/// **When NOT to call:**
/// - Between training steps (causes reallocation storms)
/// - Between epochs (same issue — buffers are immediately re-needed)
pub fn clear_cache() {
    // SAFETY: mlx_clear_cache has no preconditions and is idempotent.
    unsafe { mlx_sys::mlx_clear_cache() };
}

/// Reset the peak memory counter to zero.
///
/// After calling this, [`get_peak_memory`] tracks the new maximum from
/// this point forward.
pub fn reset_peak_memory() {
    // SAFETY: mlx_reset_peak_memory has no preconditions and is idempotent.
    unsafe { mlx_sys::mlx_reset_peak_memory() };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_queries_dont_crash() {
        let _active = get_active_memory();
        let _peak = get_peak_memory();
        let _cache = get_cache_memory();
        let _limit = get_memory_limit();
    }

    #[test]
    fn test_clear_cache() {
        clear_cache(); // idempotent, should not crash
    }

    #[test]
    fn test_reset_peak_memory() {
        reset_peak_memory();
    }

    #[test]
    fn test_set_memory_limit_roundtrip() {
        let original = get_memory_limit();
        let prev = set_memory_limit(1024 * 1024 * 1024); // 1 GB
        assert_eq!(prev, original);
        set_memory_limit(original); // restore
    }

    #[test]
    fn test_set_cache_limit_roundtrip() {
        let original = get_memory_limit(); // cache limit defaults to memory limit
        let prev = set_cache_limit(512 * 1024 * 1024); // 512 MB
        // Restore (use max of prev and original to avoid going below default)
        set_cache_limit(prev.max(original));
    }
}
