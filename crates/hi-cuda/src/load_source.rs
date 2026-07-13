//! Byte source for resident-weight model loads: buffered mmap (today's path)
//! or io_uring O_DIRECT bulk reads, chosen per model file.
//!
//! Loaders fault tensor bytes through the hi-gguf mmap — fine when the file
//! is warm in the page cache, ~0.7 GiB/s when cold (readahead-driven faults).
//! [`LoadByteSource`] swaps the byte source behind one helper: when the ring
//! is active, whole-tensor extents are read O_DIRECT at queue depth
//! ([`crate::expert_uring::IoUringReader::read_extent_chunked`], ~6.5 GiB/s
//! measured cold on this class of NVMe) into anonymous aligned memory —
//! which also sidesteps the slow cudaMemcpy-from-file-backed-pages path.
//!
//! Selection is tri-state via `HI_CUDA_LOAD_IOURING`: `1` forces the ring
//! (with the probe + buffered fallback), `0` forces mmap, unset = AUTO.
//! The auto heuristics (shared with the expert tier's
//! `HI_CUDA_EXPERT_IOURING` auto mode) are [`auto_for_load`] and
//! [`auto_for_expert_stream`]; they differ deliberately:
//!
//! * **Loads are one-shot**: after the upload the page-cache copy is mostly
//!   worthless, so cold-or-oversized reads want O_DIRECT — ring when the
//!   extents are not already warm OR do not fit a conservative fraction of
//!   `MemAvailable`; mmap only when warm (buffered re-reads beat O_DIRECT).
//! * **Expert streaming re-reads forever**: the page cache IS the implicit
//!   tier, so a set that fits should be allowed to warm up — ring only when
//!   the streamable bytes exceed the fraction AND the extents are not
//!   already warm.
//!
//! Every measurement failure degrades to mmap (status quo), never an error.

use anyhow::{Result, anyhow};
use hi_gguf::GgufFile;

#[cfg(target_os = "linux")]
use crate::expert_uring::{BulkBytes, IoUringReader, LOAD_CHUNK};

/// Fraction of `MemAvailable` beyond which a byte set is considered too big
/// to be comfortably page-cache-resident.
pub(crate) const AUTO_MEM_FRACTION: f64 = 0.5;
/// One-shot loads: mmap only when at least this fraction of the sampled
/// extents is already resident (below it, even a partially-warm buffered read
/// loses to a full-speed O_DIRECT pass).
pub(crate) const LOAD_WARM_RESIDENCY: f64 = 0.9;
/// Expert streaming: at this residency the page cache is doing its job and
/// O_DIRECT would only bypass it.
pub(crate) const EXPERT_WARM_RESIDENCY: f64 = 0.5;

/// Tri-state env knob: `Some(true)` = forced on, `Some(false)` = forced off,
/// `None` = unset (auto).
pub(crate) fn tri_state_env(name: &str) -> Option<bool> {
    tri_state_value(std::env::var(name).ok().as_deref())
}

/// Pure mapping behind [`tri_state_env`] (unit-testable without process-global
/// env mutation): any set non-`0` value forces on.
pub(crate) fn tri_state_value(value: Option<&str>) -> Option<bool> {
    value.map(|value| value != "0")
}

/// Inputs to the auto decision, all best-effort.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AutoInputs {
    /// Bytes the caller is about to read (tensor extents / streamable experts).
    pub needed_bytes: u64,
    /// `MemAvailable` at decision time (`None` = unknown).
    pub mem_available: Option<u64>,
    /// Sampled page-cache residency of the extents (`None` = unmeasurable).
    pub residency: Option<f64>,
}

#[derive(Debug)]
pub(crate) struct AutoDecision {
    pub use_ring: bool,
    pub why: String,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn exceeds_mem_fraction(inputs: &AutoInputs) -> Option<u64> {
    inputs
        .mem_available
        .filter(|&avail| inputs.needed_bytes as f64 > avail as f64 * AUTO_MEM_FRACTION)
}

/// One-shot resident loads: ring when the extents are cold OR too big to
/// cache; mmap when warm (or nothing could be measured).
pub(crate) fn auto_for_load(inputs: &AutoInputs) -> AutoDecision {
    if let Some(residency) = inputs.residency
        && residency >= LOAD_WARM_RESIDENCY
    {
        return AutoDecision {
            use_ring: false,
            why: format!(
                "extents {:.0}% page-cache-resident; buffered mmap wins warm",
                residency * 100.0
            ),
        };
    }
    if let Some(avail) = exceeds_mem_fraction(inputs) {
        return AutoDecision {
            use_ring: true,
            why: format!(
                "{:.1} GiB to read exceeds {:.0}% of MemAvailable ({:.1} GiB)",
                gib(inputs.needed_bytes),
                AUTO_MEM_FRACTION * 100.0,
                gib(avail)
            ),
        };
    }
    match inputs.residency {
        Some(residency) => AutoDecision {
            use_ring: true,
            why: format!(
                "extents only {:.0}% page-cache-resident; cold reads win with O_DIRECT",
                residency * 100.0
            ),
        },
        None => AutoDecision {
            use_ring: false,
            why: "residency unmeasurable and the read fits in RAM; keeping buffered mmap"
                .to_string(),
        },
    }
}

/// Repeated expert streaming: ring only when the streamable set is too big
/// to cache AND not already warm; otherwise let the page cache serve.
pub(crate) fn auto_for_expert_stream(inputs: &AutoInputs) -> AutoDecision {
    if let Some(residency) = inputs.residency
        && residency >= EXPERT_WARM_RESIDENCY
    {
        return AutoDecision {
            use_ring: false,
            why: format!(
                "expert extents {:.0}% page-cache-resident; buffered mmap wins warm",
                residency * 100.0
            ),
        };
    }
    if let Some(avail) = exceeds_mem_fraction(inputs) {
        return AutoDecision {
            use_ring: true,
            why: format!(
                "{:.1} GiB streamable exceeds {:.0}% of MemAvailable ({:.1} GiB) and extents are {} resident",
                gib(inputs.needed_bytes),
                AUTO_MEM_FRACTION * 100.0,
                gib(avail),
                match inputs.residency {
                    Some(residency) => format!("{:.0}%", residency * 100.0),
                    None => "not measurably".to_string(),
                }
            ),
        };
    }
    AutoDecision {
        use_ring: false,
        why: format!(
            "{:.1} GiB streamable fits the page-cache budget; buffered mmap can warm up",
            gib(inputs.needed_bytes)
        ),
    }
}

/// Whole-tensor bytes handed to a loader: the mmap borrow (today's path) or
/// an owned O_DIRECT bulk read.
pub(crate) enum LoadedBytes<'a> {
    Mmap(&'a [u8]),
    #[cfg(target_os = "linux")]
    Ring(BulkBytes),
}

impl LoadedBytes<'_> {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(bytes) => bytes,
            #[cfg(target_os = "linux")]
            Self::Ring(bytes) => bytes.as_slice(),
        }
    }
}

/// The per-model-load byte source: constructed once in front of a loader's
/// matrix loop, decides mmap vs ring for the whole set, then serves each
/// tensor's bytes. Ring read errors degrade to the mmap view per tensor (with
/// one log line), never a load failure.
pub(crate) struct LoadByteSource<'g> {
    gguf: &'g GgufFile,
    #[cfg(target_os = "linux")]
    ring: Option<(IoUringReader, &'static str)>,
}

impl<'g> LoadByteSource<'g> {
    /// A source that always serves the mmap view (small side-files like the
    /// vision mmproj, or call sites that opt out).
    pub(crate) fn mmap_only(gguf: &'g GgufFile) -> Self {
        Self {
            gguf,
            #[cfg(target_os = "linux")]
            ring: None,
        }
    }

    /// Decide once for the tensors about to be loaded (`HI_CUDA_LOAD_IOURING`
    /// tri-state; auto = [`auto_for_load`] over the extents' size and sampled
    /// page-cache residency) and probe the ring if chosen. `label` names the
    /// loader in the log line.
    pub(crate) fn for_tensors<'names>(
        gguf: &'g GgufFile,
        label: &str,
        names: impl Iterator<Item = &'names str>,
    ) -> Self {
        #[cfg(target_os = "linux")]
        {
            let forced = tri_state_env("HI_CUDA_LOAD_IOURING");
            if forced == Some(false) {
                return Self::mmap_only(gguf);
            }
            let mut extents: Vec<(usize, u64, usize)> = Vec::new();
            let mut needed_bytes = 0u64;
            for name in names {
                let Ok(range) = gguf.tensor_file_range(name) else {
                    continue;
                };
                needed_bytes += range.len;
                extents.push((
                    range.shard,
                    range.file_offset,
                    usize::try_from(range.len).unwrap_or(usize::MAX),
                ));
            }
            let paths: Vec<std::path::PathBuf> = (0..gguf.shard_count())
                .filter_map(|shard| gguf.shard_path(shard).map(std::path::Path::to_path_buf))
                .collect();
            let mode = match forced {
                Some(true) => {
                    eprintln!("hi-cuda {label} load: io=iouring (HI_CUDA_LOAD_IOURING=1)");
                    "forced"
                }
                Some(false) => unreachable!("handled above"),
                None => {
                    let inputs = AutoInputs {
                        needed_bytes,
                        mem_available: crate::expert_pool::ram_tier::mem_available_bytes(),
                        residency: crate::expert_uring::sampled_extent_residency(
                            &paths, &extents, 64,
                        ),
                    };
                    let decision = auto_for_load(&inputs);
                    eprintln!(
                        "hi-cuda {label} load: io={} (auto: {})",
                        if decision.use_ring { "iouring" } else { "mmap" },
                        decision.why
                    );
                    if !decision.use_ring {
                        return Self::mmap_only(gguf);
                    }
                    "auto"
                }
            };
            match IoUringReader::open(&paths, crate::expert_uring::DEFAULT_QD) {
                Ok(reader) => {
                    for note in reader.notes() {
                        eprintln!("hi-cuda {label} load: io_uring {note}");
                    }
                    Self {
                        gguf,
                        ring: Some((reader, mode)),
                    }
                }
                Err(err) => {
                    eprintln!(
                        "hi-cuda {label} load: io_uring unavailable ({err:#}); using buffered mmap reads"
                    );
                    Self::mmap_only(gguf)
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (label, names);
            Self::mmap_only(gguf)
        }
    }

    pub(crate) fn gguf(&self) -> &'g GgufFile {
        self.gguf
    }

    pub(crate) fn is_ring(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.ring.is_some()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// One tensor's whole bytes plus its info, from the chosen backend.
    pub(crate) fn tensor_bytes(
        &self,
        name: &str,
    ) -> Result<(LoadedBytes<'g>, &'g hi_gguf::TensorInfo)> {
        let info = self
            .gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow!("GGUF tensor {name} is missing"))?;
        #[cfg(target_os = "linux")]
        if let Some((reader, _)) = &self.ring {
            match self.ring_tensor_bytes(reader, name) {
                Ok(bytes) => return Ok((LoadedBytes::Ring(bytes), info)),
                Err(err) => {
                    eprintln!(
                        "hi-cuda load: io_uring read of {name} failed ({err:#}); falling back to the mmap view"
                    );
                }
            }
        }
        let view = self.gguf.tensor_view(info)?;
        Ok((LoadedBytes::Mmap(view.bytes), info))
    }

    #[cfg(target_os = "linux")]
    fn ring_tensor_bytes(&self, reader: &IoUringReader, name: &str) -> Result<BulkBytes> {
        let range = self.gguf.tensor_file_range(name)?;
        reader.read_extent_chunked(
            range.shard,
            range.file_offset,
            usize::try_from(range.len)?,
            LOAD_CHUNK,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn auto_for_load_truth_table() {
        // Warm extents -> mmap regardless of size.
        let warm = auto_for_load(&AutoInputs {
            needed_bytes: 200 * GIB,
            mem_available: Some(64 * GIB),
            residency: Some(0.95),
        });
        assert!(!warm.use_ring, "{}", warm.why);
        assert!(warm.why.contains("95%"), "{}", warm.why);

        // Too big for the cache -> ring, even with unknown residency.
        let big = auto_for_load(&AutoInputs {
            needed_bytes: 145 * GIB,
            mem_available: Some(218 * GIB),
            residency: None,
        });
        assert!(big.use_ring, "{}", big.why);
        assert!(big.why.contains("exceeds"), "{}", big.why);

        // Fits but measured cold -> ring (one-shot loads gain nothing from
        // warming the cache).
        let cold = auto_for_load(&AutoInputs {
            needed_bytes: 12 * GIB,
            mem_available: Some(218 * GIB),
            residency: Some(0.02),
        });
        assert!(cold.use_ring, "{}", cold.why);

        // Partially warm below the load threshold still rings.
        let half = auto_for_load(&AutoInputs {
            needed_bytes: 12 * GIB,
            mem_available: Some(218 * GIB),
            residency: Some(0.6),
        });
        assert!(half.use_ring, "{}", half.why);

        // Nothing measurable and it fits -> status quo (mmap).
        let unknown = auto_for_load(&AutoInputs {
            needed_bytes: 12 * GIB,
            mem_available: None,
            residency: None,
        });
        assert!(!unknown.use_ring, "{}", unknown.why);
    }

    #[test]
    fn auto_for_expert_stream_truth_table() {
        // Warm -> mmap (the page cache is the tier).
        let warm = auto_for_expert_stream(&AutoInputs {
            needed_bytes: 160 * GIB,
            mem_available: Some(218 * GIB),
            residency: Some(0.8),
        });
        assert!(!warm.use_ring, "{}", warm.why);

        // Too big AND cold -> ring (the GLM-on-this-box case).
        let big_cold = auto_for_expert_stream(&AutoInputs {
            needed_bytes: 160 * GIB,
            mem_available: Some(218 * GIB),
            residency: Some(0.03),
        });
        assert!(big_cold.use_ring, "{}", big_cold.why);

        // Too big, residency unknown -> ring (cannot be cache-resident).
        let big_unknown = auto_for_expert_stream(&AutoInputs {
            needed_bytes: 160 * GIB,
            mem_available: Some(218 * GIB),
            residency: None,
        });
        assert!(big_unknown.use_ring, "{}", big_unknown.why);

        // Fits, cold -> mmap: unlike one-shot loads, streaming re-reads, so
        // the cacheable set is allowed to warm up.
        let fits_cold = auto_for_expert_stream(&AutoInputs {
            needed_bytes: 8 * GIB,
            mem_available: Some(218 * GIB),
            residency: Some(0.0),
        });
        assert!(!fits_cold.use_ring, "{}", fits_cold.why);

        // MemAvailable unknown -> conservative mmap.
        let unknown = auto_for_expert_stream(&AutoInputs {
            needed_bytes: 160 * GIB,
            mem_available: None,
            residency: Some(0.0),
        });
        assert!(!unknown.use_ring, "{}", unknown.why);
    }

    /// Byte equivalence through the facade on the synthetic streaming fixture
    /// GGUF: the ring source must serve exactly the mmap view's bytes for
    /// every tensor (constructed directly, no env). Skips the ring half where
    /// io_uring/O_DIRECT is unavailable.
    #[test]
    fn facade_ring_bytes_match_mmap_view_on_fixture() {
        let dir = crate::expert_pool::tests::fixture_dir("load-source");
        let model = dir.join("model.gguf");
        crate::expert_pool::tests::write_streaming_fixture(&model);
        let gguf = GgufFile::open(&model).unwrap();

        let mmap_source = LoadByteSource::mmap_only(&gguf);
        assert!(!mmap_source.is_ring());
        let names: Vec<String> = gguf
            .tensors()
            .iter()
            .map(|info| info.name.clone())
            .collect();
        #[cfg(target_os = "linux")]
        {
            match IoUringReader::open(&[model.clone()], crate::expert_uring::DEFAULT_QD) {
                Ok(reader) => {
                    let ring_source = LoadByteSource {
                        gguf: &gguf,
                        ring: Some((reader, "forced")),
                    };
                    assert!(ring_source.is_ring());
                    for name in &names {
                        let (via_mmap, info_a) = mmap_source.tensor_bytes(name).unwrap();
                        let (via_ring, info_b) = ring_source.tensor_bytes(name).unwrap();
                        assert_eq!(info_a.name, info_b.name);
                        assert_eq!(
                            via_ring.as_slice(),
                            via_mmap.as_slice(),
                            "facade ring vs mmap bytes for {name}"
                        );
                    }
                }
                Err(err) => eprintln!("skipping ring half: {err:#}"),
            }
        }
        for name in &names {
            let (bytes, info) = mmap_source.tensor_bytes(name).unwrap();
            assert_eq!(bytes.as_slice().len() as u64, info.byte_len().unwrap());
        }
        assert!(mmap_source.tensor_bytes("no.such.tensor").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Real cold-load wall-time A/B through the ACTUAL qwen loader: page
    /// cache dropped per-extent, then `CudaQwenGpuModel::from_gguf` once with
    /// the ring forced off and once forced on. Needs a CUDA device and a real
    /// model; run alone:
    ///   `cargo test -p hi-cuda --release --features native-cuda \
    ///      real_cold_load_wall_time_ab -- --ignored --nocapture --test-threads=1`
    /// Model via `HI_LOAD_AB_GGUF=/path/model.gguf`, defaulting to the local
    /// qwen2.5-coder-7b when present.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "loads a real model on the GPU twice; opt-in"]
    fn real_cold_load_wall_time_ab() {
        let path = std::env::var("HI_LOAD_AB_GGUF").ok().or_else(|| {
            let home = std::env::var_os("HOME")?;
            let default = std::path::PathBuf::from(home)
                .join(".hi/models/qwen2.5-coder-7b-instruct-q6_k.gguf");
            default
                .exists()
                .then(|| default.to_string_lossy().into_owned())
        });
        let Some(path) = path else {
            eprintln!("skipping: set HI_LOAD_AB_GGUF=/path/model.gguf");
            return;
        };
        let drop_and_measure_residency = || {
            let gguf = GgufFile::open(&path).unwrap();
            let paths: Vec<std::path::PathBuf> = (0..gguf.shard_count())
                .filter_map(|shard| gguf.shard_path(shard).map(std::path::Path::to_path_buf))
                .collect();
            let extents: Vec<(usize, u64, usize)> = gguf
                .tensors()
                .iter()
                .filter_map(|info| {
                    let range = gguf.tensor_file_range(&info.name).ok()?;
                    Some((
                        range.shard,
                        range.file_offset,
                        usize::try_from(range.len).ok()?,
                    ))
                })
                .collect();
            drop(gguf); // unmap before the fadvise
            for shard_path in &paths {
                let file = std::fs::File::open(shard_path).unwrap();
                use std::os::fd::AsRawFd;
                let ret = unsafe {
                    libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
                };
                assert_eq!(ret, 0);
            }
            crate::expert_uring::sampled_extent_residency(&paths, &extents, 64).unwrap_or(1.0)
        };
        for (label, forced) in [
            ("mmap (HI_CUDA_LOAD_IOURING=0)", "0"),
            ("io_uring (=1)", "1"),
        ] {
            let residency = drop_and_measure_residency();
            // SAFETY: --test-threads=1 per the doc comment (mirrors the
            // native_cuda_batch_graph smoke's env handling).
            unsafe { std::env::set_var("HI_CUDA_LOAD_IOURING", forced) };
            let started = std::time::Instant::now();
            let gguf = GgufFile::open(&path).unwrap();
            let model = crate::gpu::CudaQwenGpuModel::from_gguf(&gguf).unwrap();
            let secs = started.elapsed().as_secs_f64();
            println!(
                "cold load ({label}): {secs:.2}s (pre-load residency {:.0}%)",
                residency * 100.0
            );
            drop(model);
        }
        unsafe { std::env::remove_var("HI_CUDA_LOAD_IOURING") };
    }

    #[test]
    fn tri_state_parses_forced_and_auto() {
        // Never mutate process env in tests: exercise the pure mapping.
        assert_eq!(tri_state_value(None), None);
        assert_eq!(tri_state_value(Some("0")), Some(false));
        assert_eq!(tri_state_value(Some("1")), Some(true));
        assert_eq!(
            tri_state_value(Some("auto")),
            Some(true),
            "any set non-zero value forces on"
        );
    }
}
