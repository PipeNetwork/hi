// Quant block decoders index several parallel sub-arrays (quants, scales, mins)
// by the same counter, so the range-loop form mirrors the layout and is
// intentional over an iterator rewrite.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use hi_local_core::model::{DEFAULT_MAX_OUTPUT_TOKENS, ModelInfo, TokenizerInfo, WeightShard};
use memmap2::Mmap;
use serde::Serialize;

mod dequantize;
mod tokenizer;
mod weights;

// Facade re-exports: the public API is unchanged — `hi_gguf::X` resolves to the
// same items whether they live here or in the focused submodules.
pub use dequantize::{GgufTensorType, dequantize_tensor_as_f32};
pub use tokenizer::{GgufTokenizer, GgufTokenizerSummary, StreamingTokenDecoder};
pub use weights::*;

// Cross-module items used by the container/parsing code kept in this file.
use dequantize::unsupported_tensor_type_error;
use weights::validate_qwen_tensors;

// Test-only surface: the in-file test module drives these tokenizer helpers and
// the `ModelFamily` re-export through `use super::*`.
#[cfg(test)]
use hi_local_core::model::ModelFamily;
#[cfg(test)]
use tokenizer::{
    decode_byte_level_text, decode_sentencepiece_text, decode_tokenizer_bytes_lenient,
    encode_byte_level_text,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    mmap: Mmap,
    version: u32,
    alignment: u64,
    data_start: u64,
    /// Additional mmapped files of a split GGUF (`split.count > 1`); tensor
    /// shard index n > 0 resolves to `extra_shards[n - 1]`.
    extra_shards: Vec<GgufShard>,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    // name -> index into `tensors`, so `tensor(name)` is O(1) instead of a linear scan.
    // Matters for MoE models with tens of thousands of tensors, where per-tensor lookups
    // during model construction were O(n^2).
    tensor_index: std::collections::HashMap<String, usize>,
}

#[derive(Debug)]
struct GgufShard {
    path: PathBuf,
    mmap: Mmap,
    data_start: u64,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory-mapping {}", path.display()))?;
        let mut gguf = parse_mmap(path.to_path_buf(), mmap)?;
        gguf.absorb_split_shards()?;
        Ok(gguf)
    }

    /// llama.cpp split GGUFs (`name-00001-of-000NN.gguf`) spread tensors over
    /// N files; the first shard carries the model metadata (and possibly
    /// tensors), later shards carry only their own tensor tables. Open and
    /// absorb the siblings so the tensor map spans the whole model.
    fn absorb_split_shards(&mut self) -> Result<()> {
        let split_count = self
            .metadata_u32("split.count")
            .map(|value| value as usize)
            .unwrap_or(1);
        if split_count <= 1 {
            return Ok(());
        }
        let split_no = self.metadata_u32("split.no").unwrap_or(0);
        if split_no != 0 {
            bail!(
                "{} is shard {} of a {split_count}-file split GGUF; open the first shard (-00001-of-...)",
                self.path.display(),
                split_no + 1
            );
        }
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow!("split GGUF path has no file name"))?;
        let first_suffix = format!("-00001-of-{split_count:05}.gguf");
        let Some(stem) = file_name.strip_suffix(first_suffix.as_str()) else {
            bail!(
                "split GGUF {} (split.count = {split_count}) must be named like <model>{first_suffix}",
                self.path.display()
            );
        };
        let expected_total = self
            .metadata_u32("split.tensors.count")
            .map(|value| value as usize);
        for shard_no in 1..split_count {
            let sibling = self.path.with_file_name(format!(
                "{stem}-{:05}-of-{split_count:05}.gguf",
                shard_no + 1
            ));
            let file = File::open(&sibling)
                .with_context(|| format!("opening split GGUF shard {}", sibling.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("memory-mapping {}", sibling.display()))?;
            let shard = parse_mmap(sibling.clone(), mmap)?;
            let shard_split_no = shard.metadata_u32("split.no").unwrap_or(0) as usize;
            if shard_split_no != shard_no {
                bail!(
                    "split GGUF shard {} reports split.no {shard_split_no}; expected {shard_no}",
                    sibling.display()
                );
            }
            let shard_index = self.extra_shards.len() + 1;
            for mut info in shard.tensors {
                info.shard = shard_index;
                if self
                    .tensor_index
                    .insert(info.name.clone(), self.tensors.len())
                    .is_some()
                {
                    bail!(
                        "split GGUF shard {} duplicates tensor {}",
                        sibling.display(),
                        info.name
                    );
                }
                self.tensors.push(info);
            }
            self.extra_shards.push(GgufShard {
                path: sibling,
                mmap: shard.mmap,
                data_start: shard.data_start,
            });
        }
        if let Some(expected) = expected_total
            && self.tensors.len() != expected
        {
            bail!(
                "split GGUF has {} tensors across shards; split.tensors.count says {expected}",
                self.tensors.len()
            );
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// O(1) lookup of a tensor's metadata by name (vs a linear scan of `tensors()`).
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(*self.tensor_index.get(name)?)
    }

    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let info = self.tensors.get(*self.tensor_index.get(name)?)?;
        self.tensor_view(info).ok()
    }

    pub fn tensor_view<'a>(&'a self, info: &'a TensorInfo) -> Result<TensorView<'a>> {
        let byte_len = info.byte_len()?;
        let (mmap, data_start) = if info.shard == 0 {
            (&self.mmap, self.data_start)
        } else {
            let shard = self.extra_shards.get(info.shard - 1).ok_or_else(|| {
                anyhow!(
                    "tensor {} references missing GGUF shard {}",
                    info.name,
                    info.shard
                )
            })?;
            (&shard.mmap, shard.data_start)
        };
        let start = checked_add(data_start, info.offset, "tensor data offset")?;
        let end = checked_add(start, byte_len, "tensor data length")?;
        let start = usize::try_from(start).context("tensor data offset does not fit usize")?;
        let end = usize::try_from(end).context("tensor data length does not fit usize")?;
        let bytes = mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("tensor {} points outside GGUF data section", info.name))?;
        Ok(TensorView { info, bytes })
    }

    /// Number of files backing this GGUF (1 unless it is a split model).
    pub fn shard_count(&self) -> usize {
        1 + self.extra_shards.len()
    }

    /// Filesystem path of one shard (0 = the file passed to [`GgufFile::open`]).
    pub fn shard_path(&self, shard: usize) -> Option<&Path> {
        if shard == 0 {
            Some(&self.path)
        } else {
            self.extra_shards
                .get(shard - 1)
                .map(|shard| shard.path.as_path())
        }
    }

    /// Absolute byte extent of a tensor's data within its backing shard file.
    /// This is the on-disk location (`shard file offset`, not mmap-relative),
    /// for direct-I/O readers and readahead planning; sub-ranges of the tensor
    /// are plain offsets from `file_offset`.
    pub fn tensor_file_range(&self, name: &str) -> Result<TensorFileRange> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| anyhow!("tensor {name} missing from GGUF"))?;
        let data_start = if info.shard == 0 {
            self.data_start
        } else {
            self.extra_shards
                .get(info.shard - 1)
                .ok_or_else(|| {
                    anyhow!("tensor {name} references missing GGUF shard {}", info.shard)
                })?
                .data_start
        };
        Ok(TensorFileRange {
            shard: info.shard,
            file_offset: checked_add(data_start, info.offset, "tensor file offset")?,
            len: info.byte_len()?,
        })
    }

    /// Page-cache advice for one tensor's full mmap extent. See
    /// [`GgufFile::advise_tensor_range`].
    pub fn advise_tensor(&self, name: &str, advice: GgufMemoryAdvice) -> Result<()> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| anyhow!("tensor {name} missing from GGUF"))?;
        self.advise_tensor_range(name, 0, info.byte_len()?, advice)
    }

    /// `madvise` a byte sub-range of one tensor's data on the backing mmap.
    /// `MADV_RANDOM` on streamed-expert extents kills readahead amplification
    /// (a fault no longer drags in megabytes of neighboring experts);
    /// `WillNeed` starts asynchronous readahead of exactly the extent about to
    /// be copied. The advice is applied page-aligned (the range is widened to
    /// page boundaries, never narrowed). No-op on non-unix platforms.
    pub fn advise_tensor_range(
        &self,
        name: &str,
        range_offset: u64,
        range_len: u64,
        advice: GgufMemoryAdvice,
    ) -> Result<()> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| anyhow!("tensor {name} missing from GGUF"))?;
        let byte_len = info.byte_len()?;
        let range_end = checked_add(range_offset, range_len, "tensor advice range")?;
        if range_end > byte_len {
            bail!(
                "advice range {range_offset}..{range_end} exceeds tensor {name} ({byte_len} bytes)"
            );
        }
        let (mmap, data_start) = if info.shard == 0 {
            (&self.mmap, self.data_start)
        } else {
            let shard = self.extra_shards.get(info.shard - 1).ok_or_else(|| {
                anyhow!("tensor {name} references missing GGUF shard {}", info.shard)
            })?;
            (&shard.mmap, shard.data_start)
        };
        let start = checked_add(
            checked_add(data_start, info.offset, "tensor data offset")?,
            range_offset,
            "tensor advice offset",
        )?;
        let start = usize::try_from(start).context("tensor advice offset does not fit usize")?;
        let len = usize::try_from(range_len).context("tensor advice length does not fit usize")?;
        advise_mmap_range(mmap, start, len, advice)
    }

    /// Open every shard a second time for O_DIRECT positioned reads that
    /// bypass the page cache entirely (the "twin fd" pattern: the mmap fds
    /// stay buffered for the trunk, this reader serves streamed-expert
    /// extents). Linux-only; fails with a clear error where the OS or the
    /// filesystem (e.g. tmpfs) does not support O_DIRECT, in which case the
    /// caller falls back to the buffered mmap path.
    pub fn direct_io_reader(&self) -> Result<GgufDirectReader> {
        GgufDirectReader::open(self)
    }

    pub fn qwen_config(&self) -> Result<QwenGgufConfig> {
        QwenGgufConfig::from_gguf(self)
    }

    pub fn tokenizer(&self) -> Result<GgufTokenizer> {
        GgufTokenizer::from_gguf(self)
    }

    pub fn qwen_tensor_validation(&self) -> Result<QwenTensorValidation> {
        let config = self.qwen_config()?;
        Ok(validate_qwen_tensors(self, &config))
    }

    pub fn validate_qwen_tensors(&self) -> Result<QwenTensorValidation> {
        let validation = self.qwen_tensor_validation()?;
        if validation.valid {
            Ok(validation)
        } else {
            bail!(
                "invalid Qwen GGUF tensor table: {}",
                validation.errors.join("; ")
            );
        }
    }

    pub fn summary(&self) -> Result<GgufSummary> {
        let qwen = self.qwen_config().ok();
        let tokenizer = self.tokenizer().ok().map(|tokenizer| tokenizer.summary());
        let qwen_tensors = qwen
            .as_ref()
            .map(|config| validate_qwen_tensors(self, config));
        let metadata_keys = self.metadata.keys().cloned().collect();
        let tensors = self
            .tensors
            .iter()
            .map(|tensor| GgufTensorSummary {
                name: tensor.name.clone(),
                shape: tensor.dimensions.clone(),
                dtype: tensor.dtype.label().to_string(),
                offset: tensor.offset,
                bytes: tensor.byte_len().unwrap_or(0),
            })
            .collect();
        Ok(GgufSummary {
            path: self.path.clone(),
            version: self.version,
            alignment: self.alignment,
            data_start: self.data_start,
            metadata_count: self.metadata.len(),
            metadata_keys,
            tensor_count: self.tensors.len(),
            tensors,
            qwen,
            tokenizer,
            qwen_tensors,
        })
    }
}

/// Absolute byte extent of one tensor's data within its backing shard file
/// (on-disk offsets, not mmap-relative). See [`GgufFile::tensor_file_range`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorFileRange {
    /// Index of the shard file holding the tensor (0 = primary file); resolve
    /// to a path with [`GgufFile::shard_path`].
    pub shard: usize,
    /// Byte offset of the tensor's first data byte within the shard file.
    pub file_offset: u64,
    /// Tensor data length in bytes.
    pub len: u64,
}

/// Page-cache policy for a tensor's mmap extent (a safe subset of `madvise`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMemoryAdvice {
    /// Default kernel readahead.
    Normal,
    /// Expect random access: disable readahead (use for streamed-expert
    /// extents so one expert fault does not drag neighbors into the cache).
    Random,
    /// Expect sequential access: aggressive readahead.
    Sequential,
    /// Start asynchronous readahead of the range now.
    WillNeed,
}

#[cfg(unix)]
fn advise_mmap_range(
    mmap: &Mmap,
    start: usize,
    len: usize,
    advice: GgufMemoryAdvice,
) -> Result<()> {
    const PAGE: usize = 4096;
    let advice = match advice {
        GgufMemoryAdvice::Normal => memmap2::Advice::Normal,
        GgufMemoryAdvice::Random => memmap2::Advice::Random,
        GgufMemoryAdvice::Sequential => memmap2::Advice::Sequential,
        GgufMemoryAdvice::WillNeed => memmap2::Advice::WillNeed,
    };
    // madvise requires a page-aligned address: widen the range down to the
    // containing page (the mmap base itself is page-aligned).
    let aligned_start = start & !(PAGE - 1);
    let aligned_len = len + (start - aligned_start);
    if aligned_len == 0 {
        return Ok(());
    }
    mmap.advise_range(advice, aligned_start, aligned_len)
        .context("madvise on GGUF mmap range")
}

#[cfg(not(unix))]
fn advise_mmap_range(
    _mmap: &Mmap,
    _start: usize,
    _len: usize,
    _advice: GgufMemoryAdvice,
) -> Result<()> {
    Ok(())
}

/// O_DIRECT logical-block granularity: file offset, read length and buffer
/// address must all be multiples of the device's logical block size. 4096
/// covers every NVMe/ext4/xfs configuration in practice.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const DIRECT_IO_BLOCK: usize = 4096;

/// Positioned O_DIRECT reads over every shard of a (split) GGUF, bypassing
/// the page cache. Reads land in an internal block-aligned scratch allocation
/// and the requested sub-range is copied out, so callers may ask for arbitrary
/// (unaligned) extents. Thread-safe: `read_range` uses positioned reads on
/// shared fds, so concurrent expert fetches need no locking.
#[derive(Debug)]
pub struct GgufDirectReader {
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    shards: Vec<File>,
}

impl GgufDirectReader {
    #[cfg(target_os = "linux")]
    fn open(gguf: &GgufFile) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let mut shards = Vec::with_capacity(gguf.shard_count());
        for shard in 0..gguf.shard_count() {
            let path = gguf
                .shard_path(shard)
                .ok_or_else(|| anyhow!("GGUF shard {shard} has no path"))?;
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
            shards.push(file);
        }
        Ok(Self { shards })
    }

    #[cfg(not(target_os = "linux"))]
    fn open(_gguf: &GgufFile) -> Result<Self> {
        bail!("O_DIRECT GGUF reads are only supported on Linux")
    }

    /// Read `len` bytes at absolute file offset `file_offset` of `shard`
    /// (offsets from [`GgufFile::tensor_file_range`] plus any sub-range).
    pub fn read_range(&self, shard: usize, file_offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; len];
        self.read_range_into(shard, file_offset, &mut out)?;
        Ok(out)
    }

    /// Read exactly `out.len()` bytes at `file_offset` of `shard` into `out`.
    #[cfg(target_os = "linux")]
    pub fn read_range_into(&self, shard: usize, file_offset: u64, out: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        if out.is_empty() {
            return Ok(());
        }
        let file = self
            .shards
            .get(shard)
            .ok_or_else(|| anyhow!("O_DIRECT read references missing GGUF shard {shard}"))?;
        let block = DIRECT_IO_BLOCK as u64;
        let aligned_start = file_offset / block * block;
        let head = (file_offset - aligned_start) as usize;
        let aligned_len = (head + out.len()).div_ceil(DIRECT_IO_BLOCK) * DIRECT_IO_BLOCK;
        let mut scratch = AlignedBlockBuf::new(aligned_len)?;
        let buf = scratch.as_mut_slice();
        // O_DIRECT reads may come back short (and MUST stop short at EOF when
        // the file length is not block-aligned); loop until the caller's range
        // is covered.
        let mut filled = 0usize;
        while filled < head + out.len() {
            let read = file
                .read_at(&mut buf[filled..], aligned_start + filled as u64)
                .with_context(|| {
                    format!("O_DIRECT read of {aligned_len} bytes at {aligned_start}")
                })?;
            if read == 0 {
                bail!(
                    "O_DIRECT read hit EOF: wanted {} bytes at {file_offset} of shard {shard}",
                    out.len()
                );
            }
            filled += read;
        }
        out.copy_from_slice(&buf[head..head + out.len()]);
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn read_range_into(&self, _shard: usize, _file_offset: u64, _out: &mut [u8]) -> Result<()> {
        bail!("O_DIRECT GGUF reads are only supported on Linux")
    }
}

/// Heap allocation aligned to [`DIRECT_IO_BLOCK`], as O_DIRECT requires of the
/// destination buffer.
#[cfg(target_os = "linux")]
struct AlignedBlockBuf {
    ptr: *mut u8,
    len: usize,
}

#[cfg(target_os = "linux")]
impl AlignedBlockBuf {
    fn new(len: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len, DIRECT_IO_BLOCK)
            .context("O_DIRECT scratch layout")?;
        // SAFETY: layout has non-zero size (callers round up to >= one block).
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            bail!("allocating {len}-byte aligned O_DIRECT scratch failed");
        }
        Ok(Self { ptr, len })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is a live allocation of exactly `len` bytes owned by self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for AlignedBlockBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, DIRECT_IO_BLOCK)
            .expect("layout validated at construction");
        // SAFETY: allocated with the identical layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

// SAFETY: AlignedBlockBuf is a plain owned allocation (no thread affinity).
#[cfg(target_os = "linux")]
unsafe impl Send for AlignedBlockBuf {}

pub fn inspect_model(path: impl AsRef<Path>, model_id: Option<String>) -> Result<ModelInfo> {
    let path = path.as_ref();
    let gguf = GgufFile::open(path)?;
    let qwen = gguf.qwen_config()?;
    let id = model_id
        .filter(|id| !id.trim().is_empty())
        .or_else(|| {
            gguf.metadata_string("general.name")
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("local-gguf-model")
                .to_string()
        });
    let metadata =
        std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    let shard_path = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.gguf")
        .to_string();
    let tokenizer = gguf.tokenizer().ok();
    let chat_template = gguf.chat_template().is_some();

    Ok(ModelInfo {
        id,
        path: path.to_path_buf(),
        family: qwen.family,
        model_type: qwen.architecture.clone(),
        architecture: qwen.architecture.clone(),
        context_length: Some(qwen.context_length),
        max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        tokenizer: TokenizerInfo {
            tokenizer_json: false,
            tokenizer_config: tokenizer.is_some() || chat_template,
            special_tokens_map: tokenizer
                .as_ref()
                .is_some_and(|tokenizer| !tokenizer.special_ids.is_empty()),
        },
        chat_template,
        weight_shards: vec![WeightShard {
            path: shard_path,
            bytes: metadata.len(),
            tensor_count: Some(gguf.tensors.len()),
        }],
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: GgufTensorType,
    pub offset: u64,
    /// Which file of a split GGUF holds this tensor's data (0 = the primary
    /// shard; single-file models are always 0).
    pub shard: usize,
}

impl TensorInfo {
    pub fn element_count(&self) -> Result<u64> {
        if self.dimensions.is_empty() {
            bail!("tensor {} has zero dimensions", self.name);
        }
        self.dimensions.iter().try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| anyhow!("tensor {} element count overflows u64", self.name))
        })
    }

    pub fn byte_len(&self) -> Result<u64> {
        let element_count = self.element_count()?;
        self.dtype.byte_len(element_count).with_context(|| {
            format!(
                "tensor {} has dtype {} and element count {element_count}",
                self.name,
                self.dtype.label()
            )
        })
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]

pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
}


#[derive(Clone, Debug, Serialize)]
pub struct GgufSummary {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub data_start: u64,
    pub metadata_count: usize,
    pub metadata_keys: Vec<String>,
    pub tensor_count: usize,
    pub tensors: Vec<GgufTensorSummary>,
    pub qwen: Option<QwenGgufConfig>,
    pub tokenizer: Option<GgufTokenizerSummary>,
    pub qwen_tensors: Option<QwenTensorValidation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GgufTensorSummary {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub offset: u64,
    pub bytes: u64,
}

impl GgufFile {
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.metadata_string("tokenizer.chat_template")
    }

    fn required_u32(&self, key: &str) -> Result<u32> {
        self.metadata_u32(key)
            .ok_or_else(|| anyhow!("GGUF metadata missing required numeric field {key}"))
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Uint8(value)) => Some((*value).into()),
            Some(MetadataValue::Uint16(value)) => Some((*value).into()),
            Some(MetadataValue::Uint32(value)) => Some(*value),
            Some(MetadataValue::Uint64(value)) => u32::try_from(*value).ok(),
            Some(MetadataValue::Int8(value)) => u32::try_from(*value).ok(),
            Some(MetadataValue::Int16(value)) => u32::try_from(*value).ok(),
            Some(MetadataValue::Int32(value)) => u32::try_from(*value).ok(),
            Some(MetadataValue::Int64(value)) => u32::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Float32(value)) => Some(*value),
            Some(MetadataValue::Float64(value)) => Some(*value as f32),
            _ => None,
        }
    }

    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(MetadataValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    fn metadata_string_array(&self, key: &str) -> Result<Option<Vec<String>>> {
        let Some(value) = self.metadata.get(key) else {
            return Ok(None);
        };
        let MetadataValue::Array(values) = value else {
            bail!("GGUF metadata {key} must be an array of strings");
        };
        values
            .iter()
            .map(|value| match value {
                MetadataValue::String(value) => Ok(value.clone()),
                _ => bail!("GGUF metadata {key} must be an array of strings"),
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    pub fn metadata_f32_array(&self, key: &str) -> Result<Option<Vec<f32>>> {
        let Some(value) = self.metadata.get(key) else {
            return Ok(None);
        };
        let MetadataValue::Array(values) = value else {
            bail!("GGUF metadata {key} must be an array of floats");
        };
        values
            .iter()
            .map(|value| match value {
                MetadataValue::Float32(value) => Ok(*value),
                MetadataValue::Float64(value) => Ok(*value as f32),
                _ => bail!("GGUF metadata {key} must be an array of floats"),
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    pub fn metadata_i32_array(&self, key: &str) -> Result<Option<Vec<i32>>> {
        let Some(value) = self.metadata.get(key) else {
            return Ok(None);
        };
        let MetadataValue::Array(values) = value else {
            bail!("GGUF metadata {key} must be an array of integers");
        };
        values
            .iter()
            .map(|value| match value {
                MetadataValue::Uint8(value) => Ok(i32::from(*value)),
                MetadataValue::Int8(value) => Ok(i32::from(*value)),
                MetadataValue::Uint16(value) => Ok(i32::from(*value)),
                MetadataValue::Int16(value) => Ok(i32::from(*value)),
                MetadataValue::Uint32(value) => i32::try_from(*value)
                    .with_context(|| format!("GGUF metadata {key} has out-of-range integer")),
                MetadataValue::Int32(value) => Ok(*value),
                MetadataValue::Uint64(value) => i32::try_from(*value)
                    .with_context(|| format!("GGUF metadata {key} has out-of-range integer")),
                MetadataValue::Int64(value) => i32::try_from(*value)
                    .with_context(|| format!("GGUF metadata {key} has out-of-range integer")),
                _ => bail!("GGUF metadata {key} must be an array of integers"),
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }
}

fn parse_mmap(path: PathBuf, mmap: Mmap) -> Result<GgufFile> {
    let mut reader = Reader::new(&mmap);
    let magic = reader.read_bytes(4)?;
    if magic != GGUF_MAGIC {
        bail!("bad GGUF magic in {}; expected GGUF", path.display());
    }
    let version = reader.read_u32()?;
    if !matches!(version, 2 | 3) {
        bail!("unsupported GGUF version {version}; hi-local accepts GGUF v2 and v3");
    }
    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;
    let tensor_count_usize =
        usize::try_from(tensor_count).context("GGUF tensor count does not fit usize")?;
    let metadata_count_usize =
        usize::try_from(metadata_count).context("GGUF metadata count does not fit usize")?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count_usize {
        let key = reader.read_string()?;
        if metadata.contains_key(&key) {
            bail!("duplicate GGUF metadata key {key}");
        }
        let value_type = MetadataValueType::from_raw(reader.read_u32()?)?;
        let value = reader.read_metadata_value(value_type)?;
        metadata.insert(key, value);
    }

    let alignment = metadata_alignment(&metadata)?;
    let mut tensors = Vec::with_capacity(tensor_count_usize);
    for _ in 0..tensor_count_usize {
        let name = reader.read_string()?;
        let n_dimensions = reader.read_u32()?;
        if n_dimensions == 0 || n_dimensions > 4 {
            bail!("tensor {name} has unsupported dimension count {n_dimensions}");
        }
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            let dim = reader.read_u64()?;
            if dim == 0 {
                bail!("tensor {name} has a zero-sized dimension");
            }
            dimensions.push(dim);
        }
        let dtype_raw = reader.read_u32()?;
        let dtype = GgufTensorType::from_raw(dtype_raw)
            .map_err(|_| unsupported_tensor_type_error(dtype_raw, Some(&name)))?;
        let offset = reader.read_u64()?;
        if offset % alignment != 0 {
            bail!("tensor {name} offset {offset} is not aligned to GGUF alignment {alignment}");
        }
        tensors.push(TensorInfo {
            name,
            dimensions,
            dtype,
            offset,
            shard: 0,
        });
    }

    let data_start = align_to(reader.position() as u64, alignment)?;
    let file_len = mmap.len() as u64;
    if data_start > file_len {
        bail!("GGUF tensor data section starts beyond end of file");
    }
    for tensor in &tensors {
        let byte_len = tensor.byte_len()?;
        let start = checked_add(data_start, tensor.offset, "tensor data offset")?;
        let end = checked_add(start, byte_len, "tensor data length")?;
        if end > file_len {
            bail!(
                "tensor {} byte range {}..{} exceeds GGUF file length {}",
                tensor.name,
                start,
                end,
                file_len
            );
        }
    }

    let tensor_index = tensors
        .iter()
        .enumerate()
        .map(|(idx, tensor)| (tensor.name.clone(), idx))
        .collect();
    Ok(GgufFile {
        path,
        mmap,
        version,
        alignment,
        data_start,
        extra_shards: Vec::new(),
        metadata,
        tensors,
        tensor_index,
    })
}

#[derive(Clone, Copy)]
enum MetadataValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl MetadataValueType {
    fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => bail!("unsupported GGUF metadata value type {other}"),
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn read_metadata_value(&mut self, value_type: MetadataValueType) -> Result<MetadataValue> {
        Ok(match value_type {
            MetadataValueType::Uint8 => MetadataValue::Uint8(self.read_u8()?),
            MetadataValueType::Int8 => MetadataValue::Int8(self.read_i8()?),
            MetadataValueType::Uint16 => MetadataValue::Uint16(self.read_u16()?),
            MetadataValueType::Int16 => MetadataValue::Int16(self.read_i16()?),
            MetadataValueType::Uint32 => MetadataValue::Uint32(self.read_u32()?),
            MetadataValueType::Int32 => MetadataValue::Int32(self.read_i32()?),
            MetadataValueType::Float32 => MetadataValue::Float32(self.read_f32()?),
            MetadataValueType::Bool => MetadataValue::Bool(self.read_u8()? != 0),
            MetadataValueType::String => MetadataValue::String(self.read_string()?),
            MetadataValueType::Array => {
                let element_type = MetadataValueType::from_raw(self.read_u32()?)?;
                if matches!(element_type, MetadataValueType::Array) {
                    bail!("nested GGUF metadata arrays are not supported");
                }
                let len = self.read_u64()?;
                let len = usize::try_from(len).context("GGUF metadata array length too large")?;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_metadata_value(element_type)?);
                }
                MetadataValue::Array(values)
            }
            MetadataValueType::Uint64 => MetadataValue::Uint64(self.read_u64()?),
            MetadataValueType::Int64 => MetadataValue::Int64(self.read_i64()?),
            MetadataValueType::Float64 => MetadataValue::Float64(self.read_f64()?),
        })
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        let len = usize::try_from(len).context("GGUF string length too large")?;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(ToString::to_string)
            .context("GGUF string is not valid UTF-8")
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow!("GGUF read offset overflow"))?;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| anyhow!("truncated GGUF file while reading {len} bytes"))?;
        self.pos = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.read_bytes(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.read_bytes(2)?);
        Ok(i16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.read_bytes(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.read_bytes(4)?);
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.read_bytes(4)?);
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.read_bytes(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.read_bytes(8)?);
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.read_bytes(8)?);
        Ok(f64::from_le_bytes(bytes))
    }
}

fn metadata_alignment(metadata: &BTreeMap<String, MetadataValue>) -> Result<u64> {
    let alignment = match metadata.get("general.alignment") {
        Some(MetadataValue::Uint32(value)) => (*value).into(),
        Some(MetadataValue::Uint64(value)) => *value,
        Some(MetadataValue::Int32(value)) => u64::try_from(*value).unwrap_or(0),
        Some(MetadataValue::Int64(value)) => u64::try_from(*value).unwrap_or(0),
        None => DEFAULT_ALIGNMENT,
        Some(_) => bail!("GGUF metadata general.alignment must be an integer"),
    };
    if alignment == 0 {
        bail!("GGUF metadata general.alignment must be greater than zero");
    }
    Ok(alignment)
}

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        bail!("alignment must be greater than zero");
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        checked_add(value, alignment - remainder, "alignment padding")
    }
}

pub(crate) fn checked_add(left: u64, right: u64, context: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{context} overflows u64"))
}

pub(crate) fn array_len_as_u32(value: &MetadataValue) -> Option<u32> {
    match value {
        MetadataValue::Array(values) => u32::try_from(values.len()).ok(),
        _ => None,
    }
}

pub(crate) fn ensure_token_id_in_range(id: u32, len: usize, label: &str) -> Result<()> {
    if usize::try_from(id).ok().is_some_and(|idx| idx < len) {
        Ok(())
    } else {
        bail!("{label} {id} is outside tokenizer vocab of size {len}");
    }
}



#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const LLAMA3_CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

    #[test]
    fn lenient_byte_decode_keeps_complete_and_drops_partial_utf8() {
        // Complete 3-byte character (U+2019 RIGHT SINGLE QUOTATION MARK).
        assert_eq!(decode_tokenizer_bytes_lenient(&[0xE2, 0x80, 0x99]), "’");
        // Incomplete trailing sequence is dropped rather than erroring — it is
        // completed by a later token during streaming (or was cut off at max_tokens).
        assert_eq!(decode_tokenizer_bytes_lenient(&[0x41, 0xE2, 0x80]), "A");
        // A genuinely invalid byte in the middle becomes U+FFFD, decoding continues.
        assert_eq!(
            decode_tokenizer_bytes_lenient(&[0x41, 0xFF, 0x42]),
            "A\u{FFFD}B"
        );
        assert_eq!(
            decode_tokenizer_bytes_lenient(b"plain ascii"),
            "plain ascii"
        );
    }

    #[test]
    fn sentencepiece_decode_tolerates_truncated_byte_fallback() {
        // A full byte-fallback character round-trips (▁ becomes a space).
        assert_eq!(
            decode_sentencepiece_text("\u{2581}caf<0xC3><0xA9>").unwrap(),
            " café"
        );
        // Cut off mid-character (only the lead byte of é): the partial char is
        // dropped and the request no longer fails.
        assert_eq!(
            decode_sentencepiece_text("\u{2581}caf<0xC3>").unwrap(),
            " caf"
        );
        // U+2581 emitted via byte-fallback tokens (0xE2 0x96 0x81) must render as a
        // space, not leak the raw SentencePiece marker glyph into the output.
        assert_eq!(
            decode_sentencepiece_text("42<0xE2><0x96><0x81>The").unwrap(),
            "42 The"
        );
    }

    #[test]
    fn byte_level_decode_tolerates_truncated_multibyte() {
        // Byte-level tokens whose reconstructed bytes end mid-character must not
        // error; the incomplete tail is dropped.
        let encoded = encode_byte_level_text("é".as_bytes()); // two byte-tokens
        let mut chars = encoded.chars();
        let partial: String = std::iter::once(chars.next().unwrap()).collect();
        assert_eq!(decode_byte_level_text(&partial).unwrap(), "");
        assert_eq!(decode_byte_level_text(&encoded).unwrap(), "é");
    }

    #[test]
    fn parses_header_metadata_tensor_table_and_qwen_config() {
        let path = tempfile_path("tiny-qwen");
        write_tiny_qwen(&path, 1, 0);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();

        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.alignment(), 32);
        assert_eq!(gguf.tensors().len(), 1);
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.attention_head_count_kv, 1);
        assert_eq!(config.vocab_size, Some(2));
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_llama_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("tiny-llama");
        write_tiny_llama(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();

        assert_eq!(config.architecture, "llama");
        assert_eq!(config.family, ModelFamily::Llama);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.default_rope_freq_base(), 10_000.0);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
        assert_eq!(gguf.chat_template(), Some(LLAMA3_CHAT_TEMPLATE));
    }

    #[test]
    fn parses_mistral_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("tiny-mistral");
        write_tiny_mistral(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.default_rope_freq_base(), 10_000.0);
        assert!(validation.valid);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_mistral_config_with_dense_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-aliases");
        write_tiny_mistral_dense_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert!(gguf.tensor("blk.0.attn_q.weight").is_none());
        assert!(gguf.tensor("token_embd.weight").is_none());
        assert!(gguf.tensor("model.embed_tokens.weight").is_some());
        assert!(gguf.tensor("output.weight").is_none());
        assert!(gguf.tensor("lm_head.weight").is_some());
        assert!(gguf.tensor("blk.0.self_attn.q_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.gate_proj.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_model_layers_dense_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-model-layers-aliases");
        write_tiny_mistral_model_layers_dense_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert!(gguf.tensor("blk.0.self_attn.q_proj.weight").is_none());
        assert!(
            gguf.tensor("model.layers.0.self_attn.q_proj.weight")
                .is_some()
        );
        assert!(gguf.tensor("model.layers.0.mlp.gate_proj.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_feed_forward_hf_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-feed-forward-hf-aliases");
        write_tiny_mistral_feed_forward_hf_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("model.layers.0.feed_forward.gate_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.layers.0.feed_forward.down_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mistral_config_with_attn_container_split_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-attn-container-split-aliases");
        write_tiny_mistral_attn_container_split_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert!(gguf.tensor("blk.0.attn.q_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.attn.o_proj.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_ffn_container_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-ffn-container-aliases");
        write_tiny_mistral_ffn_container_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert!(gguf.tensor("blk.0.ffn.gate_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.ffn.down_proj.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_language_model_wrapper_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-language-model-wrapper-aliases");
        write_tiny_mistral_language_model_wrapper_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert!(
            gguf.tensor("language_model.model.embed_tokens.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("language_model.model.layers.0.self_attn.q_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("language_model.model.layers.0.mlp.down_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mistral_config_with_packed_dense_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-packed-aliases");
        write_tiny_mistral_packed_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.self_attn.q_proj.weight").is_none());
        assert!(gguf.tensor("blk.0.self_attn.qkv_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.gate_proj.weight").is_none());
        assert!(gguf.tensor("blk.0.mlp.gate_up_proj.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_model_layers_packed_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-model-layers-packed-aliases");
        write_tiny_mistral_model_layers_packed_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.self_attn.qkv_proj.weight").is_none());
        assert!(
            gguf.tensor("model.layers.0.self_attn.qkv_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.layers.0.mlp.gate_up_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mistral_config_with_w_pack_and_w1w3_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-w-pack-aliases");
        write_tiny_mistral_alternate_packed_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.self_attn.q_proj.weight").is_none());
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.gate_proj.weight").is_none());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_attn_qkv_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-attn-qkv-aliases");
        write_tiny_mistral_attn_qkv_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.self_attn.qkv_proj.weight").is_none());
        assert!(gguf.tensor("blk.0.attn.qkv.weight").is_some());
        assert!(gguf.tensor("blk.0.attn.qkv.bias").is_some());
        assert!(gguf.tensor("blk.0.attn.out_proj.weight").is_some());
    }

    #[test]
    fn parses_mistral_config_with_transformer_h_packed_alias_tensor_layout() {
        let path = tempfile_path("tiny-mistral-transformer-h-packed-aliases");
        write_tiny_mistral_transformer_h_packed_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.family, ModelFamily::Mistral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("transformer.ln_f.weight").is_some());
        assert!(gguf.tensor("transformer.h.0.attn.c_attn.weight").is_some());
        assert!(gguf.tensor("transformer.h.0.mlp.fc2.weight").is_some());
    }

    #[test]
    fn inspect_model_reports_mistral_family() {
        let path = tempfile_path("inspect-mistral");
        write_tiny_mistral(&path);

        let info = inspect_model(&path, Some("tiny-mistral".to_string())).unwrap();

        assert_eq!(info.id, "tiny-mistral");
        assert_eq!(info.family, ModelFamily::Mistral);
        assert_eq!(info.model_type, "mistral");
        assert_eq!(info.context_length, Some(16));
        assert!(!info.chat_template);
        assert_eq!(info.weight_shards[0].tensor_count, Some(11));
    }

    #[test]
    fn parses_mixtral_config_with_moe_tensor_layout() {
        let path = tempfile_path("tiny-mixtral");
        write_tiny_mixtral(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(3));
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert_eq!(config.default_rope_freq_base(), 10_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_mixtral_config_with_per_expert_moe_tensor_layout() {
        let path = tempfile_path("tiny-mixtral-per-expert");
        write_tiny_mixtral_per_expert(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 15);
        assert_eq!(gguf.tensors().len(), 15);
    }

    #[test]
    fn parses_mixtral_config_with_alias_per_expert_moe_tensor_layout() {
        let path = tempfile_path("tiny-mixtral-alias-per-expert");
        write_tiny_mixtral_alias_per_expert(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 15);
        assert!(gguf.tensor("blk.0.ffn_gate_inp.weight").is_none());
        assert!(gguf.tensor("blk.0.block_sparse_moe.gate.weight").is_some());
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.experts.0.w1.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mixtral_config_with_per_expert_packed_gate_up_moe_tensor_layout() {
        let path = tempfile_path("tiny-mixtral-per-expert-packed-gate-up");
        write_tiny_mixtral_per_expert_packed_gate_up(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 13);
        assert!(gguf.tensor("blk.0.ffn_gate.0.weight").is_none());
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.experts.0.gate_up_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mixtral_config_with_router_alias_per_expert_moe_tensor_layout() {
        let path = tempfile_path("tiny-mixtral-router-alias-per-expert");
        write_tiny_mixtral_router_alias_per_expert(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 15);
        assert!(gguf.tensor("blk.0.ffn_gate_inp.weight").is_none());
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.router.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mixtral_config_with_output_and_router_bias_aliases() {
        let path = tempfile_path("tiny-mixtral-output-router-bias-aliases");
        write_tiny_mixtral_output_router_bias_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("lm_head.bias").is_some());
        assert!(gguf.tensor("blk.0.block_sparse_moe.router.bias").is_some());
    }

    #[test]
    fn parses_mixtral_config_with_moe_expert_bias_aliases() {
        let path = tempfile_path("tiny-mixtral-moe-expert-bias-aliases");
        write_tiny_mixtral_moe_expert_bias_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.experts.0.w1.bias")
                .is_some()
        );
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.experts.1.w2.bias")
                .is_some()
        );
    }

    #[test]
    fn parses_mixtral_config_with_feed_forward_kind_first_expert_aliases() {
        let path = tempfile_path("tiny-mixtral-feed-forward-kind-first-experts");
        write_tiny_mixtral_feed_forward_kind_first_experts(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 15);
        assert!(
            gguf.tensor("blk.0.feed_forward.block_sparse_moe.experts.w1.0.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("blk.0.feed_forward.block_sparse_moe.gate.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mixtral_config_with_mlp_block_sparse_moe_aliases() {
        let path = tempfile_path("tiny-mixtral-mlp-block-sparse-moe-aliases");
        write_tiny_mixtral_mlp_block_sparse_moe_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 15);
        assert!(
            gguf.tensor("blk.0.mlp.block_sparse_moe.experts.0.w1.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("blk.0.mlp.block_sparse_moe.gate.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_mixtral_config_with_plural_shared_expert_aliases() {
        let path = tempfile_path("tiny-mixtral-shared-experts-plural");
        write_tiny_mixtral_shared_experts_plural(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.optional_tensors_present, 3);
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.shared_experts.w1.weight")
                .is_some()
        );
        assert!(gguf.tensor("blk.0.ffn_gate_shexp.weight").is_none());
    }

    #[test]
    fn parses_mixtral_config_with_packed_shared_expert_aliases() {
        let path = tempfile_path("tiny-mixtral-packed-shared-expert");
        write_tiny_mixtral_packed_shared_expert(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.optional_tensors_present, 2);
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.shared_experts.gate_up_proj.weight")
                .is_some()
        );
        assert!(gguf.tensor("blk.0.ffn_gate_shexp.weight").is_none());
    }

    #[test]
    fn parses_mixtral_config_with_shared_expert_gate_aliases() {
        let path = tempfile_path("tiny-mixtral-shared-expert-gate");
        write_tiny_mixtral_shared_expert_gate(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "mixtral");
        assert_eq!(config.family, ModelFamily::Mixtral);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.shared_expert_gate.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("blk.0.block_sparse_moe.shared_expert_gate.bias")
                .is_some()
        );
    }

    #[test]
    fn inspect_model_reports_mixtral_family() {
        let path = tempfile_path("inspect-mixtral");
        write_tiny_mixtral(&path);

        let info = inspect_model(&path, Some("tiny-mixtral".to_string())).unwrap();

        assert_eq!(info.id, "tiny-mixtral");
        assert_eq!(info.family, ModelFamily::Mixtral);
        assert_eq!(info.model_type, "mixtral");
        assert_eq!(info.context_length, Some(16));
        assert!(!info.chat_template);
        assert_eq!(info.weight_shards[0].tensor_count, Some(12));
    }

    #[test]
    fn parses_deepseek_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("tiny-deepseek");
        write_tiny_deepseek_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(8));
        assert_eq!(config.default_rope_freq_base(), 10_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_deepseek_config_with_moe_tensor_layout() {
        let path = tempfile_path("tiny-deepseek-moe");
        write_tiny_deepseek_moe(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.feed_forward_length, Some(3));
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn inspect_model_reports_deepseek_family() {
        let path = tempfile_path("inspect-deepseek");
        write_tiny_deepseek_dense(&path);

        let info = inspect_model(&path, Some("tiny-deepseek".to_string())).unwrap();

        assert_eq!(info.id, "tiny-deepseek");
        assert_eq!(info.family, ModelFamily::DeepSeek);
        assert_eq!(info.model_type, "deepseek");
        assert_eq!(info.context_length, Some(16));
        assert!(!info.chat_template);
        assert_eq!(info.weight_shards[0].tensor_count, Some(11));
    }

    #[test]
    fn rejects_deepseek_mla_tensor_layout_with_clear_error() {
        let path = tempfile_path("deepseek-mla");
        write_tiny_deepseek_mla(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf.qwen_config().unwrap_err().to_string();

        assert!(err.contains("unsupported DeepSeek GGUF tensor layout"));
        assert!(err.contains("MLA attention"));
        assert!(err.contains("blk.0.attn_kv_a_mqa.weight"));
    }

    #[test]
    fn rejects_deepseek_mla_metadata_with_exact_key() {
        let path = tempfile_path("deepseek-mla-metadata");
        write_tiny_deepseek_mla_metadata(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf.qwen_config().unwrap_err().to_string();

        assert!(err.contains("unsupported DeepSeek GGUF metadata"));
        assert!(err.contains("deepseek2.attention.q_lora_rank"));
        assert!(err.contains("MLA attention"));
    }

    #[test]
    fn parses_deepseek_mla_metadata_with_split_attention_tensor_layout() {
        let path = tempfile_path("deepseek-mla-metadata-split");
        write_tiny_deepseek_mla_metadata_split(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek2");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn parses_deepseek_mla_metadata_with_packed_attention_tensor_layout() {
        let path = tempfile_path("deepseek-mla-metadata-packed");
        write_tiny_deepseek_mla_metadata_packed(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek2");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_some());
    }

    #[test]
    fn parses_deepseek_mla_sidecar_tensor_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("deepseek-mla-sidecar-split");
        write_tiny_deepseek_mla_sidecar_split(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek2");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.attn_kv_a_mqa.weight").is_some());
    }

    #[test]
    fn parses_deepseek_true_mla_with_attention_alias_tensor_layout() {
        let path = tempfile_path("deepseek-true-mla-attention-aliases");
        write_tiny_deepseek_true_mla_attention_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek2");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert!(config.attention_mla_tensor_layout);
        assert_eq!(config.attention_q_lora_rank, Some(2));
        assert_eq!(config.attention_kv_lora_rank, Some(2));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("transformer.encoder.layers.0.attention.q_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("transformer.encoder.layers.0.attention.kv_a_proj_with_mqa.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_deepseek_true_mla_with_self_attn_alias_tensor_layout() {
        let path = tempfile_path("deepseek-true-mla-self-attn-aliases");
        write_tiny_deepseek_true_mla_self_attn_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "deepseek2");
        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert!(config.attention_mla_tensor_layout);
        assert_eq!(config.attention_q_lora_rank, Some(2));
        assert_eq!(config.attention_kv_lora_rank, Some(2));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("model.layers.0.self_attn.q_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.layers.0.self_attn.kv_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.layers.0.self_attn.o_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("tiny-glm");
        write_tiny_glm_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(8));
        assert_eq!(config.default_rope_freq_base(), 1_000_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_glm_config_with_transformer_style_alias_tensor_layout() {
        let path = tempfile_path("tiny-glm-transformer-aliases");
        write_tiny_glm_transformer_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("transformer.embedding.word_embeddings.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("transformer.encoder.layers.0.self_attention.query_key_value.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("transformer.encoder.layers.0.mlp.dense_h_to_4h.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_config_with_gpt_neox_style_alias_tensor_layout() {
        let path = tempfile_path("tiny-glm-gpt-neox-aliases");
        write_tiny_glm_gpt_neox_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("gpt_neox.embed_in.weight").is_some());
        assert!(
            gguf.tensor("gpt_neox.layers.0.attention.query_key_value.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("gpt_neox.layers.0.mlp.dense_h_to_4h.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_config_with_model_transformer_w_alias_tensor_layout() {
        let path = tempfile_path("tiny-glm-model-transformer-w-aliases");
        write_tiny_glm_model_transformer_w_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("model.transformer.wte.weight").is_some());
        assert!(
            gguf.tensor("model.transformer.layers.0.self_attention.Wq.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.transformer.layers.0.mlp.w1.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_mla_metadata_with_split_attention_tensor_layout() {
        let path = tempfile_path("glm-mla-metadata-split");
        write_tiny_glm_mla_metadata_split(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn parses_glm_mla_metadata_with_packed_attention_tensor_layout() {
        let path = tempfile_path("glm-mla-metadata-packed");
        write_tiny_glm_mla_metadata_packed(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_some());
    }

    #[test]
    fn parses_glm_mla_sidecar_tensor_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("glm-mla-sidecar-split");
        write_tiny_glm_mla_sidecar_split(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("blk.0.attn_kv_a_proj_with_mqa.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_true_mla_with_self_attention_alias_tensor_layout() {
        let path = tempfile_path("glm-true-mla-self-attention-aliases");
        write_tiny_glm_true_mla_self_attention_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert!(config.attention_mla_tensor_layout);
        assert_eq!(config.attention_q_lora_rank, Some(2));
        assert_eq!(config.attention_kv_lora_rank, Some(2));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("transformer.encoder.layers.0.self_attention.q_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("transformer.encoder.layers.0.self_attention.kv_b_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_true_mla_with_attention_alias_tensor_layout() {
        let path = tempfile_path("glm-true-mla-attention-aliases");
        write_tiny_glm_true_mla_attention_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert!(config.attention_mla_tensor_layout);
        assert_eq!(config.attention_q_lora_rank, Some(2));
        assert_eq!(config.attention_kv_lora_rank, Some(2));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("transformer.encoder.layers.0.attention.q_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("transformer.encoder.layers.0.attention.kv_a_proj_with_mqa.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_true_mla_with_self_attn_alias_tensor_layout() {
        let path = tempfile_path("glm-true-mla-self-attn-aliases");
        write_tiny_glm_true_mla_self_attn_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert!(config.attention_mla_tensor_layout);
        assert_eq!(config.attention_q_lora_rank, Some(2));
        assert_eq!(config.attention_kv_lora_rank, Some(2));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("model.layers.0.self_attn.q_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.layers.0.self_attn.kv_a_proj.weight")
                .is_some()
        );
        assert!(
            gguf.tensor("model.layers.0.self_attn.o_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_glm_moe_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("glm-moe");
        write_tiny_glm_moe(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4moe");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(3));
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert_eq!(config.default_rope_freq_base(), 1_000_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
    }

    /// `expert_group_count = 1` (GLM-5.2's shape) parses and stays loadable;
    /// the guard must only reject actual group-limited routing.
    #[test]
    fn accepts_single_expert_group_metadata() {
        let path = tempfile_path("glm-moe-group1");
        write_tiny_glm_moe_grouped(&path, 1, 1);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        assert_eq!(config.expert_group_count, Some(1));
        assert_eq!(config.expert_group_used_count, Some(1));
        let validation = gguf.validate_qwen_tensors().unwrap();
        assert!(validation.valid, "{:?}", validation.errors);
    }

    /// Group-limited routing (DeepSeek-V3 n_group > 1) is unimplemented in
    /// the qwen MoE paths; loading such a checkpoint must fail loudly rather
    /// than route experts silently wrong.
    #[test]
    fn rejects_group_limited_expert_routing() {
        let path = tempfile_path("glm-moe-group4");
        write_tiny_glm_moe_grouped(&path, 4, 2);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        assert_eq!(config.expert_group_count, Some(4));
        assert_eq!(config.expert_group_used_count, Some(2));

        let validation = gguf.qwen_tensor_validation().unwrap();
        assert!(!validation.valid);
        let err = gguf.validate_qwen_tensors().unwrap_err().to_string();
        assert!(
            err.contains("group-limited expert routing is unimplemented"),
            "{err}"
        );
        assert!(err.contains("expert_group_count = 4"), "{err}");
    }

    #[test]
    fn rejects_expert_group_used_count_above_group_count() {
        let path = tempfile_path("glm-moe-group-used");
        write_tiny_glm_moe_grouped(&path, 1, 3);

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf.validate_qwen_tensors().unwrap_err().to_string();
        assert!(
            err.contains("expert_group_used_count 3 must be <= expert_group_count 1"),
            "{err}"
        );
    }

    #[test]
    fn parses_glm_flash_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("glm-flash");
        write_tiny_glm_flash_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4flash");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(8));
        assert_eq!(config.expert_count, None);
        assert_eq!(config.default_rope_freq_base(), 1_000_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn parses_glm_flash_moe_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("glm-flash-moe");
        write_tiny_glm_flash_moe(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "glm4flash");
        assert_eq!(config.family, ModelFamily::GlmFlash);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(3));
        assert_eq!(config.expert_feed_forward_length, Some(3));
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert_eq!(config.default_rope_freq_base(), 1_000_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn inspect_model_reports_glm_family() {
        let path = tempfile_path("inspect-glm");
        write_tiny_glm_dense(&path);

        let info = inspect_model(&path, Some("tiny-glm".to_string())).unwrap();

        assert_eq!(info.id, "tiny-glm");
        assert_eq!(info.family, ModelFamily::GlmFlash);
        assert_eq!(info.model_type, "glm4");
        assert_eq!(info.context_length, Some(16));
        assert!(!info.chat_template);
        assert_eq!(info.weight_shards[0].tensor_count, Some(11));
    }

    #[test]
    fn rejects_glm_mla_tensor_layout_with_clear_error() {
        let path = tempfile_path("glm-mla");
        write_tiny_glm_mla(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf.qwen_config().unwrap_err().to_string();

        assert!(err.contains("unsupported GLM GGUF tensor layout"));
        assert!(err.contains("MLA attention"));
        assert!(err.contains("blk.0.attn_kv_a_proj_with_mqa.weight"));
    }

    #[test]
    fn parses_gemma_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("tiny-gemma");
        write_tiny_gemma(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "gemma");
        assert_eq!(config.family, ModelFamily::Gemma);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.default_rope_freq_base(), 10_000.0);
        assert!(validation.valid);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_gemma2_config_with_post_norms_and_softcapping() {
        let path = tempfile_path("tiny-gemma2");
        write_tiny_gemma2(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "gemma2");
        assert_eq!(config.family, ModelFamily::Gemma);
        assert!(config.is_gemma());
        assert!(!config.is_gemma3());
        assert_eq!(config.attention_sliding_window, Some(4096));
        assert_eq!(config.attn_logit_softcapping, Some(50.0));
        assert_eq!(config.final_logit_softcapping, Some(30.0));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(gguf.tensor("blk.0.post_ffw_norm.weight").is_some());
    }

    #[test]
    fn parses_gemma_config_with_pre_feedforward_alias_tensor_layout() {
        let path = tempfile_path("tiny-gemma-pre-feedforward-aliases");
        write_tiny_gemma_pre_feedforward_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "gemma");
        assert_eq!(config.family, ModelFamily::Gemma);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("model.layers.0.pre_feedforward_layernorm.weight")
                .is_some()
        );
    }

    #[test]
    fn parses_gemma_config_with_post_feedforward_alias_tensor_layout() {
        let path = tempfile_path("tiny-gemma-post-feedforward-aliases");
        write_tiny_gemma_post_feedforward_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "gemma");
        assert_eq!(config.family, ModelFamily::Gemma);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(
            gguf.tensor("model.layers.0.post_feedforward_layernorm.weight")
                .is_some()
        );
        assert!(gguf.tensor("model.layers.0.self_attn.out.weight").is_some());
        assert!(gguf.tensor("model.layers.0.mlp.proj.weight").is_some());
    }

    #[test]
    fn parses_gemma_config_with_dense_bias_alias_tensor_layout() {
        let path = tempfile_path("tiny-gemma-dense-bias-aliases");
        write_tiny_gemma_dense_bias_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "gemma");
        assert_eq!(config.family, ModelFamily::Gemma);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.optional_tensors_present, 4);
        assert!(
            gguf.tensor("model.layers.0.self_attn.o_proj.bias")
                .is_some()
        );
        assert!(gguf.tensor("model.layers.0.mlp.gate_proj.bias").is_some());
        assert!(gguf.tensor("model.layers.0.mlp.up_proj.bias").is_some());
        assert!(gguf.tensor("model.layers.0.mlp.down_proj.bias").is_some());
    }

    #[test]
    fn inspect_model_reports_gemma_family() {
        let path = tempfile_path("inspect-gemma");
        write_tiny_gemma(&path);

        let info = inspect_model(&path, Some("tiny-gemma".to_string())).unwrap();

        assert_eq!(info.id, "tiny-gemma");
        assert_eq!(info.family, ModelFamily::Gemma);
        assert_eq!(info.model_type, "gemma");
        assert_eq!(info.context_length, Some(16));
        assert!(!info.chat_template);
        assert_eq!(info.weight_shards[0].tensor_count, Some(11));
    }

    #[test]
    fn parses_phi_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("tiny-phi");
        write_tiny_phi(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.default_rope_freq_base(), 10_000.0);
        assert!(validation.valid);
        assert_eq!(gguf.tensor("token_embd.weight").unwrap().bytes.len(), 16);
    }

    #[test]
    fn parses_phi_config_with_split_qkv_bias_layout() {
        let path = tempfile_path("tiny-phi-split-qkv-bias");
        write_tiny_phi_split_qkv_biases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.optional_tensors_present, 3);
    }

    #[test]
    fn inspect_model_reports_phi_family() {
        let path = tempfile_path("inspect-phi");
        write_tiny_phi(&path);

        let info = inspect_model(&path, Some("tiny-phi".to_string())).unwrap();

        assert_eq!(info.id, "tiny-phi");
        assert_eq!(info.family, ModelFamily::Phi);
        assert_eq!(info.model_type, "phi3");
        assert_eq!(info.context_length, Some(16));
        assert!(!info.chat_template);
        assert_eq!(info.weight_shards[0].tensor_count, Some(11));
    }

    #[test]
    fn parses_phi_config_with_packed_projection_tensor_layout() {
        let path = tempfile_path("tiny-phi-packed-qkv");
        write_tiny_phi_packed(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn parses_phi_config_with_packed_qkv_bias_layout() {
        let path = tempfile_path("tiny-phi-packed-qkv-bias");
        write_tiny_phi_packed_qkv_bias(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.optional_tensors_present, 1);
    }

    #[test]
    fn parses_phi_config_with_packed_qkv_alias_layout() {
        let path = tempfile_path("tiny-phi-packed-qkv-alias");
        write_tiny_phi_packed_aliases(
            &path,
            "blk.0.self_attn.qkv_proj.weight",
            None,
            "blk.0.mlp.gate_up_proj.weight",
        );

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.attn_qkv.weight").is_none());
        assert!(gguf.tensor("blk.0.self_attn.qkv_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.ffn_gate_up.weight").is_none());
        assert!(gguf.tensor("blk.0.mlp.gate_up_proj.weight").is_some());
    }

    #[test]
    fn parses_phi_config_with_mixer_packed_qkv_layout() {
        let path = tempfile_path("tiny-phi-mixer-packed-qkv");
        write_tiny_phi_mixer_packed_qkv(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.mixer.Wqkv.weight").is_some());
        assert!(gguf.tensor("blk.0.mixer.out_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.fc2.weight").is_some());
    }

    #[test]
    fn parses_phi_config_with_packed_qkv_bias_alias_layout() {
        let path = tempfile_path("tiny-phi-packed-qkv-bias-alias");
        write_tiny_phi_packed_aliases(
            &path,
            "blk.0.query_key_value.weight",
            Some("blk.0.query_key_value.bias"),
            "blk.0.mlp.up_gate_proj.weight",
        );

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.optional_tensors_present, 1);
        assert!(gguf.tensor("blk.0.attn_qkv.bias").is_none());
        assert!(gguf.tensor("blk.0.query_key_value.bias").is_some());
    }

    #[test]
    fn parses_phi_config_with_packed_ffn_gate_up_layout() {
        let path = tempfile_path("tiny-phi-packed-ffn-gate-up");
        write_tiny_phi_packed_ffn_gate_up(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn parses_phi_config_with_packed_ffn_up_gate_layout() {
        let path = tempfile_path("tiny-phi-packed-ffn-up-gate");
        write_tiny_phi_packed_ffn_up_gate(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
    }

    #[test]
    fn parses_phi_config_with_fused_ffn_up_layout() {
        // llama.cpp stores Phi-3's SwiGLU gate+up fused under `ffn_up` at 2x width
        // with no separate `ffn_gate`. It must validate as a packed FFN layout.
        let path = tempfile_path("tiny-phi-fused-ffn-up");
        write_tiny_phi_fused_ffn_up(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "phi3");
        assert_eq!(config.family, ModelFamily::Phi);
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.ffn_gate.weight").is_none());
    }

    #[test]
    fn inspect_model_reports_gguf_chat_template() {
        let path = tempfile_path("tiny-llama-template");
        write_tiny_llama(&path);

        let info = inspect_model(&path, None).unwrap();
        let gguf = GgufFile::open(&path).unwrap();

        assert!(info.chat_template);
        assert!(info.tokenizer.tokenizer_config);
        assert_eq!(gguf.chat_template(), Some(LLAMA3_CHAT_TEMPLATE));
    }

    #[test]
    fn inspect_model_builds_openai_model_info() {
        let path = tempfile_path("inspect");
        write_tiny_qwen(&path, 1, 0);

        let info = inspect_model(&path, Some("tiny-qwen".to_string())).unwrap();

        assert_eq!(info.id, "tiny-qwen");
        assert_eq!(info.family, ModelFamily::Qwen2);
        assert_eq!(info.context_length, Some(16));
        assert_eq!(info.weight_shards[0].tensor_count, Some(1));
    }

    #[test]
    fn tokenizer_encodes_and_decodes_byte_level_bpe() {
        let path = tempfile_path("tokenizer");
        write_tokenizer_fixture(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let tokenizer = gguf.tokenizer().unwrap();

        assert_eq!(tokenizer.encode("hello!").unwrap(), vec![8, 4]);
        assert_eq!(
            tokenizer.encode("hello<|endoftext|>!").unwrap(),
            vec![8, 10, 4]
        );
        assert_eq!(
            tokenizer.decode_with_options(&[8, 9, 4], false).unwrap(),
            "hello world!"
        );
        assert_eq!(tokenizer.decode(&[8, 10]).unwrap(), "hello");
        assert_eq!(tokenizer.summary().merge_count, 4);
    }

    /// The heap/linked-list BPE must be observationally identical to the naive
    /// algorithm it replaced (repeatedly merge the lowest-rank pair, leftmost
    /// among ties): sweep random strings over an alphabet chosen to force
    /// overlapping candidates, equal-rank ties, and merge chains.
    #[test]
    fn bpe_heap_encoder_matches_naive_reference() {
        fn naive_apply_bpe(tokenizer: &GgufTokenizer, encoded: &str) -> Vec<String> {
            let mut symbols = encoded.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
            if symbols.len() < 2 || tokenizer.merge_ranks.is_empty() {
                return symbols;
            }
            loop {
                let mut best: Option<(usize, usize)> = None;
                for idx in 0..symbols.len().saturating_sub(1) {
                    let pair = (symbols[idx].clone(), symbols[idx + 1].clone());
                    let Some(rank) = tokenizer.merge_ranks.get(&pair).copied() else {
                        continue;
                    };
                    if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                        best = Some((rank, idx));
                    }
                }
                let Some((_, idx)) = best else {
                    break;
                };
                let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                symbols.splice(idx..idx + 2, [merged]);
                if symbols.len() < 2 {
                    break;
                }
            }
            symbols
        }

        let path = tempfile_path("bpe-heap-property");
        // Merge table with chains (a+a=aa, aa+a, aa+aa), equal-length
        // alternatives, and cross-symbol follow-ups; tokens cover every merge
        // result so encode() resolves without the unknown fallback.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 4);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
        let tokens = [
            "a",
            "b",
            "c",
            "d",
            " ",
            "\u{0120}",
            "aa",
            "ab",
            "ba",
            "aaa",
            "aaaa",
            "ab\u{0120}",
            "\u{0120}a",
            "\u{0120}ab",
            "bc",
            "abc",
            "cd",
            "abcd",
            "<|unk|>",
        ];
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &tokens);
        let merges = [
            "a a",
            "a b",
            "b a",
            "aa a",
            "aa aa",
            "b c",
            "ab c",
            "c d",
            "abc d",
            "\u{0120} a",
            "\u{0120}a b",
            "ab \u{0120}",
        ];
        write_kv_string_array(&mut bytes, "tokenizer.ggml.merges", &merges);
        write_kv_u32(&mut bytes, "tokenizer.ggml.unknown_token_id", 18);
        std::fs::write(&path, bytes).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let tokenizer = gguf.tokenizer().unwrap();

        let alphabet = ['a', 'b', 'c', 'd', ' ', 'e'];
        let mut state = 0x9e37_79b9u64;
        for case in 0..400 {
            let len = (lcg_next(&mut state) % 60) as usize + (case % 3);
            let text: String = (0..len)
                .map(|_| alphabet[(lcg_next(&mut state) % alphabet.len() as u64) as usize])
                .collect();
            let encoded = encode_byte_level_text(text.as_bytes());
            let fast: Vec<&str> = tokenizer.apply_bpe(&encoded);
            let naive = naive_apply_bpe(&tokenizer, &encoded);
            assert_eq!(
                fast, naive,
                "heap BPE diverged from naive reference for {text:?}"
            );
        }

        // Adversarial: long uniform runs (maximal equal-rank tie pressure) and
        // a run long enough that an O(n^2) reintroduction would be obvious.
        for text in [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "abababababababab",
            "aaab aaab aaab",
        ] {
            let encoded = encode_byte_level_text(text.as_bytes());
            let fast: Vec<&str> = tokenizer.apply_bpe(&encoded);
            let naive = naive_apply_bpe(&tokenizer, &encoded);
            assert_eq!(fast, naive, "heap BPE diverged for {text:?}");
        }
    }

    #[test]
    fn tokenizer_encodes_and_decodes_llama_sentencepiece() {
        let path = tempfile_path("llama-tokenizer");
        write_llama_tokenizer_fixture(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let tokenizer = gguf.tokenizer().unwrap();

        assert_eq!(
            tokenizer.encode("hello world\n!").unwrap(),
            vec![4, 5, 7, 8]
        );
        assert_eq!(tokenizer.encode("<s>hello</s>").unwrap(), vec![1, 4, 2]);
        assert_eq!(
            tokenizer.decode_with_options(&[4, 5, 7, 8], false).unwrap(),
            " hello world\n!"
        );
        assert_eq!(tokenizer.decode(&[1, 4, 2]).unwrap(), " hello");
        assert!(tokenizer.summary().has_scores);
    }

    // Deterministic LCG so the streaming-vs-batch property sweep is reproducible.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    fn assert_streaming_decode_matches_batch(tokenizer: &GgufTokenizer, seeds: u64) {
        let vocab = tokenizer.summary().token_count as u64;
        for skip_special in [true, false] {
            for seed in 0..seeds {
                let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                let len = (lcg_next(&mut state) % 40 + 1) as usize;
                let ids = (0..len)
                    .map(|_| (lcg_next(&mut state) % vocab) as u32)
                    .collect::<Vec<_>>();
                let batch = tokenizer.decode_with_options(&ids, skip_special).unwrap();
                let mut decoder = tokenizer.streaming_decoder(skip_special);
                let mut streamed = String::new();
                for id in &ids {
                    streamed.push_str(&decoder.push(tokenizer, *id).unwrap());
                }
                streamed.push_str(&decoder.finish());
                assert_eq!(
                    streamed, batch,
                    "streamed decode diverged from batch decode for ids {ids:?} (skip_special={skip_special})"
                );
            }
        }
    }

    fn write_streaming_bpe_tokenizer_fixture(path: &Path) {
        // Byte-level BPE vocab with single-byte pieces that split multi-byte UTF-8
        // (emoji F0 9F 98 80, é C3 A9) plus a stray continuation byte for the
        // invalid/U+FFFD path.
        let byte_pieces: Vec<String> = [
            vec![0xF0u8],
            vec![0x9F],
            vec![0x98],
            vec![0x80],
            vec![0xF0, 0x9F],
            vec![0xC3, 0xA9],
        ]
        .iter()
        .map(|bytes| encode_byte_level_text(bytes))
        .collect();
        let mut tokens: Vec<&str> = vec!["h", "e", "hello", "\u{0120}world", "!", "<|endoftext|>"];
        tokens.extend(byte_pieces.iter().map(String::as_str));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 4);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &tokens);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 5);
        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    fn write_streaming_spm_tokenizer_fixture(path: &Path) {
        // SentencePiece vocab with byte-fallback markers for a 3-byte char (我 =
        // E6 88 91), plus adversarial regular pieces that only form a marker when
        // concatenated ("<0x" + "AB>") to exercise the cross-piece carry.
        let tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}",
            "\u{2581}hello",
            "hello",
            "<0x0A>",
            "<0xE6>",
            "<0x88>",
            "<0x91>",
            "<0x",
            "AB>",
            "<0xAB",
            "!",
            "a",
        ];
        let scores = [
            -100.0, 0.0, 0.0, -5.0, -0.1, -1.0, -10.0, -10.0, -10.0, -10.0, -2.0, -2.0, -2.0, -0.1,
            -0.1,
        ];
        let token_types = [2, 3, 3, 1, 1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 1];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 7);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_string(&mut bytes, "tokenizer.ggml.model", "llama");
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &tokens);
        write_kv_f32_array(&mut bytes, "tokenizer.ggml.scores", &scores);
        write_kv_i32_array(&mut bytes, "tokenizer.ggml.token_type", &token_types);
        write_kv_u32(&mut bytes, "tokenizer.ggml.bos_token_id", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);
        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    fn write_streaming_plain_tokenizer_fixture(path: &Path) {
        let tokens = ["<unk>", "\u{2581}hi", "there", "\u{2581}", "!"];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 4);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_string(&mut bytes, "tokenizer.ggml.model", "wordpiece");
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &tokens);
        write_kv_u32(&mut bytes, "tokenizer.ggml.unknown_token_id", 0);
        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn streaming_decoder_matches_batch_decode_byte_level_bpe() {
        let path = tempfile_path("streaming-bpe-tokenizer");
        write_streaming_bpe_tokenizer_fixture(&path);
        let tokenizer = GgufFile::open(&path).unwrap().tokenizer().unwrap();

        // Emoji split across four single-byte tokens: nothing emitted until the
        // final byte completes the character.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 6).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 7).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 8).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 9).unwrap(), "\u{1F600}");
        // Special token mid-stream is skipped without disturbing held bytes.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 10).unwrap(), ""); // F0 9F held
        assert_eq!(decoder.push(&tokenizer, 5).unwrap(), ""); // <|endoftext|> skipped
        assert_eq!(decoder.push(&tokenizer, 8).unwrap(), ""); // 98 held
        assert_eq!(decoder.push(&tokenizer, 9).unwrap(), "\u{1F600}");
        // A stray continuation byte is definitively invalid on arrival (it can never
        // start a character), so U+FFFD is emitted immediately — same as batch.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 8).unwrap(), "\u{FFFD}");
        assert_eq!(decoder.push(&tokenizer, 0).unwrap(), "h");
        // Truncated mid-character: the held tail is dropped at finish.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 2).unwrap(), "hello");
        assert_eq!(decoder.push(&tokenizer, 6).unwrap(), "");
        assert_eq!(decoder.finish(), "");

        assert_streaming_decode_matches_batch(&tokenizer, 500);
    }

    #[test]
    fn streaming_decoder_matches_batch_decode_sentencepiece() {
        let path = tempfile_path("streaming-spm-tokenizer");
        write_streaming_spm_tokenizer_fixture(&path);
        let tokenizer = GgufFile::open(&path).unwrap().tokenizer().unwrap();

        // Byte-fallback run spanning three tokens completes a 3-byte character.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 7).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 8).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 9).unwrap(), "\u{6211}");
        // A marker split across regular pieces ("<0x" + "AB>") is a byte-fallback
        // in the batch decoder's concatenation, so it must be here too. Byte 0xAB is
        // a continuation byte — definitively invalid alone — so U+FFFD, as in batch.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 10).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 11).unwrap(), "\u{FFFD}");
        assert_eq!(decoder.push(&tokenizer, 13).unwrap(), "!");
        // An incomplete byte run is dropped when a regular character flushes it.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 7).unwrap(), "");
        assert_eq!(decoder.push(&tokenizer, 13).unwrap(), "!");
        // A held would-be marker that never completes is emitted as text at finish.
        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 12).unwrap(), ""); // "<0xAB" carried
        assert_eq!(decoder.finish(), "<0xAB");

        assert_streaming_decode_matches_batch(&tokenizer, 500);
    }

    #[test]
    fn streaming_decoder_matches_batch_decode_plain() {
        let path = tempfile_path("streaming-plain-tokenizer");
        write_streaming_plain_tokenizer_fixture(&path);
        let tokenizer = GgufFile::open(&path).unwrap().tokenizer().unwrap();

        let mut decoder = tokenizer.streaming_decoder(true);
        assert_eq!(decoder.push(&tokenizer, 1).unwrap(), " hi");
        assert_eq!(decoder.push(&tokenizer, 2).unwrap(), "there");
        assert_eq!(decoder.finish(), "");

        assert_streaming_decode_matches_batch(&tokenizer, 200);
    }

    #[test]
    fn reports_incomplete_qwen_tensor_table() {
        let path = tempfile_path("incomplete-qwen");
        write_tiny_qwen(&path, 1, 0);

        let validation = GgufFile::open(&path)
            .unwrap()
            .qwen_tensor_validation()
            .unwrap();

        assert!(!validation.valid);
        assert!(
            validation
                .errors
                .iter()
                .any(|error| error.contains("missing required tensor output_norm.weight"))
        );
    }

    #[test]
    fn validates_dense_qwen_tensor_table() {
        let path = tempfile_path("full-qwen");
        write_full_qwen(&path);

        let validation = GgufFile::open(&path)
            .unwrap()
            .validate_qwen_tensors()
            .unwrap();

        assert!(validation.valid);
        assert_eq!(validation.required_tensors, 11);
        assert_eq!(validation.optional_tensors_present, 0);
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn validates_dense_qwen_f32_tensor_table() {
        let path = tempfile_path("full-qwen-f32");
        write_full_qwen_f32(&path);

        let validation = GgufFile::open(&path)
            .unwrap()
            .validate_qwen_tensors()
            .unwrap();

        assert!(validation.valid);
        assert_eq!(validation.required_tensors, 11);
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn validates_qwen_moe_tensor_table() {
        let path = tempfile_path("moe-qwen");
        write_moe_qwen(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3moe");
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid);
        assert_eq!(validation.required_tensors, 12);
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn validates_qwen_moe_packed_gate_up_tensor_table() {
        let path = tempfile_path("moe-qwen-packed-gate-up");
        write_moe_qwen_packed_gate_up(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3moe");
        assert_eq!(config.expert_count, Some(2));
        assert_eq!(config.expert_used_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn parses_qwen_next_dense_config_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("qwen-next-dense");
        write_qwen_next_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert_eq!(config.context_length, 16);
        assert_eq!(config.embedding_length, 4);
        assert_eq!(config.feed_forward_length, Some(8));
        assert_eq!(config.default_rope_freq_base(), 1_000_000.0);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
    }

    #[test]
    fn parses_qwen_next_dense_with_attention_head_norm_aliases() {
        let path = tempfile_path("qwen-next-attention-head-norm-aliases");
        write_qwen_next_attention_head_norm_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert_eq!(validation.optional_tensors_present, 2);
        assert!(
            gguf.tensor("blk.0.self_attention.q_layernorm.weight")
                .is_some()
        );
        assert!(gguf.tensor("blk.0.attention.k_norm.weight").is_some());
    }

    #[test]
    fn parses_qwen_next_gated_attention_tensor_layout() {
        let path = tempfile_path("qwen-next-gated-attention");
        write_qwen_next_gated_attention_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
        assert_eq!(
            qwen_dense_gated_attention_q_weight_name(&gguf, "blk.0", 4, 4).as_deref(),
            Some("blk.0.attn_q.weight")
        );
    }

    #[test]
    fn rejects_qwen_next_ssm_metadata_with_exact_key() {
        let path = tempfile_path("qwen-next-ssm-metadata");
        write_qwen_next_ssm_metadata(&path);

        let err = GgufFile::open(&path)
            .unwrap()
            .qwen_config()
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsupported Qwen GGUF metadata"));
        assert!(err.contains("qwen3next.ssm.state_size"));
        assert!(err.contains("unsupported feature SSM"));
    }

    #[test]
    fn parses_qwen_next_ssm_metadata_with_dense_tensor_layout() {
        let path = tempfile_path("qwen-next-ssm-metadata-dense");
        write_qwen_next_ssm_metadata_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
    }

    #[test]
    fn parses_qwen_next_ssm_metadata_with_packed_tensor_layout() {
        let path = tempfile_path("qwen-next-ssm-metadata-packed");
        write_qwen_next_ssm_metadata_packed(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 8);
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_some());
    }

    #[test]
    fn parses_qwen_next_ssm_metadata_with_hf_packed_alias_tensor_layout() {
        let path = tempfile_path("qwen-next-ssm-metadata-hf-packed-aliases");
        write_qwen_next_ssm_metadata_hf_packed_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 8);
        assert!(gguf.tensor("blk.0.attention.qkv_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.gate_up_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_none());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_none());
    }

    #[test]
    fn parses_qwen_next_recurrent_ssm_tensor_layout() {
        let path = tempfile_path("qwen-next-recurrent-ssm");
        write_qwen_next_recurrent_ssm(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert!(config.recurrent_ssm_tensor_layout);
        assert_eq!(config.ssm_conv_kernel, Some(1));
        assert_eq!(config.ssm_inner_size, Some(2));
        assert_eq!(config.ssm_state_size, Some(2));
        assert_eq!(config.ssm_time_step_rank, Some(1));
        assert_eq!(config.ssm_group_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 14);
    }

    #[test]
    fn parses_qwen_next_recurrent_ssm_optimized_tensor_layout() {
        let path = tempfile_path("qwen-next-recurrent-ssm-optimized");
        write_qwen_next_recurrent_ssm_optimized(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen3next");
        assert_eq!(config.family, ModelFamily::Qwen3);
        assert!(config.recurrent_ssm_tensor_layout);
        assert_eq!(config.ssm_conv_kernel, Some(1));
        assert_eq!(config.ssm_inner_size, Some(2));
        assert_eq!(config.ssm_state_size, Some(2));
        assert_eq!(config.ssm_time_step_rank, Some(1));
        assert_eq!(config.ssm_group_count, Some(1));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 15);
        assert!(gguf.tensor("blk.0.ssm_in.weight").is_none());
        assert!(gguf.tensor("blk.0.attn_qkv.weight").is_some());
        assert!(gguf.tensor("blk.0.attn_gate.weight").is_some());
    }

    #[test]
    fn parses_qwen_ssm_metadata_with_dense_tensor_layout() {
        let path = tempfile_path("qwen-ssm-metadata-dense");
        write_qwen_ssm_metadata_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.family, ModelFamily::Qwen2);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
    }

    #[test]
    fn parses_qwen_ssm_metadata_with_packed_tensor_layout() {
        let path = tempfile_path("qwen-ssm-metadata-packed");
        write_qwen_ssm_metadata_packed(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.family, ModelFamily::Qwen2);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 8);
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_some());
    }

    #[test]
    fn parses_qwen_ssm_metadata_with_hf_packed_alias_tensor_layout() {
        let path = tempfile_path("qwen-ssm-metadata-hf-packed-aliases");
        write_qwen_ssm_metadata_hf_packed_aliases(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.family, ModelFamily::Qwen2);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 8);
        assert!(gguf.tensor("blk.0.attention.qkv_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.mlp.gate_up_proj.weight").is_some());
        assert!(gguf.tensor("blk.0.self_attn.W_pack.weight").is_none());
        assert!(gguf.tensor("blk.0.mlp.w1w3.weight").is_none());
    }

    #[test]
    fn parses_qwen_ssm_sidecar_tensor_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("qwen-ssm-sidecar-dense");
        write_qwen_ssm_sidecar_dense(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.family, ModelFamily::Qwen2);
        assert_eq!(config.feed_forward_length, Some(8));
        assert!(validation.valid, "{:?}", validation.errors);
        assert!(gguf.tensor("blk.0.ssm_in.weight").is_some());
    }

    #[test]
    fn parses_qwen_unequal_custom_kv_lengths_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("qwen-custom-kv");
        write_qwen_custom_kv_lengths(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.attention_key_length, Some(2));
        assert_eq!(config.attention_value_length, Some(3));
        assert_eq!(config.attention_key_head_dim(), Some(2));
        assert_eq!(config.attention_value_head_dim(), Some(3));
        assert_eq!(config.attention_head_dim(), None);
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
    }

    #[test]
    fn parses_qwen_equal_custom_kv_lengths_with_decoder_compatible_tensor_layout() {
        let path = tempfile_path("qwen-equal-custom-kv");
        write_qwen_equal_custom_kv_lengths(&path);

        let gguf = GgufFile::open(&path).unwrap();
        let config = gguf.qwen_config().unwrap();
        let validation = gguf.validate_qwen_tensors().unwrap();

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.attention_key_length, Some(2));
        assert_eq!(config.attention_value_length, Some(2));
        assert_eq!(config.attention_head_dim(), Some(2));
        assert!(validation.valid, "{:?}", validation.errors);
        assert_eq!(validation.required_tensors, 11);
    }

    #[test]
    fn rejects_qwen_ssm_tensor_layout_with_exact_tensor() {
        let path = tempfile_path("qwen-ssm-tensor");
        write_qwen_ssm_tensor(&path);

        let err = GgufFile::open(&path)
            .unwrap()
            .qwen_config()
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsupported Qwen GGUF tensor layout"));
        assert!(err.contains("blk.0.ssm_in.weight"));
        assert!(err.contains("unsupported feature SSM"));
    }

    #[test]
    fn rejects_qwen_ssm_metadata_with_exact_key() {
        let path = tempfile_path("qwen-ssm-metadata");
        write_qwen_ssm_metadata(&path);

        let err = GgufFile::open(&path)
            .unwrap()
            .qwen_config()
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsupported Qwen GGUF metadata"));
        assert!(err.contains("qwen2.ssm.state_size"));
        assert!(err.contains("unsupported feature SSM"));
    }

    #[test]
    fn rejects_unsupported_tensor_type() {
        let cases = [(42, "future GGUF tensor type")];

        for (raw, label) in cases {
            let path = tempfile_path(&format!("unsupported-dtype-{raw}"));
            write_tiny_qwen(&path, raw, 0);

            let err = GgufFile::open(&path).unwrap_err().to_string();
            assert!(err.contains("tensor token_embd.weight"), "{raw}: {err}");
            assert!(
                err.contains(&format!("unsupported GGUF tensor type {raw} ({label})")),
                "{raw}: {err}"
            );
        }
    }

    #[test]
    fn parses_fixture_for_every_supported_tensor_type() {
        for raw in 0..=41 {
            let Ok(dtype) = GgufTensorType::from_raw(raw) else {
                continue;
            };
            let element_count = dtype.block_element_count().unwrap_or(4);
            let byte_len = dtype.byte_len(element_count).unwrap();
            let path = tempfile_path(&format!("supported-dtype-{raw}"));
            let bytes = vec![0; byte_len as usize];
            write_single_tensor_gguf(&path, "token_embd.weight", &[element_count], raw, &bytes);

            let gguf = GgufFile::open(&path).unwrap();
            let tensor = &gguf.tensors()[0];
            assert_eq!(gguf.tensors().len(), 1, "raw {raw}");
            assert_eq!(tensor.name, "token_embd.weight", "raw {raw}");
            assert_eq!(tensor.dtype, dtype, "raw {raw}");
            assert_eq!(tensor.dimensions, vec![element_count], "raw {raw}");
            assert_eq!(tensor.byte_len().unwrap(), byte_len, "raw {raw}");
            assert_eq!(
                gguf.tensor("token_embd.weight").unwrap().bytes.len(),
                byte_len as usize,
                "raw {raw}"
            );
        }
    }

    #[test]
    fn rejects_quantized_tensor_shape_with_tensor_and_type() {
        let path = tempfile_path("invalid-quantized-shape");
        write_tiny_qwen(&path, 2, 0);

        let err = GgufFile::open(&path).unwrap_err().to_string();
        assert!(err.contains("tensor token_embd.weight"));
        assert!(err.contains("dtype Q4_0"));
        assert!(err.contains("element count 8"));
    }

    #[test]
    fn decodes_supported_dense_numeric_tensors() {
        assert_eq!(
            dequantize_tensor_as_f32(&[0, 127, 128, 255], GgufTensorType::I8, 4).unwrap(),
            &[0.0, 127.0, -128.0, -1.0]
        );

        let mut i16 = Vec::new();
        i16.extend_from_slice(&(-2i16).to_le_bytes());
        i16.extend_from_slice(&513i16.to_le_bytes());
        assert_eq!(
            dequantize_tensor_as_f32(&i16, GgufTensorType::I16, 2).unwrap(),
            &[-2.0, 513.0]
        );

        let mut i32 = Vec::new();
        i32.extend_from_slice(&(-3i32).to_le_bytes());
        i32.extend_from_slice(&65_536i32.to_le_bytes());
        assert_eq!(
            dequantize_tensor_as_f32(&i32, GgufTensorType::I32, 2).unwrap(),
            &[-3.0, 65_536.0]
        );

        let mut i64 = Vec::new();
        i64.extend_from_slice(&(-4i64).to_le_bytes());
        i64.extend_from_slice(&1_048_576i64.to_le_bytes());
        assert_eq!(
            dequantize_tensor_as_f32(&i64, GgufTensorType::I64, 2).unwrap(),
            &[-4.0, 1_048_576.0]
        );

        let mut f64 = Vec::new();
        f64.extend_from_slice(&1.25f64.to_le_bytes());
        f64.extend_from_slice(&(-2.5f64).to_le_bytes());
        assert_eq!(
            dequantize_tensor_as_f32(&f64, GgufTensorType::F64, 2).unwrap(),
            &[1.25, -2.5]
        );
    }

    #[test]
    fn decodes_supported_quantized_blocks() {
        let mut q8 = Vec::new();
        q8.extend_from_slice(&f16_bits(0.5).to_le_bytes());
        q8.extend((0..32).map(|idx| idx as i8 as u8));
        let q8 = dequantize_tensor_as_f32(&q8, GgufTensorType::Q8_0, 32).unwrap();
        assert_eq!(q8[0], 0.0);
        assert_eq!(q8[31], 15.5);

        let mut q8_1 = Vec::new();
        q8_1.extend_from_slice(&f16_bits(0.5).to_le_bytes());
        q8_1.extend_from_slice(&f16_bits(1.0).to_le_bytes());
        q8_1.extend((0..32).map(|idx| idx as i8 as u8));
        let q8_1 = dequantize_tensor_as_f32(&q8_1, GgufTensorType::Q8_1, 32).unwrap();
        assert_eq!(q8_1[0], 0.0);
        assert_eq!(q8_1[31], 15.5);

        let mut q4 = Vec::new();
        q4.extend_from_slice(&f16_bits(1.0).to_le_bytes());
        q4.extend([0x8f; 16]);
        let q4_bytes = q4.clone();
        let q4 = dequantize_tensor_as_f32(&q4_bytes, GgufTensorType::Q4_0, 32).unwrap();
        assert_eq!(q4[0], 7.0);
        assert_eq!(q4[16], 0.0);
        for dtype in [
            GgufTensorType::Q4_0_4_4,
            GgufTensorType::Q4_0_4_8,
            GgufTensorType::Q4_0_8_8,
        ] {
            assert_eq!(dequantize_tensor_as_f32(&q4_bytes, dtype, 32).unwrap(), q4);
        }

        let mut q4_1 = Vec::new();
        q4_1.extend_from_slice(&f16_bits(0.5).to_le_bytes());
        q4_1.extend_from_slice(&f16_bits(1.0).to_le_bytes());
        q4_1.extend([0x8f; 16]);
        let q4_1 = dequantize_tensor_as_f32(&q4_1, GgufTensorType::Q4_1, 32).unwrap();
        assert_eq!(q4_1[0], 8.5);
        assert_eq!(q4_1[16], 5.0);

        let mut q1 = vec![0u8; 18];
        q1[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        q1[2] = 0b1010_0101;
        let q1 = dequantize_tensor_as_f32(&q1, GgufTensorType::Q1_0, 128).unwrap();
        assert_eq!(q1[0], 0.5);
        assert_eq!(q1[1], -0.5);
        assert_eq!(q1[2], 0.5);
        assert_eq!(q1[3], -0.5);
        assert_eq!(q1[4], -0.5);
        assert_eq!(q1[7], 0.5);

        let mut mxfp4 = vec![0u8; 17];
        mxfp4[0] = 128;
        mxfp4[1] = 0x91;
        mxfp4[2] = 0x5f;
        let mxfp4 = dequantize_tensor_as_f32(&mxfp4, GgufTensorType::MXFP4, 32).unwrap();
        assert_eq!(mxfp4[0], 1.0);
        assert_eq!(mxfp4[1], -12.0);
        assert_eq!(mxfp4[16], -1.0);
        assert_eq!(mxfp4[17], 6.0);

        let mut nvfp4 = vec![0u8; 36];
        nvfp4[0] = 0x38;
        nvfp4[1] = 0x40;
        nvfp4[4] = 0x95;
        nvfp4[5] = 0x2f;
        nvfp4[12] = 0x41;
        let nvfp4 = dequantize_tensor_as_f32(&nvfp4, GgufTensorType::NVFP4, 64).unwrap();
        assert_eq!(nvfp4[0], 3.0);
        assert_eq!(nvfp4[1], -6.0);
        assert_eq!(nvfp4[8], -0.5);
        assert_eq!(nvfp4[9], 1.0);
        assert_eq!(nvfp4[16], 1.0);
        assert_eq!(nvfp4[24], 4.0);

        let mut iq2xxs = vec![0u8; 66];
        iq2xxs[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq2xxs[2..4].copy_from_slice(&0x0100u16.to_le_bytes());
        iq2xxs[6..8].copy_from_slice(&0x0001u16.to_le_bytes());
        iq2xxs[8..10].copy_from_slice(&0x1000u16.to_le_bytes());
        let iq2xxs = dequantize_tensor_as_f32(&iq2xxs, GgufTensorType::IQ2_XXS, 256).unwrap();
        assert_eq!(iq2xxs[0], -1.5);
        assert_eq!(iq2xxs[1], 1.5);
        assert_eq!(iq2xxs[7], -1.5);
        assert_eq!(iq2xxs[8], 8.0625);
        assert_eq!(iq2xxs[9], 1.5);

        let mut iq2xs = vec![0u8; 74];
        iq2xs[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq2xs[2..4].copy_from_slice(&0x0200u16.to_le_bytes());
        iq2xs[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        iq2xs[66] = 0x31;
        let iq2xs = dequantize_tensor_as_f32(&iq2xs, GgufTensorType::IQ2_XS, 256).unwrap();
        assert_eq!(iq2xs[0], -1.5);
        assert_eq!(iq2xs[1], 1.5);
        assert_eq!(iq2xs[7], -1.5);
        assert_eq!(iq2xs[8], 8.0625);
        assert_eq!(iq2xs[9], 1.5);

        let mut iq3xxs = vec![0u8; 98];
        iq3xxs[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq3xxs[3] = 1;
        iq3xxs[66..70].copy_from_slice(&0x1000_0001u32.to_le_bytes());
        let iq3xxs = dequantize_tensor_as_f32(&iq3xxs, GgufTensorType::IQ3_XXS, 256).unwrap();
        assert_eq!(iq3xxs[0], -1.5);
        assert_eq!(iq3xxs[1], 1.5);
        assert_eq!(iq3xxs[3], 1.5);
        assert_eq!(iq3xxs[4], 7.5);
        assert_eq!(iq3xxs[7], -1.5);

        let mut iq1s = vec![0u8; 50];
        iq1s[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq1s[3] = 1;
        iq1s[34..36].copy_from_slice(&0x9000u16.to_le_bytes());
        let iq1s = dequantize_tensor_as_f32(&iq1s, GgufTensorType::IQ1_S, 256).unwrap();
        assert_eq!(iq1s[0], -1.6875);
        assert_eq!(iq1s[1], -1.6875);
        assert_eq!(iq1s[8], 1.3125);
        assert_eq!(iq1s[9], -1.6875);

        let mut iq2s = vec![0u8; 82];
        iq2s[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq2s[3] = 1;
        iq2s[34] = 0x81;
        iq2s[66] = 1;
        iq2s[74] = 0x10;
        let iq2s = dequantize_tensor_as_f32(&iq2s, GgufTensorType::IQ2_S, 256).unwrap();
        assert_eq!(iq2s[0], -4.6875);
        assert_eq!(iq2s[1], 4.6875);
        assert_eq!(iq2s[5], 1.5);
        assert_eq!(iq2s[7], -1.5);
        assert_eq!(iq2s[8], 8.0625);

        let mut iq1m = vec![0u8; 56];
        iq1m[1] = 1;
        iq1m[32] = 0x09;
        let scale_bits = f16_bits(0.5);
        let iq1m_sc = [
            ((scale_bits & 0x000f) << 12) | 0x0001,
            (scale_bits & 0x00f0) << 8,
            (scale_bits & 0x0f00) << 4,
            scale_bits & 0xf000,
        ];
        for (idx, scale) in iq1m_sc.iter().enumerate() {
            iq1m[48 + 2 * idx..50 + 2 * idx].copy_from_slice(&scale.to_le_bytes());
        }
        let iq1m = dequantize_tensor_as_f32(&iq1m, GgufTensorType::IQ1_M, 256).unwrap();
        assert_eq!(iq1m[0], -1.6875);
        assert_eq!(iq1m[1], -0.1875);
        assert_eq!(iq1m[2], 1.3125);
        assert_eq!(iq1m[7], -1.6875);
        assert_eq!(iq1m[8], 1.6875);

        let mut iq3s = vec![0u8; 110];
        iq3s[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq3s[3] = 1;
        iq3s[66] = 1;
        iq3s[74] = 0x81;
        iq3s[106] = 0x31;
        let iq3s = dequantize_tensor_as_f32(&iq3s, GgufTensorType::IQ3_S, 256).unwrap();
        assert_eq!(iq3s[0], -10.5);
        assert_eq!(iq3s[1], 7.5);
        assert_eq!(iq3s[2], 13.5);
        assert_eq!(iq3s[4], 4.5);
        assert_eq!(iq3s[7], -1.5);

        let mut iq4 = Vec::new();
        iq4.extend_from_slice(&f16_bits(0.5).to_le_bytes());
        iq4.extend([0xf0; 16]);
        let iq4_bytes = iq4.clone();
        let iq4 = dequantize_tensor_as_f32(&iq4_bytes, GgufTensorType::IQ4_NL, 32).unwrap();
        assert_eq!(iq4[0], -63.5);
        assert_eq!(iq4[16], 56.5);
        for dtype in [
            GgufTensorType::IQ4_NL_4_4,
            GgufTensorType::IQ4_NL_4_8,
            GgufTensorType::IQ4_NL_8_8,
        ] {
            assert_eq!(
                dequantize_tensor_as_f32(&iq4_bytes, dtype, 32).unwrap(),
                iq4
            );
        }

        let mut iq4xs = vec![0u8; 136];
        iq4xs[0..2].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        iq4xs[2..4].copy_from_slice(&0x0006u16.to_le_bytes());
        iq4xs[4] = 0xf1;
        iq4xs[8] = 0xf0;
        iq4xs[24] = 0x80;
        let iq4xs = dequantize_tensor_as_f32(&iq4xs, GgufTensorType::IQ4_XS, 256).unwrap();
        assert_eq!(iq4xs[0], -63.5);
        assert_eq!(iq4xs[16], 56.5);
        assert_eq!(iq4xs[32], 63.5);
        assert_eq!(iq4xs[48], -0.5);

        let mut tq1 = vec![0u8; 54];
        tq1[0] = 0;
        tq1[1] = 86;
        tq1[2] = 171;
        tq1[32] = 86;
        tq1[48] = 171;
        tq1[52..54].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        let tq1 = dequantize_tensor_as_f32(&tq1, GgufTensorType::TQ1_0, 256).unwrap();
        assert_eq!(tq1[0], -0.5);
        assert_eq!(tq1[1], 0.0);
        assert_eq!(tq1[2], 0.5);
        assert_eq!(tq1[160], 0.0);
        assert_eq!(tq1[240], 0.5);

        let mut tq2 = vec![0u8; 66];
        tq2[0] = 0b11_10_01_00;
        tq2[32] = 0b10_01_00_11;
        tq2[64..66].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        let tq2 = dequantize_tensor_as_f32(&tq2, GgufTensorType::TQ2_0, 256).unwrap();
        assert_eq!(tq2[0], -0.5);
        assert_eq!(tq2[32], 0.0);
        assert_eq!(tq2[64], 0.5);
        assert_eq!(tq2[96], 1.0);
        assert_eq!(tq2[128], 1.0);
        assert_eq!(tq2[160], -0.5);
        assert_eq!(tq2[192], 0.0);
        assert_eq!(tq2[224], 0.5);

        let mut q5 = Vec::new();
        q5.extend_from_slice(&f16_bits(1.0).to_le_bytes());
        q5.extend_from_slice(&0x0001_0000u32.to_le_bytes());
        q5.extend([0x8f; 16]);
        let q5 = dequantize_tensor_as_f32(&q5, GgufTensorType::Q5_0, 32).unwrap();
        assert_eq!(q5[0], -1.0);
        assert_eq!(q5[16], 8.0);

        let mut q5_1 = Vec::new();
        q5_1.extend_from_slice(&f16_bits(0.5).to_le_bytes());
        q5_1.extend_from_slice(&f16_bits(1.0).to_le_bytes());
        q5_1.extend_from_slice(&0x0001_0000u32.to_le_bytes());
        q5_1.extend([0x8f; 16]);
        let q5_1 = dequantize_tensor_as_f32(&q5_1, GgufTensorType::Q5_1, 32).unwrap();
        assert_eq!(q5_1[0], 8.5);
        assert_eq!(q5_1[16], 13.0);

        let mut q2k = vec![0u8; 84];
        q2k[0..16].fill(0x11);
        q2k[0] = 0x21;
        q2k[16] = 0xe4;
        q2k[48] = 0x03;
        q2k[80..82].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        q2k[82..84].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        let q2k = dequantize_tensor_as_f32(&q2k, GgufTensorType::Q2_K, 256).unwrap();
        assert_eq!(q2k[0], -1.0);
        assert_eq!(q2k[32], 0.5);
        assert_eq!(q2k[64], 1.5);
        assert_eq!(q2k[96], 2.5);
        assert_eq!(q2k[128], 2.5);

        let mut q3k = vec![0u8; 110];
        q3k[0] = 0b0001_0001;
        q3k[32] = 0b0000_0111;
        q3k[48] = 0b0000_0010;
        q3k[64] = 0b0000_0010;
        q3k[96..104].fill(0x11);
        q3k[104..108].fill(0xaa);
        q3k[108..110].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        let q3k = dequantize_tensor_as_f32(&q3k, GgufTensorType::Q3_K, 256).unwrap();
        assert_eq!(q3k[0], 3.0);
        assert_eq!(q3k[16], -2.0);
        assert_eq!(q3k[32], -3.0);
        assert_eq!(q3k[128], 2.0);

        let mut q4k = vec![0u8; 144];
        q4k[0..2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        q4k[2..4].copy_from_slice(&f16_bits(0.0).to_le_bytes());
        q4k[4..16].fill(1);
        q4k[16..144].fill(0x21);
        let q4k = dequantize_tensor_as_f32(&q4k, GgufTensorType::Q4_K, 256).unwrap();
        assert_eq!(q4k[0], 1.0);
        assert_eq!(q4k[32], 2.0);

        let mut q5k = vec![0u8; 176];
        q5k[0..2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        q5k[2..4].copy_from_slice(&f16_bits(0.5).to_le_bytes());
        q5k[4..16].fill(1);
        q5k[16] = 0b0000_0101;
        q5k[48..176].fill(0x21);
        let q5k = dequantize_tensor_as_f32(&q5k, GgufTensorType::Q5_K, 256).unwrap();
        assert_eq!(q5k[0], 16.5);
        assert_eq!(q5k[32], 1.5);
        assert_eq!(q5k[64], 16.5);

        let mut q6k = vec![0u8; 210];
        q6k[0] = 0xff;
        q6k[32] = 0xff;
        q6k[128] = 0xff;
        q6k[192..208].fill(1);
        q6k[208..210].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        let q6k = dequantize_tensor_as_f32(&q6k, GgufTensorType::Q6_K, 256).unwrap();
        assert_eq!(q6k[0], 31.0);
        assert_eq!(q6k[32], 31.0);
        assert_eq!(q6k[64], 31.0);
        assert_eq!(q6k[96], 31.0);
        assert_eq!(q6k[1], -32.0);

        let mut q8k = vec![0u8; 292];
        q8k[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        q8k[4] = 0;
        q8k[5] = 2;
        q8k[6] = 255;
        q8k[259] = 4;
        q8k[260..292].fill(0xff);
        let q8k = dequantize_tensor_as_f32(&q8k, GgufTensorType::Q8_K, 256).unwrap();
        assert_eq!(q8k[0], 0.0);
        assert_eq!(q8k[1], 1.0);
        assert_eq!(q8k[2], -0.5);
        assert_eq!(q8k[255], 2.0);
    }

    #[test]
    fn rejects_unaligned_tensor_offsets() {
        let path = tempfile_path("unaligned");
        write_tiny_qwen(&path, 1, 1);

        let err = GgufFile::open(&path).unwrap_err().to_string();
        assert!(err.contains("not aligned"));
    }

    fn write_tokenizer_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 5);

        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
        write_kv_string_array(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &[
                "h",
                "e",
                "l",
                "o",
                "!",
                "he",
                "hel",
                "hell",
                "hello",
                "\u{0120}world",
                "<|endoftext|>",
            ],
        );
        write_kv_string_array(
            &mut bytes,
            "tokenizer.ggml.merges",
            &["h e", "he l", "hel l", "hell o"],
        );
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 10);

        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    fn write_llama_tokenizer_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 8);

        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_string(&mut bytes, "tokenizer.ggml.model", "llama");
        write_kv_string_array(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &[
                "<unk>",
                "<s>",
                "</s>",
                "\u{2581}",
                "\u{2581}hello",
                "\u{2581}world",
                "hello",
                "<0x0A>",
                "!",
            ],
        );
        write_kv_f32_array(
            &mut bytes,
            "tokenizer.ggml.scores",
            &[-100.0, 0.0, 0.0, -5.0, -0.1, -0.1, -1.0, -10.0, -0.1],
        );
        write_kv_i32_array(
            &mut bytes,
            "tokenizer.ggml.token_type",
            &[2, 3, 3, 1, 1, 1, 1, 6, 1],
        );
        write_kv_u32(&mut bytes, "tokenizer.ggml.unknown_token_id", 0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.bos_token_id", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 2);

        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    fn write_full_qwen(path: &Path) {
        write_full_qwen_with_dtype(path, 1, 2, 1);
    }

    fn write_full_qwen_f32(path: &Path) {
        write_full_qwen_with_dtype(path, 0, 4, 0);
    }

    fn write_full_qwen_with_dtype(
        path: &Path,
        tensor_type: u32,
        element_size: usize,
        file_type: u32,
    ) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * element_size]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "full-qwen");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", file_type);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, tensor_type);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_moe_qwen(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
            ("blk.0.ffn_gate_exps.weight", vec![4, 3, 2]),
            ("blk.0.ffn_up_exps.weight", vec![4, 3, 2]),
            ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "qwen3moe");
        write_kv_string(&mut bytes, "general.name", "moe-qwen");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3moe.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3moe.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3moe.expert_feed_forward_length", 3);
        write_kv_u32(&mut bytes, "qwen3moe.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3moe.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3moe.expert_count", 2);
        write_kv_u32(&mut bytes, "qwen3moe.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_moe_qwen_packed_gate_up(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
            ("blk.0.ffn_gate_up_exps.weight", vec![4, 6, 2]),
            ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "qwen3moe");
        write_kv_string(&mut bytes, "general.name", "moe-qwen-packed-gate-up");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3moe.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3moe.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3moe.expert_feed_forward_length", 3);
        write_kv_u32(&mut bytes, "qwen3moe.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3moe.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3moe.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3moe.expert_count", 2);
        write_kv_u32(&mut bytes, "qwen3moe.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_custom_kv_lengths(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 2]),
            ("blk.0.attn_k.weight", vec![4, 2]),
            ("blk.0.attn_v.weight", vec![4, 3]),
            ("blk.0.attn_output.weight", vec![3, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "qwen-custom-kv");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.key_length", 2);
        write_kv_u32(&mut bytes, "qwen2.attention.value_length", 3);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_equal_custom_kv_lengths(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 2]),
            ("blk.0.attn_k.weight", vec![4, 2]),
            ("blk.0.attn_v.weight", vec![4, 2]),
            ("blk.0.attn_output.weight", vec![2, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "qwen-equal-custom-kv");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.key_length", 2);
        write_kv_u32(&mut bytes, "qwen2.attention.value_length", 2);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_dense(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "qwen-next-dense");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_attention_head_norm_aliases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.self_attention.q_layernorm.weight", vec![4]),
            ("blk.0.attention.k_norm.weight", vec![4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(
            &mut bytes,
            "general.name",
            "qwen-next-attention-head-norm-aliases",
        );
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_gated_attention_dense(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 8]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "qwen-next-gated-attention");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_ssm_metadata_dense(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "qwen-next-ssm-metadata-dense");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 16);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_ssm_metadata_packed(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.self_attn.W_pack.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.mlp.w1w3.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "qwen-next-ssm-metadata-packed");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 16);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_ssm_metadata_hf_packed_aliases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attention.qkv_proj.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.mlp.gate_up_proj.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(
            &mut bytes,
            "general.name",
            "qwen-next-ssm-metadata-hf-packed-aliases",
        );
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 16);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_recurrent_ssm(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.attn_post_norm.weight", vec![4]),
            ("blk.0.ssm_in.weight", vec![4, 8]),
            ("blk.0.ssm_conv1d.weight", vec![1, 6]),
            ("blk.0.ssm_dt.bias", vec![1]),
            ("blk.0.ssm_a", vec![1]),
            ("blk.0.ssm_ba.weight", vec![4, 2]),
            ("blk.0.ssm_norm.weight", vec![2]),
            ("blk.0.ssm_out.weight", vec![2, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 18);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "qwen-next-recurrent-ssm");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3next.ssm.conv_kernel", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.inner_size", 2);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 2);
        write_kv_u32(&mut bytes, "qwen3next.ssm.time_step_rank", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.group_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_recurrent_ssm_optimized(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.attn_post_norm.weight", vec![4]),
            ("blk.0.attn_qkv.weight", vec![4, 6]),
            ("blk.0.attn_gate.weight", vec![4, 2]),
            ("blk.0.ssm_conv1d.weight", vec![1, 6]),
            ("blk.0.ssm_dt.bias", vec![1]),
            ("blk.0.ssm_a", vec![1]),
            ("blk.0.ssm_ba.weight", vec![4, 2]),
            ("blk.0.ssm_norm.weight", vec![2]),
            ("blk.0.ssm_out.weight", vec![2, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 18);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(
            &mut bytes,
            "general.name",
            "qwen-next-recurrent-ssm-optimized",
        );
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen3next.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen3next.ssm.conv_kernel", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.inner_size", 2);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 2);
        write_kv_u32(&mut bytes, "qwen3next.ssm.time_step_rank", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.group_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_ssm_metadata_dense(path: &Path) {
        write_qwen_ssm_metadata_dense_with_sidecar(path, false);
    }

    fn write_qwen_ssm_sidecar_dense(path: &Path) {
        write_qwen_ssm_metadata_dense_with_sidecar(path, true);
    }

    fn write_qwen_ssm_metadata_dense_with_sidecar(path: &Path, include_sidecar: bool) {
        let mut tensor_specs = vec![
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        if include_sidecar {
            tensor_specs.push(("blk.0.ssm_in.weight", vec![4, 4]));
        }
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(
            &mut bytes,
            "general.name",
            if include_sidecar {
                "qwen-ssm-sidecar-dense"
            } else {
                "qwen-ssm-metadata-dense"
            },
        );
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen2.ssm.state_size", 16);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_ssm_metadata_packed(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.self_attn.W_pack.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.mlp.w1w3.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "qwen-ssm-metadata-packed");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen2.ssm.state_size", 16);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_ssm_metadata_hf_packed_aliases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attention.qkv_proj.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.mlp.gate_up_proj.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(
            &mut bytes,
            "general.name",
            "qwen-ssm-metadata-hf-packed-aliases",
        );
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "qwen2.ssm.state_size", 16);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_next_ssm_metadata(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 7);

        write_kv_string(&mut bytes, "general.architecture", "qwen3next");
        write_kv_string(&mut bytes, "general.name", "qwen-next-ssm-metadata");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "qwen3next.context_length", 16);
        write_kv_u32(&mut bytes, "qwen3next.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen3next.block_count", 1);
        write_kv_u32(&mut bytes, "qwen3next.ssm.state_size", 16);

        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_ssm_tensor(path: &Path) {
        let mut data = Vec::new();
        pad_to_alignment(&mut data, 32);
        let offset = data.len() as u64;
        data.extend(vec![0; 4 * 4 * 2]);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 1);
        write_u64(&mut bytes, 7);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "qwen-ssm");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);

        write_string(&mut bytes, "blk.0.ssm_in.weight");
        write_u32(&mut bytes, 2);
        write_u64(&mut bytes, 4);
        write_u64(&mut bytes, 4);
        write_u32(&mut bytes, 1);
        write_u64(&mut bytes, offset);

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_qwen_ssm_metadata(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 0);
        write_u64(&mut bytes, 7);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "qwen-ssm-metadata");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.ssm.state_size", 16);

        pad_to_alignment(&mut bytes, 32);
        fs::write(path, bytes).unwrap();
    }

    fn write_single_tensor_gguf(
        path: &Path,
        tensor_name: &str,
        dims: &[u64],
        tensor_type: u32,
        tensor_bytes: &[u8],
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 1);
        write_u64(&mut bytes, 1);

        write_kv_u32(&mut bytes, "general.alignment", 32);

        write_string(&mut bytes, tensor_name);
        write_u32(&mut bytes, dims.len() as u32);
        for dim in dims {
            write_u64(&mut bytes, *dim);
        }
        write_u32(&mut bytes, tensor_type);
        write_u64(&mut bytes, 0);

        pad_to_alignment(&mut bytes, 32);
        bytes.extend_from_slice(tensor_bytes);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_qwen(path: &Path, tensor_type: u32, tensor_offset: u64) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 1);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "qwen2");
        write_kv_string(&mut bytes, "general.name", "tiny-qwen");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "qwen2.context_length", 16);
        write_kv_u32(&mut bytes, "qwen2.embedding_length", 4);
        write_kv_u32(&mut bytes, "qwen2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "qwen2.block_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        write_kv_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        write_string(&mut bytes, "token_embd.weight");
        write_u32(&mut bytes, 2);
        write_u64(&mut bytes, 2);
        write_u64(&mut bytes, 4);
        write_u32(&mut bytes, tensor_type);
        write_u64(&mut bytes, tensor_offset);

        pad_to_alignment(&mut bytes, 32);
        if tensor_offset > 0 {
            bytes.extend(vec![0; tensor_offset as usize]);
        }
        bytes.extend_from_slice(&[0; 16]);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_mixtral(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
            ("blk.0.ffn_gate_exps.weight", vec![4, 3, 2]),
            ("blk.0.ffn_up_exps.weight", vec![4, 3, 2]),
            ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
        ];
        write_tiny_mixtral_with_tensors(path, "tiny-mixtral", &tensor_specs);
    }

    fn write_tiny_mixtral_per_expert(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
            ("blk.0.ffn_gate.0.weight", vec![4, 3]),
            ("blk.0.ffn_up.0.weight", vec![4, 3]),
            ("blk.0.ffn_down.0.weight", vec![3, 4]),
            ("blk.0.ffn_gate.1.weight", vec![4, 3]),
            ("blk.0.ffn_up.1.weight", vec![4, 3]),
            ("blk.0.ffn_down.1.weight", vec![3, 4]),
        ];
        write_tiny_mixtral_with_tensors(path, "tiny-mixtral-per-expert", &tensor_specs);
    }

    fn write_tiny_mixtral_alias_per_expert(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.gate.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
        ];
        write_tiny_mixtral_with_tensors(path, "tiny-mixtral-alias-per-expert", &tensor_specs);
    }

    fn write_tiny_mixtral_per_expert_packed_gate_up(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.gate.weight", vec![4, 2]),
            (
                "blk.0.block_sparse_moe.experts.0.gate_up_proj.weight",
                vec![4, 6],
            ),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            (
                "blk.0.block_sparse_moe.experts.1.gate_up_proj.weight",
                vec![4, 6],
            ),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
        ];
        write_tiny_mixtral_with_tensors(
            path,
            "tiny-mixtral-per-expert-packed-gate-up",
            &tensor_specs,
        );
    }

    fn write_tiny_mixtral_router_alias_per_expert(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.router.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
        ];
        write_tiny_mixtral_with_tensors(
            path,
            "tiny-mixtral-router-alias-per-expert",
            &tensor_specs,
        );
    }

    fn write_tiny_mixtral_output_router_bias_aliases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("lm_head.bias", vec![2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.router.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.router.bias", vec![2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
        ];
        write_tiny_mixtral_with_tensors(
            path,
            "tiny-mixtral-output-router-bias-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mixtral_moe_expert_bias_aliases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.router.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w1.bias", vec![3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.bias", vec![3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.0.w2.bias", vec![4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w1.bias", vec![3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.bias", vec![3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w2.bias", vec![4]),
        ];
        write_tiny_mixtral_with_tensors(
            path,
            "tiny-mixtral-moe-expert-bias-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mixtral_feed_forward_kind_first_experts(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            (
                "blk.0.feed_forward.block_sparse_moe.gate.weight",
                vec![4, 2],
            ),
            (
                "blk.0.feed_forward.block_sparse_moe.experts.w1.0.weight",
                vec![4, 3],
            ),
            (
                "blk.0.feed_forward.block_sparse_moe.experts.w3.0.weight",
                vec![4, 3],
            ),
            (
                "blk.0.feed_forward.block_sparse_moe.experts.w2.0.weight",
                vec![3, 4],
            ),
            (
                "blk.0.feed_forward.block_sparse_moe.experts.w1.1.weight",
                vec![4, 3],
            ),
            (
                "blk.0.feed_forward.block_sparse_moe.experts.w3.1.weight",
                vec![4, 3],
            ),
            (
                "blk.0.feed_forward.block_sparse_moe.experts.w2.1.weight",
                vec![3, 4],
            ),
        ];
        write_tiny_mixtral_with_tensors(
            path,
            "tiny-mixtral-feed-forward-kind-first-experts",
            &tensor_specs,
        );
    }

    fn write_tiny_mixtral_mlp_block_sparse_moe_aliases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.mlp.block_sparse_moe.gate.weight", vec![4, 2]),
            ("blk.0.mlp.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.mlp.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.mlp.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.mlp.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.mlp.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.mlp.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
        ];
        write_tiny_mixtral_with_tensors(
            path,
            "tiny-mixtral-mlp-block-sparse-moe-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mixtral_shared_experts_plural(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.gate.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
            (
                "blk.0.block_sparse_moe.shared_experts.w1.weight",
                vec![4, 3],
            ),
            (
                "blk.0.block_sparse_moe.shared_experts.w3.weight",
                vec![4, 3],
            ),
            (
                "blk.0.block_sparse_moe.shared_experts.w2.weight",
                vec![3, 4],
            ),
        ];
        write_tiny_mixtral_with_tensors(path, "tiny-mixtral-shared-experts-plural", &tensor_specs);
    }

    fn write_tiny_mixtral_packed_shared_expert(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.gate.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
            (
                "blk.0.block_sparse_moe.shared_experts.gate_up_proj.weight",
                vec![4, 6],
            ),
            (
                "blk.0.block_sparse_moe.shared_experts.w2.weight",
                vec![3, 4],
            ),
        ];
        write_tiny_mixtral_with_tensors(path, "tiny-mixtral-packed-shared-expert", &tensor_specs);
    }

    fn write_tiny_mixtral_shared_expert_gate(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.block_sparse_moe.gate.weight", vec![4, 2]),
            ("blk.0.block_sparse_moe.experts.0.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.0.w2.weight", vec![3, 4]),
            ("blk.0.block_sparse_moe.experts.1.w1.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w3.weight", vec![4, 3]),
            ("blk.0.block_sparse_moe.experts.1.w2.weight", vec![3, 4]),
            (
                "blk.0.block_sparse_moe.shared_experts.w1.weight",
                vec![4, 3],
            ),
            (
                "blk.0.block_sparse_moe.shared_experts.w3.weight",
                vec![4, 3],
            ),
            (
                "blk.0.block_sparse_moe.shared_experts.w2.weight",
                vec![3, 4],
            ),
            (
                "blk.0.block_sparse_moe.shared_expert_gate.weight",
                vec![4, 1],
            ),
            ("blk.0.block_sparse_moe.shared_expert_gate.bias", vec![1]),
        ];
        write_tiny_mixtral_with_tensors(path, "tiny-mixtral-shared-expert-gate", &tensor_specs);
    }

    fn write_tiny_mixtral_with_tensors(path: &Path, name: &str, tensor_specs: &[(&str, Vec<u64>)]) {
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (*name, dims.clone(), offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 14);

        write_kv_string(&mut bytes, "general.architecture", "mixtral");
        write_kv_string(&mut bytes, "general.name", name);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mixtral.context_length", 16);
        write_kv_u32(&mut bytes, "mixtral.embedding_length", 4);
        write_kv_u32(&mut bytes, "mixtral.feed_forward_length", 3);
        write_kv_u32(&mut bytes, "mixtral.block_count", 1);
        write_kv_u32(&mut bytes, "mixtral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mixtral.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "mixtral.expert_count", 2);
        write_kv_u32(&mut bytes, "mixtral.expert_used_count", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_deepseek_dense(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            12,
            |bytes| {
                write_deepseek_base_metadata(bytes, "deepseek", "tiny-deepseek", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_moe(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
                ("blk.0.ffn_gate_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_up_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
            ],
            14,
            |bytes| {
                write_deepseek_base_metadata(bytes, "deepseek", "tiny-deepseek-moe", Some(3));
                write_kv_u32(bytes, "deepseek.expert_count", 2);
                write_kv_u32(bytes, "deepseek.expert_used_count", 1);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_mla(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![("blk.0.attn_kv_a_mqa.weight", vec![4, 4])],
            12,
            |bytes| {
                write_deepseek_base_metadata(bytes, "deepseek2", "tiny-deepseek-mla", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_mla_metadata(path: &Path) {
        write_tiny_deepseek_gguf(path, vec![("token_embd.weight", vec![4, 2])], 13, |bytes| {
            write_deepseek_base_metadata(bytes, "deepseek2", "tiny-deepseek-mla", Some(8));
            write_kv_u32(bytes, "deepseek2.attention.q_lora_rank", 2);
            write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
            write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
        });
    }

    fn write_tiny_deepseek_mla_metadata_split(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            13,
            |bytes| {
                write_deepseek_base_metadata(
                    bytes,
                    "deepseek2",
                    "tiny-deepseek-mla-split",
                    Some(8),
                );
                write_kv_u32(bytes, "deepseek2.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_mla_metadata_packed(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.self_attn.W_pack.weight", vec![4, 12]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.mlp.w1w3.weight", vec![4, 16]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            13,
            |bytes| {
                write_deepseek_base_metadata(
                    bytes,
                    "deepseek2",
                    "tiny-deepseek-mla-packed",
                    Some(8),
                );
                write_kv_u32(bytes, "deepseek2.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_mla_sidecar_split(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.attn_kv_a_mqa.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            13,
            |bytes| {
                write_deepseek_base_metadata(
                    bytes,
                    "deepseek2",
                    "tiny-deepseek-mla-sidecar",
                    Some(8),
                );
                write_kv_u32(bytes, "deepseek2.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_true_mla_attention_aliases(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("transformer.embedding.word_embeddings.weight", vec![4, 2]),
                ("transformer.output_layer.weight", vec![4, 2]),
                ("transformer.encoder.final_layernorm.weight", vec![4]),
                (
                    "transformer.encoder.layers.0.input_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.post_attention_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.attention.q_a_proj.weight",
                    vec![4, 2],
                ),
                (
                    "transformer.encoder.layers.0.attention.q_a_layernorm.weight",
                    vec![2],
                ),
                (
                    "transformer.encoder.layers.0.attention.q_b_proj.weight",
                    vec![2, 4],
                ),
                (
                    "transformer.encoder.layers.0.attention.kv_a_proj_with_mqa.weight",
                    vec![4, 6],
                ),
                (
                    "transformer.encoder.layers.0.attention.kv_a_layernorm.weight",
                    vec![2],
                ),
                (
                    "transformer.encoder.layers.0.attention.kv_b_proj.weight",
                    vec![2, 4],
                ),
                (
                    "transformer.encoder.layers.0.attention.dense.weight",
                    vec![4, 4],
                ),
                (
                    "transformer.encoder.layers.0.mlp.gate_proj.weight",
                    vec![4, 8],
                ),
                (
                    "transformer.encoder.layers.0.mlp.up_proj.weight",
                    vec![4, 8],
                ),
                (
                    "transformer.encoder.layers.0.mlp.down_proj.weight",
                    vec![8, 4],
                ),
            ],
            17,
            |bytes| {
                write_deepseek_base_metadata(
                    bytes,
                    "deepseek2",
                    "tiny-deepseek-true-mla-attention-aliases",
                    Some(8),
                );
                write_kv_u32(bytes, "deepseek2.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "deepseek2.attention.kv_lora_rank", 2);
                write_kv_u32(bytes, "deepseek2.attention.qk_nope_head_dim", 0);
                write_kv_u32(bytes, "deepseek2.attention.qk_rope_head_dim", 4);
                write_kv_u32(bytes, "deepseek2.attention.v_head_dim", 4);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_deepseek_true_mla_self_attn_aliases(path: &Path) {
        write_tiny_deepseek_gguf(
            path,
            vec![
                ("model.embed_tokens.weight", vec![4, 2]),
                ("model.lm_head.weight", vec![4, 2]),
                ("model.norm.weight", vec![4]),
                ("model.layers.0.input_layernorm.weight", vec![4]),
                ("model.layers.0.post_attention_layernorm.weight", vec![4]),
                ("model.layers.0.self_attn.q_a_proj.weight", vec![4, 2]),
                ("model.layers.0.self_attn.q_a_norm.weight", vec![2]),
                ("model.layers.0.self_attn.q_b_proj.weight", vec![2, 4]),
                ("model.layers.0.self_attn.kv_a_proj.weight", vec![4, 6]),
                ("model.layers.0.self_attn.kv_a_norm.weight", vec![2]),
                ("model.layers.0.self_attn.kv_b_proj.weight", vec![2, 4]),
                ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
                ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
                ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
                ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
            ],
            17,
            |bytes| {
                write_deepseek_base_metadata(
                    bytes,
                    "deepseek2",
                    "tiny-deepseek-true-mla-self-attn-aliases",
                    Some(8),
                );
                write_kv_u32(bytes, "deepseek2.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "deepseek2.attention.kv_lora_rank", 2);
                write_kv_u32(bytes, "deepseek2.attention.qk_nope_head_dim", 0);
                write_kv_u32(bytes, "deepseek2.attention.qk_rope_head_dim", 4);
                write_kv_u32(bytes, "deepseek2.attention.v_head_dim", 4);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_dense(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            12,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_transformer_aliases(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("transformer.embedding.word_embeddings.weight", vec![4, 2]),
                ("transformer.output_layer.weight", vec![4, 2]),
                ("transformer.encoder.final_layernorm.weight", vec![4]),
                (
                    "transformer.encoder.layers.0.input_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.post_attention_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.query_key_value.weight",
                    vec![4, 12],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.query_key_value.bias",
                    vec![12],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.dense.weight",
                    vec![4, 4],
                ),
                (
                    "transformer.encoder.layers.0.mlp.dense_h_to_4h.weight",
                    vec![4, 16],
                ),
                (
                    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight",
                    vec![8, 4],
                ),
            ],
            12,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-transformer-aliases", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_gpt_neox_aliases(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("gpt_neox.embed_in.weight", vec![4, 2]),
                ("gpt_neox.embed_out.weight", vec![4, 2]),
                ("gpt_neox.final_layer_norm.weight", vec![4]),
                ("gpt_neox.layers.0.input_layernorm.weight", vec![4]),
                ("gpt_neox.layers.0.post_attention_layernorm.weight", vec![4]),
                (
                    "gpt_neox.layers.0.attention.query_key_value.weight",
                    vec![4, 12],
                ),
                ("gpt_neox.layers.0.attention.query_key_value.bias", vec![12]),
                ("gpt_neox.layers.0.attention.dense.weight", vec![4, 4]),
                ("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", vec![4, 16]),
                ("gpt_neox.layers.0.mlp.dense_4h_to_h.weight", vec![8, 4]),
            ],
            12,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-gpt-neox-aliases", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_model_transformer_w_aliases(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("model.transformer.wte.weight", vec![4, 2]),
                ("model.transformer.lm_head.weight", vec![4, 2]),
                ("model.transformer.final_layer_norm.weight", vec![4]),
                (
                    "model.transformer.layers.0.pre_attention_layernorm.weight",
                    vec![4],
                ),
                (
                    "model.transformer.layers.0.pre_feedforward_layernorm.weight",
                    vec![4],
                ),
                (
                    "model.transformer.layers.0.self_attention.Wq.weight",
                    vec![4, 4],
                ),
                (
                    "model.transformer.layers.0.self_attention.Wk.weight",
                    vec![4, 4],
                ),
                (
                    "model.transformer.layers.0.self_attention.Wv.weight",
                    vec![4, 4],
                ),
                (
                    "model.transformer.layers.0.self_attention.Wo.weight",
                    vec![4, 4],
                ),
                ("model.transformer.layers.0.mlp.w1.weight", vec![4, 8]),
                ("model.transformer.layers.0.mlp.w3.weight", vec![4, 8]),
                ("model.transformer.layers.0.mlp.w2.weight", vec![8, 4]),
            ],
            12,
            |bytes| {
                write_glm_base_metadata(
                    bytes,
                    "glm4",
                    "tiny-glm-model-transformer-w-aliases",
                    Some(8),
                );
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_mla_metadata_split(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            13,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-mla-split", Some(8));
                write_kv_u32(bytes, "glm4.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_mla_metadata_packed(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.self_attn.W_pack.weight", vec![4, 12]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.mlp.w1w3.weight", vec![4, 16]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            13,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-mla-packed", Some(8));
                write_kv_u32(bytes, "glm4.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_mla_sidecar_split(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.attn_kv_a_proj_with_mqa.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            13,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-mla-sidecar", Some(8));
                write_kv_u32(bytes, "glm4.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_true_mla_self_attention_aliases(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("transformer.embedding.word_embeddings.weight", vec![4, 2]),
                ("transformer.output_layer.weight", vec![4, 2]),
                ("transformer.encoder.final_layernorm.weight", vec![4]),
                (
                    "transformer.encoder.layers.0.input_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.post_attention_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.q_a_proj.weight",
                    vec![4, 2],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.q_a_layernorm.weight",
                    vec![2],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.q_b_proj.weight",
                    vec![2, 4],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.kv_a_proj_with_mqa.weight",
                    vec![4, 6],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.kv_a_layernorm.weight",
                    vec![2],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.kv_b_proj.weight",
                    vec![2, 4],
                ),
                (
                    "transformer.encoder.layers.0.self_attention.dense.weight",
                    vec![4, 4],
                ),
                (
                    "transformer.encoder.layers.0.mlp.gate_proj.weight",
                    vec![4, 8],
                ),
                (
                    "transformer.encoder.layers.0.mlp.up_proj.weight",
                    vec![4, 8],
                ),
                (
                    "transformer.encoder.layers.0.mlp.down_proj.weight",
                    vec![8, 4],
                ),
            ],
            17,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-true-mla-self-aliases", Some(8));
                write_kv_u32(bytes, "glm4.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "glm4.attention.kv_lora_rank", 2);
                write_kv_u32(bytes, "glm4.attention.qk_nope_head_dim", 0);
                write_kv_u32(bytes, "glm4.attention.qk_rope_head_dim", 4);
                write_kv_u32(bytes, "glm4.attention.v_head_dim", 4);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_true_mla_attention_aliases(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("transformer.embedding.word_embeddings.weight", vec![4, 2]),
                ("transformer.output_layer.weight", vec![4, 2]),
                ("transformer.encoder.final_layernorm.weight", vec![4]),
                (
                    "transformer.encoder.layers.0.input_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.post_attention_layernorm.weight",
                    vec![4],
                ),
                (
                    "transformer.encoder.layers.0.attention.q_a_proj.weight",
                    vec![4, 2],
                ),
                (
                    "transformer.encoder.layers.0.attention.q_a_layernorm.weight",
                    vec![2],
                ),
                (
                    "transformer.encoder.layers.0.attention.q_b_proj.weight",
                    vec![2, 4],
                ),
                (
                    "transformer.encoder.layers.0.attention.kv_a_proj_with_mqa.weight",
                    vec![4, 6],
                ),
                (
                    "transformer.encoder.layers.0.attention.kv_a_layernorm.weight",
                    vec![2],
                ),
                (
                    "transformer.encoder.layers.0.attention.kv_b_proj.weight",
                    vec![2, 4],
                ),
                (
                    "transformer.encoder.layers.0.attention.dense.weight",
                    vec![4, 4],
                ),
                (
                    "transformer.encoder.layers.0.mlp.gate_proj.weight",
                    vec![4, 8],
                ),
                (
                    "transformer.encoder.layers.0.mlp.up_proj.weight",
                    vec![4, 8],
                ),
                (
                    "transformer.encoder.layers.0.mlp.down_proj.weight",
                    vec![8, 4],
                ),
            ],
            17,
            |bytes| {
                write_glm_base_metadata(
                    bytes,
                    "glm4",
                    "tiny-glm-true-mla-attention-aliases",
                    Some(8),
                );
                write_kv_u32(bytes, "glm4.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "glm4.attention.kv_lora_rank", 2);
                write_kv_u32(bytes, "glm4.attention.qk_nope_head_dim", 0);
                write_kv_u32(bytes, "glm4.attention.qk_rope_head_dim", 4);
                write_kv_u32(bytes, "glm4.attention.v_head_dim", 4);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_true_mla_self_attn_aliases(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("model.embed_tokens.weight", vec![4, 2]),
                ("model.lm_head.weight", vec![4, 2]),
                ("model.norm.weight", vec![4]),
                ("model.layers.0.input_layernorm.weight", vec![4]),
                ("model.layers.0.post_attention_layernorm.weight", vec![4]),
                ("model.layers.0.self_attn.q_a_proj.weight", vec![4, 2]),
                ("model.layers.0.self_attn.q_a_layernorm.weight", vec![2]),
                ("model.layers.0.self_attn.q_b_proj.weight", vec![2, 4]),
                ("model.layers.0.self_attn.kv_a_proj.weight", vec![4, 6]),
                ("model.layers.0.self_attn.kv_a_layernorm.weight", vec![2]),
                ("model.layers.0.self_attn.kv_b_proj.weight", vec![2, 4]),
                ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
                ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
                ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
                ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
            ],
            17,
            |bytes| {
                write_glm_base_metadata(
                    bytes,
                    "glm4",
                    "tiny-glm-true-mla-self-attn-aliases",
                    Some(8),
                );
                write_kv_u32(bytes, "glm4.attention.q_lora_rank", 2);
                write_kv_u32(bytes, "glm4.attention.kv_lora_rank", 2);
                write_kv_u32(bytes, "glm4.attention.qk_nope_head_dim", 0);
                write_kv_u32(bytes, "glm4.attention.qk_rope_head_dim", 4);
                write_kv_u32(bytes, "glm4.attention.v_head_dim", 4);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_moe(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
                ("blk.0.ffn_gate_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_up_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
            ],
            14,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4moe", "tiny-glm-moe", Some(3));
                write_kv_u32(bytes, "glm4moe.expert_count", 2);
                write_kv_u32(bytes, "glm4moe.expert_used_count", 1);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_moe_grouped(path: &Path, groups: u32, used_groups: u32) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
                ("blk.0.ffn_gate_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_up_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
            ],
            16,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4moe", "tiny-glm-moe-grouped", Some(3));
                write_kv_u32(bytes, "glm4moe.expert_count", 2);
                write_kv_u32(bytes, "glm4moe.expert_used_count", 1);
                write_kv_u32(bytes, "glm4moe.expert_group_count", groups);
                write_kv_u32(bytes, "glm4moe.expert_group_used_count", used_groups);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_flash_dense(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate.weight", vec![4, 8]),
                ("blk.0.ffn_up.weight", vec![4, 8]),
                ("blk.0.ffn_down.weight", vec![8, 4]),
            ],
            12,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4flash", "tiny-glm-flash", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_flash_moe(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![
                ("token_embd.weight", vec![4, 2]),
                ("output_norm.weight", vec![4]),
                ("blk.0.attn_norm.weight", vec![4]),
                ("blk.0.ffn_norm.weight", vec![4]),
                ("blk.0.attn_q.weight", vec![4, 4]),
                ("blk.0.attn_k.weight", vec![4, 4]),
                ("blk.0.attn_v.weight", vec![4, 4]),
                ("blk.0.attn_output.weight", vec![4, 4]),
                ("blk.0.ffn_gate_inp.weight", vec![4, 2]),
                ("blk.0.ffn_gate_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_up_exps.weight", vec![4, 3, 2]),
                ("blk.0.ffn_down_exps.weight", vec![3, 4, 2]),
            ],
            14,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4flash", "tiny-glm-flash-moe", Some(3));
                write_kv_u32(bytes, "glm4flash.expert_count", 2);
                write_kv_u32(bytes, "glm4flash.expert_used_count", 1);
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_tiny_glm_mla(path: &Path) {
        write_tiny_glm_gguf(
            path,
            vec![("blk.0.attn_kv_a_proj_with_mqa.weight", vec![4, 4])],
            12,
            |bytes| {
                write_glm_base_metadata(bytes, "glm4", "tiny-glm-mla", Some(8));
                write_kv_u32(bytes, "tokenizer.ggml.eos_token_id", 1);
                write_kv_string_array(bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
            },
        );
    }

    fn write_deepseek_base_metadata(
        bytes: &mut Vec<u8>,
        architecture: &str,
        name: &str,
        feed_forward_length: Option<u32>,
    ) {
        write_kv_string(bytes, "general.architecture", architecture);
        write_kv_string(bytes, "general.name", name);
        write_kv_u32(bytes, "general.alignment", 32);
        write_kv_u32(bytes, "general.file_type", 1);
        write_kv_u32(bytes, &format!("{architecture}.context_length"), 16);
        write_kv_u32(bytes, &format!("{architecture}.embedding_length"), 4);
        if let Some(ff) = feed_forward_length {
            write_kv_u32(bytes, &format!("{architecture}.feed_forward_length"), ff);
        }
        write_kv_u32(bytes, &format!("{architecture}.block_count"), 1);
        write_kv_u32(bytes, &format!("{architecture}.attention.head_count"), 1);
        write_kv_u32(bytes, &format!("{architecture}.attention.head_count_kv"), 1);
    }

    fn write_glm_base_metadata(
        bytes: &mut Vec<u8>,
        architecture: &str,
        name: &str,
        feed_forward_length: Option<u32>,
    ) {
        write_kv_string(bytes, "general.architecture", architecture);
        write_kv_string(bytes, "general.name", name);
        write_kv_u32(bytes, "general.alignment", 32);
        write_kv_u32(bytes, "general.file_type", 1);
        write_kv_u32(bytes, &format!("{architecture}.context_length"), 16);
        write_kv_u32(bytes, &format!("{architecture}.embedding_length"), 4);
        if let Some(ff) = feed_forward_length {
            write_kv_u32(bytes, &format!("{architecture}.feed_forward_length"), ff);
        }
        write_kv_u32(bytes, &format!("{architecture}.block_count"), 1);
        write_kv_u32(bytes, &format!("{architecture}.attention.head_count"), 1);
        write_kv_u32(bytes, &format!("{architecture}.attention.head_count_kv"), 1);
    }

    fn write_tiny_deepseek_gguf(
        path: &Path,
        tensor_specs: Vec<(&'static str, Vec<u64>)>,
        metadata_count: u64,
        write_metadata: impl FnOnce(&mut Vec<u8>),
    ) {
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, metadata_count);
        write_metadata(&mut bytes);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_glm_gguf(
        path: &Path,
        tensor_specs: Vec<(&'static str, Vec<u64>)>,
        metadata_count: u64,
        write_metadata: impl FnOnce(&mut Vec<u8>),
    ) {
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, metadata_count);
        write_metadata(&mut bytes);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_llama(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, 1);
        write_u64(&mut bytes, 13);

        write_kv_string(&mut bytes, "general.architecture", "llama");
        write_kv_string(&mut bytes, "general.name", "tiny-llama");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "llama.context_length", 16);
        write_kv_u32(&mut bytes, "llama.embedding_length", 4);
        write_kv_u32(&mut bytes, "llama.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "llama.block_count", 1);
        write_kv_u32(&mut bytes, "llama.attention.head_count", 1);
        write_kv_u32(&mut bytes, "llama.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);
        write_kv_string(&mut bytes, "tokenizer.chat_template", LLAMA3_CHAT_TEMPLATE);

        write_string(&mut bytes, "token_embd.weight");
        write_u32(&mut bytes, 2);
        write_u64(&mut bytes, 2);
        write_u64(&mut bytes, 4);
        write_u32(&mut bytes, 1);
        write_u64(&mut bytes, 0);

        pad_to_alignment(&mut bytes, 32);
        bytes.extend_from_slice(&[0; 16]);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_mistral(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "mistral");
        write_kv_string(&mut bytes, "general.name", "tiny-mistral");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mistral.context_length", 16);
        write_kv_u32(&mut bytes, "mistral.embedding_length", 4);
        write_kv_u32(&mut bytes, "mistral.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "mistral.block_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_mistral_dense_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("norm.weight", vec![4]),
            ("blk.0.input_layernorm.weight", vec![4]),
            ("blk.0.post_attention_layernorm.weight", vec![4]),
            ("blk.0.self_attn.q_proj.weight", vec![4, 4]),
            ("blk.0.self_attn.k_proj.weight", vec![4, 4]),
            ("blk.0.self_attn.v_proj.weight", vec![4, 4]),
            ("blk.0.self_attn.o_proj.weight", vec![4, 4]),
            ("blk.0.mlp.gate_proj.weight", vec![4, 8]),
            ("blk.0.mlp.up_proj.weight", vec![4, 8]),
            ("blk.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "mistral");
        write_kv_string(&mut bytes, "general.name", "tiny-mistral-aliases");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mistral.context_length", 16);
        write_kv_u32(&mut bytes, "mistral.embedding_length", 4);
        write_kv_u32(&mut bytes, "mistral.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "mistral.block_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_mistral_model_layers_dense_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-model-layers-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_feed_forward_hf_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.feed_forward.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.feed_forward.up_proj.weight", vec![4, 8]),
            ("model.layers.0.feed_forward.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-feed-forward-hf-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_attn_container_split_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("blk.0.input_layernorm.weight", vec![4]),
            ("blk.0.post_attention_layernorm.weight", vec![4]),
            ("blk.0.attn.q_proj.weight", vec![4, 4]),
            ("blk.0.attn.k_proj.weight", vec![4, 4]),
            ("blk.0.attn.v_proj.weight", vec![4, 4]),
            ("blk.0.attn.o_proj.weight", vec![4, 4]),
            ("blk.0.mlp.gate_proj.weight", vec![4, 8]),
            ("blk.0.mlp.up_proj.weight", vec![4, 8]),
            ("blk.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-attn-container-split-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_ffn_container_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("blk.0.input_layernorm.weight", vec![4]),
            ("blk.0.post_attention_layernorm.weight", vec![4]),
            ("blk.0.self_attn.q_proj.weight", vec![4, 4]),
            ("blk.0.self_attn.k_proj.weight", vec![4, 4]),
            ("blk.0.self_attn.v_proj.weight", vec![4, 4]),
            ("blk.0.self_attn.o_proj.weight", vec![4, 4]),
            ("blk.0.ffn.gate_proj.weight", vec![4, 8]),
            ("blk.0.ffn.up_proj.weight", vec![4, 8]),
            ("blk.0.ffn.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-ffn-container-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_language_model_wrapper_aliases(path: &Path) {
        let tensor_specs = [
            ("language_model.model.embed_tokens.weight", vec![4, 2]),
            ("language_model.lm_head.weight", vec![4, 2]),
            ("language_model.model.norm.weight", vec![4]),
            (
                "language_model.model.layers.0.input_layernorm.weight",
                vec![4],
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight",
                vec![4],
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                vec![4, 4],
            ),
            (
                "language_model.model.layers.0.self_attn.k_proj.weight",
                vec![4, 4],
            ),
            (
                "language_model.model.layers.0.self_attn.v_proj.weight",
                vec![4, 4],
            ),
            (
                "language_model.model.layers.0.self_attn.o_proj.weight",
                vec![4, 4],
            ),
            (
                "language_model.model.layers.0.mlp.gate_proj.weight",
                vec![4, 8],
            ),
            (
                "language_model.model.layers.0.mlp.up_proj.weight",
                vec![4, 8],
            ),
            (
                "language_model.model.layers.0.mlp.down_proj.weight",
                vec![8, 4],
            ),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-language-model-wrapper-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_packed_aliases(path: &Path) {
        write_tiny_mistral_packed_aliases_with_names(
            path,
            "blk.0.self_attn.qkv_proj.weight",
            "blk.0.self_attn.qkv_proj.bias",
            "blk.0.mlp.gate_up_proj.weight",
        );
    }

    fn write_tiny_mistral_alternate_packed_aliases(path: &Path) {
        write_tiny_mistral_packed_aliases_with_names(
            path,
            "blk.0.self_attn.W_pack.weight",
            "blk.0.self_attn.W_pack.bias",
            "blk.0.mlp.w1w3.weight",
        );
    }

    fn write_tiny_mistral_attn_qkv_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("norm.weight", vec![4]),
            ("blk.0.input_layernorm.weight", vec![4]),
            ("blk.0.post_attention_layernorm.weight", vec![4]),
            ("blk.0.attn.qkv.weight", vec![4, 12]),
            ("blk.0.attn.qkv.bias", vec![12]),
            ("blk.0.attn.out_proj.weight", vec![4, 4]),
            ("blk.0.mlp.gate_up_proj.weight", vec![4, 16]),
            ("blk.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(path, "tiny-mistral-attn-qkv-aliases", &tensor_specs);
    }

    fn write_tiny_mistral_transformer_h_packed_aliases(path: &Path) {
        let tensor_specs = [
            ("transformer.wte.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("transformer.ln_f.weight", vec![4]),
            ("transformer.h.0.ln_1.weight", vec![4]),
            ("transformer.h.0.ln_2.weight", vec![4]),
            ("transformer.h.0.attn.c_attn.weight", vec![4, 12]),
            ("transformer.h.0.attn.c_attn.bias", vec![12]),
            ("transformer.h.0.attn.c_proj.weight", vec![4, 4]),
            ("transformer.h.0.mlp.fc1.weight", vec![4, 16]),
            ("transformer.h.0.mlp.fc2.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-transformer-h-packed-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_model_layers_packed_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.qkv_proj.weight", vec![4, 12]),
            ("model.layers.0.self_attn.qkv_proj.bias", vec![12]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_up_proj.weight", vec![4, 16]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_mistral_alias_tensor_specs(
            path,
            "tiny-mistral-model-layers-packed-aliases",
            &tensor_specs,
        );
    }

    fn write_tiny_mistral_packed_aliases_with_names(
        path: &Path,
        qkv_weight: &str,
        qkv_bias: &str,
        ffn_weight: &str,
    ) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("lm_head.weight", vec![4, 2]),
            ("norm.weight", vec![4]),
            ("blk.0.input_layernorm.weight", vec![4]),
            ("blk.0.post_attention_layernorm.weight", vec![4]),
            (qkv_weight, vec![4, 12]),
            (qkv_bias, vec![12]),
            ("blk.0.self_attn.o_proj.weight", vec![4, 4]),
            (ffn_weight, vec![4, 16]),
            ("blk.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "mistral");
        write_kv_string(&mut bytes, "general.name", "tiny-mistral-packed-aliases");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mistral.context_length", 16);
        write_kv_u32(&mut bytes, "mistral.embedding_length", 4);
        write_kv_u32(&mut bytes, "mistral.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "mistral.block_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_mistral_alias_tensor_specs(
        path: &Path,
        name: &str,
        tensor_specs: &[(&str, Vec<u64>)],
    ) {
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (*name, dims.clone(), offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "mistral");
        write_kv_string(&mut bytes, "general.name", name);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "mistral.context_length", 16);
        write_kv_u32(&mut bytes, "mistral.embedding_length", 4);
        write_kv_u32(&mut bytes, "mistral.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "mistral.block_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count", 1);
        write_kv_u32(&mut bytes, "mistral.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_gemma(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "gemma");
        write_kv_string(&mut bytes, "general.name", "tiny-gemma");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "gemma.context_length", 16);
        write_kv_u32(&mut bytes, "gemma.embedding_length", 4);
        write_kv_u32(&mut bytes, "gemma.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "gemma.block_count", 1);
        write_kv_u32(&mut bytes, "gemma.attention.head_count", 1);
        write_kv_u32(&mut bytes, "gemma.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_gemma_pre_feedforward_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.pre_feedforward_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
        ];
        write_tiny_gemma_with_tensors(path, "tiny-gemma-pre-feedforward-aliases", &tensor_specs);
    }

    fn write_tiny_gemma_post_feedforward_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_feedforward_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.out.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.proj.weight", vec![8, 4]),
        ];
        write_tiny_gemma_with_tensors(path, "tiny-gemma-post-feedforward-aliases", &tensor_specs);
    }

    fn write_tiny_gemma_dense_bias_aliases(path: &Path) {
        let tensor_specs = [
            ("model.embed_tokens.weight", vec![4, 2]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_feedforward_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.bias", vec![4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.gate_proj.bias", vec![8]),
            ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.up_proj.bias", vec![8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.bias", vec![4]),
        ];
        write_tiny_gemma_with_tensors(path, "tiny-gemma-dense-bias-aliases", &tensor_specs);
    }

    fn write_tiny_gemma_with_tensors(path: &Path, name: &str, tensor_specs: &[(&str, Vec<u64>)]) {
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (*name, dims.clone(), offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "gemma");
        write_kv_string(&mut bytes, "general.name", name);
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "gemma.context_length", 16);
        write_kv_u32(&mut bytes, "gemma.embedding_length", 4);
        write_kv_u32(&mut bytes, "gemma.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "gemma.block_count", 1);
        write_kv_u32(&mut bytes, "gemma.attention.head_count", 1);
        write_kv_u32(&mut bytes, "gemma.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_gemma2(path: &Path) {
        // Gemma-2 layout: post-attention + post-FFN norms and logit soft-capping.
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.post_attention_norm.weight", vec![4]),
            ("blk.0.post_ffw_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .into_iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (name, dims, offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 15);

        write_kv_string(&mut bytes, "general.architecture", "gemma2");
        write_kv_string(&mut bytes, "general.name", "tiny-gemma2");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "gemma2.context_length", 16);
        write_kv_u32(&mut bytes, "gemma2.embedding_length", 4);
        write_kv_u32(&mut bytes, "gemma2.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "gemma2.block_count", 1);
        write_kv_u32(&mut bytes, "gemma2.attention.head_count", 1);
        write_kv_u32(&mut bytes, "gemma2.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "gemma2.attention.sliding_window", 4096);
        write_kv_f32(&mut bytes, "gemma2.attn_logit_softcapping", 50.0);
        write_kv_f32(&mut bytes, "gemma2.final_logit_softcapping", 30.0);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    fn write_tiny_phi(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_split_qkv_biases(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 4]),
            ("blk.0.attn_v.weight", vec![4, 4]),
            ("blk.0.attn_q.bias", vec![4]),
            ("blk.0.attn_k.bias", vec![4]),
            ("blk.0.attn_v.bias", vec![4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_packed(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_qkv.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_packed_qkv_bias(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_qkv.weight", vec![4, 12]),
            ("blk.0.attn_qkv.bias", vec![12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_packed_ffn_gate_up(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_qkv.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate_up.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_packed_ffn_up_gate(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_qkv.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_up_gate.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_fused_ffn_up(path: &Path) {
        // Fused gate+up under `ffn_up` at 2x width (16 = 2 * ff where ff = 8),
        // no separate `ffn_gate` — the llama.cpp Phi-3 layout.
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_qkv.weight", vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_up.weight", vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_packed_aliases(
        path: &Path,
        qkv_weight: &str,
        qkv_bias: Option<&str>,
        ffn_weight: &str,
    ) {
        let mut tensor_specs = vec![
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            (qkv_weight, vec![4, 12]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            (ffn_weight, vec![4, 16]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        if let Some(qkv_bias) = qkv_bias {
            tensor_specs.insert(5, (qkv_bias, vec![12]));
        }
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_mixer_packed_qkv(path: &Path) {
        let tensor_specs = [
            ("token_embd.weight", vec![4, 2]),
            ("output_norm.weight", vec![4]),
            ("blk.0.input_layernorm.weight", vec![4]),
            ("blk.0.post_attention_layernorm.weight", vec![4]),
            ("blk.0.mixer.Wqkv.weight", vec![4, 12]),
            ("blk.0.mixer.Wqkv.bias", vec![12]),
            ("blk.0.mixer.out_proj.weight", vec![4, 4]),
            ("blk.0.mlp.fc1.weight", vec![4, 16]),
            ("blk.0.mlp.fc2.weight", vec![8, 4]),
        ];
        write_tiny_phi_with_tensors(path, &tensor_specs);
    }

    fn write_tiny_phi_with_tensors(path: &Path, tensor_specs: &[(&str, Vec<u64>)]) {
        let mut data = Vec::new();
        let tensor_specs = tensor_specs
            .iter()
            .map(|(name, dims)| {
                pad_to_alignment(&mut data, 32);
                let offset = data.len() as u64;
                let elements = dims.iter().product::<u64>();
                data.extend(vec![0; elements as usize * 2]);
                (*name, dims.clone(), offset)
            })
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        write_u32(&mut bytes, 3);
        write_u64(&mut bytes, tensor_specs.len() as u64);
        write_u64(&mut bytes, 12);

        write_kv_string(&mut bytes, "general.architecture", "phi3");
        write_kv_string(&mut bytes, "general.name", "tiny-phi");
        write_kv_u32(&mut bytes, "general.alignment", 32);
        write_kv_u32(&mut bytes, "general.file_type", 1);
        write_kv_u32(&mut bytes, "phi3.context_length", 16);
        write_kv_u32(&mut bytes, "phi3.embedding_length", 4);
        write_kv_u32(&mut bytes, "phi3.feed_forward_length", 8);
        write_kv_u32(&mut bytes, "phi3.block_count", 1);
        write_kv_u32(&mut bytes, "phi3.attention.head_count", 1);
        write_kv_u32(&mut bytes, "phi3.attention.head_count_kv", 1);
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 1);
        write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &["hello", "world"]);

        for (name, dims, offset) in tensor_specs {
            write_string(&mut bytes, name);
            write_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                write_u64(&mut bytes, dim);
            }
            write_u32(&mut bytes, 1);
            write_u64(&mut bytes, offset);
        }

        pad_to_alignment(&mut bytes, 32);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn opens_split_gguf_across_shards() {
        fn shard_bytes(
            split_no: u32,
            tensors: &[(&str, &[f32], u64)],
            with_model_metadata: bool,
        ) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            write_u32(&mut bytes, 3);
            write_u64(&mut bytes, tensors.len() as u64);
            write_u64(&mut bytes, if with_model_metadata { 4 } else { 3 });
            write_kv_u32(&mut bytes, "general.alignment", 32);
            write_kv_u32(&mut bytes, "split.count", 2);
            write_kv_u32(&mut bytes, "split.no", split_no);
            if with_model_metadata {
                write_kv_u32(&mut bytes, "split.tensors.count", 3);
            }
            for (name, values, offset) in tensors {
                write_string(&mut bytes, name);
                write_u32(&mut bytes, 1);
                write_u64(&mut bytes, values.len() as u64);
                write_u32(&mut bytes, 0); // f32
                write_u64(&mut bytes, *offset);
            }
            while bytes.len() % 32 != 0 {
                bytes.push(0);
            }
            let data_start = bytes.len();
            for (_, values, offset) in tensors {
                let target = data_start + *offset as usize;
                assert!(bytes.len() <= target);
                bytes.resize(target, 0);
                for value in *values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            bytes
        }

        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = format!("hi-gguf-split-test-{nanos}");
        let shard1 = dir.join(format!("{base}-00001-of-00002.gguf"));
        let shard2 = dir.join(format!("{base}-00002-of-00002.gguf"));
        std::fs::write(&shard1, shard_bytes(0, &[("a", &[1.0, 2.0], 0)], true)).unwrap();
        std::fs::write(
            &shard2,
            shard_bytes(1, &[("b", &[3.0, 4.0], 0), ("c", &[5.0, 6.0], 32)], false),
        )
        .unwrap();

        let gguf = GgufFile::open(&shard1).unwrap();
        assert_eq!(gguf.tensors().len(), 3);
        assert_eq!(gguf.tensor_info("a").unwrap().shard, 0);
        assert_eq!(gguf.tensor_info("b").unwrap().shard, 1);
        assert_eq!(gguf.tensor_info("c").unwrap().shard, 1);
        let c = gguf.tensor("c").unwrap();
        let values: Vec<f32> = c
            .bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![5.0, 6.0]);
        let a = gguf.tensor("a").unwrap();
        assert_eq!(a.bytes.len(), 8);

        // Opening a later shard directly is refused with a pointer at shard 1.
        let err = GgufFile::open(&shard2).unwrap_err().to_string();
        assert!(err.contains("open the first shard"), "{err}");

        std::fs::remove_file(&shard1).unwrap();
        std::fs::remove_file(&shard2).unwrap();
    }

    /// Two-shard split GGUF with deterministic non-zero f32 payloads, for the
    /// tensor-file-range / O_DIRECT / madvise tests (zero-filled fixtures
    /// cannot catch offset arithmetic bugs). Tensor `big` spans multiple
    /// 4096-byte direct-I/O blocks; `b` sits at a non-zero data offset.
    fn write_pattern_split_gguf(dir: &Path, base: &str) -> (PathBuf, PathBuf) {
        fn shard_bytes(split_no: u32, tensors: &[(&str, &[f32], u64)], with_meta: bool) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            write_u32(&mut bytes, 3);
            write_u64(&mut bytes, tensors.len() as u64);
            write_u64(&mut bytes, if with_meta { 4 } else { 3 });
            write_kv_u32(&mut bytes, "general.alignment", 32);
            write_kv_u32(&mut bytes, "split.count", 2);
            write_kv_u32(&mut bytes, "split.no", split_no);
            if with_meta {
                write_kv_u32(&mut bytes, "split.tensors.count", 3);
            }
            for (name, values, offset) in tensors {
                write_string(&mut bytes, name);
                write_u32(&mut bytes, 1);
                write_u64(&mut bytes, values.len() as u64);
                write_u32(&mut bytes, 0); // f32
                write_u64(&mut bytes, *offset);
            }
            pad_to_alignment(&mut bytes, 32);
            let data_start = bytes.len();
            for (_, values, offset) in tensors {
                let target = data_start + *offset as usize;
                assert!(bytes.len() <= target);
                bytes.resize(target, 0);
                for value in *values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            bytes
        }

        let pattern = |len: usize, seed: f32| -> Vec<f32> {
            (0..len).map(|i| seed + i as f32 * 0.25).collect()
        };
        let shard1 = dir.join(format!("{base}-00001-of-00002.gguf"));
        let shard2 = dir.join(format!("{base}-00002-of-00002.gguf"));
        let big = pattern(3000, 1.0); // 12000 bytes: crosses two block boundaries
        let b = pattern(64, 500.0);
        let c = pattern(128, 900.0);
        std::fs::write(&shard1, shard_bytes(0, &[("big", &big, 0)], true)).unwrap();
        std::fs::write(
            &shard2,
            shard_bytes(1, &[("b", &b, 0), ("c", &c, 256)], false),
        )
        .unwrap();
        (shard1, shard2)
    }

    #[test]
    fn tensor_file_range_locates_bytes_across_shards() {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let (shard1, shard2) = write_pattern_split_gguf(&dir, &format!("hi-gguf-range-{nanos}"));

        let gguf = GgufFile::open(&shard1).unwrap();
        assert_eq!(gguf.shard_count(), 2);
        assert_eq!(gguf.shard_path(0), Some(shard1.as_path()));
        assert_eq!(gguf.shard_path(1), Some(shard2.as_path()));
        for name in ["big", "b", "c"] {
            let range = gguf.tensor_file_range(name).unwrap();
            let raw = std::fs::read(gguf.shard_path(range.shard).unwrap()).unwrap();
            let start = usize::try_from(range.file_offset).unwrap();
            let len = usize::try_from(range.len).unwrap();
            let view = gguf.tensor(name).unwrap();
            assert_eq!(len, view.bytes.len());
            assert_eq!(&raw[start..start + len], view.bytes, "tensor {name}");
        }
        assert!(gguf.tensor_file_range("missing").is_err());

        std::fs::remove_file(&shard1).unwrap();
        std::fs::remove_file(&shard2).unwrap();
    }

    #[test]
    fn advise_tensor_range_smoke() {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let (shard1, shard2) = write_pattern_split_gguf(&dir, &format!("hi-gguf-advise-{nanos}"));

        let gguf = GgufFile::open(&shard1).unwrap();
        gguf.advise_tensor("big", GgufMemoryAdvice::Random).unwrap();
        // Cross-shard, unaligned sub-range (widened to page boundaries).
        gguf.advise_tensor_range("c", 3, 100, GgufMemoryAdvice::WillNeed)
            .unwrap();
        gguf.advise_tensor("b", GgufMemoryAdvice::Sequential)
            .unwrap();
        gguf.advise_tensor("b", GgufMemoryAdvice::Normal).unwrap();
        // Advice must stay within the tensor's extent.
        let err = gguf
            .advise_tensor_range("b", 200, 100, GgufMemoryAdvice::WillNeed)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds tensor"), "{err}");

        std::fs::remove_file(&shard1).unwrap();
        std::fs::remove_file(&shard2).unwrap();
    }

    /// O_DIRECT twin-fd reads must return byte-identical data to the mmap for
    /// arbitrary (unaligned) sub-ranges. Skips when the filesystem hosting the
    /// fixture does not support O_DIRECT (e.g. tmpfs) or on non-Linux.
    #[test]
    fn o_direct_reads_match_mmap() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        let (shard1, shard2) = write_pattern_split_gguf(&dir, &format!("hi-gguf-odirect-{nanos}"));

        let gguf = GgufFile::open(&shard1).unwrap();
        let reader = match gguf.direct_io_reader() {
            Ok(reader) => reader,
            Err(err) => {
                eprintln!("skipping O_DIRECT equivalence test: {err:#}");
                std::fs::remove_file(&shard1).unwrap();
                std::fs::remove_file(&shard2).unwrap();
                return;
            }
        };
        for name in ["big", "b", "c"] {
            let range = gguf.tensor_file_range(name).unwrap();
            let view = gguf.tensor(name).unwrap();
            let len = usize::try_from(range.len).unwrap();
            // Full tensor.
            let full = reader
                .read_range(range.shard, range.file_offset, len)
                .unwrap();
            assert_eq!(full, view.bytes, "tensor {name} full read");
            // Unaligned sub-ranges, including block-boundary crossings for `big`.
            for (sub_off, sub_len) in [(0usize, 1usize), (3, 5), (7, len - 7), (len - 1, 1)] {
                let got = reader
                    .read_range(range.shard, range.file_offset + sub_off as u64, sub_len)
                    .unwrap();
                assert_eq!(
                    got,
                    &view.bytes[sub_off..sub_off + sub_len],
                    "tensor {name} sub-range {sub_off}+{sub_len}"
                );
            }
        }
        // Reading past EOF of the shard must fail, not fabricate bytes.
        let big = gguf.tensor_file_range("big").unwrap();
        let file_len = std::fs::metadata(&shard1).unwrap().len();
        assert!(reader.read_range(big.shard, file_len - 4, 64).is_err());

        std::fs::remove_file(&shard1).unwrap();
        std::fs::remove_file(&shard2).unwrap();
    }

    fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        write_string(bytes, key);
        write_u32(bytes, 8);
        write_string(bytes, value);
    }

    fn write_kv_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_string(bytes, key);
        write_u32(bytes, 9);
        write_u32(bytes, 8);
        write_u64(bytes, values.len() as u64);
        for value in values {
            write_string(bytes, value);
        }
    }

    fn write_kv_f32_array(bytes: &mut Vec<u8>, key: &str, values: &[f32]) {
        write_string(bytes, key);
        write_u32(bytes, 9);
        write_u32(bytes, 6);
        write_u64(bytes, values.len() as u64);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn write_kv_i32_array(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
        write_string(bytes, key);
        write_u32(bytes, 9);
        write_u32(bytes, 5);
        write_u64(bytes, values.len() as u64);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn write_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        write_string(bytes, key);
        write_u32(bytes, 4);
        write_u32(bytes, value);
    }

    fn write_kv_f32(bytes: &mut Vec<u8>, key: &str, value: f32) {
        write_string(bytes, key);
        write_u32(bytes, 6);
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        write_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f16_bits(value: f32) -> u16 {
        match value {
            0.0 => 0x0000,
            0.5 => 0x3800,
            1.0 => 0x3c00,
            _ => panic!("test fixture only supports simple f16 values, got {value}"),
        }
    }

    fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) {
        let remainder = bytes.len() % alignment;
        if remainder != 0 {
            bytes.extend(vec![0; alignment - remainder]);
        }
    }

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-gguf-{name}-{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }
}
