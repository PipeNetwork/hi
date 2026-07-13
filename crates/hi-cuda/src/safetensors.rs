//! Minimal memory-mapped safetensors reader + dequantization toolkit for the
//! DeepSeek-V4-Flash speculative-decoding weight artifacts (Stage B/C of
//! `docs/deepseek-v4-spec-decode-plan.md`): the official MTP module shard
//! (`mtp/model-00046-of-00046.safetensors`) and the RedHat DFlash drafter
//! (`dflash-redhat/model.safetensors`).
//!
//! Format: an 8-byte little-endian u64 header length, then that many bytes of
//! JSON mapping tensor name -> `{dtype, shape, data_offsets: [begin, end]}`
//! (offsets relative to the first byte after the header) plus an optional
//! `"__metadata__"` string map. Tensor data is raw row-major little-endian.
//!
//! Dequant conventions verified byte-level against the real artifacts
//! (2026-07-12):
//! - FP8 weights are OCP E4M3FN (`F8_E4M3`) paired with a sibling `.scale`
//!   tensor of dtype `F8_E8M0` (ue8m0, decoded as `2^(byte - 127)`) holding
//!   one **multiplicative** scale per 128x128 weight block (the model config's
//!   `quantization_config: {fmt: e4m3, scale_fmt: ue8m0, weight_block_size:
//!   [128, 128]}`). The block grid is inferred from the weight/scale shapes,
//!   so any uniform grid (with partial edge blocks) is supported.
//! - fp4 experts are E2M1 codes packed two per byte along the innermost dim
//!   (low nibble = even element, high nibble = odd element; dtype `I8`,
//!   packed shape `[rows, cols/2]`) with a ue8m0 `.scale` tensor of shape
//!   `[rows, cols/32]` — i.e. MXFP4 with 32-element groups, the exact value
//!   grid of GGUF's `MXFP4` type. [`repack_fp4_to_gguf_mxfp4`] converts the
//!   packing losslessly (nibble reorder + verbatim scale byte) into the
//!   17-bytes-per-32-values GGUF block stream hi-gguf decodes and the expert
//!   pool consumes: per block one e8m0 scale byte, then 16 nibble bytes where
//!   byte `j` holds element `j` (low nibble) and element `j + 16` (high).
//!   GGUF's halved e8m0 decode times its doubled value table equals our
//!   `2^(b-127) * e2m1`, so the raw scale byte carries over unchanged.
//! - BF16 is converted by widening to f32 (exact) and, for f16 targets,
//!   rounding to nearest-even via [`f32_to_f16_bits`].

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

/// Tensor element type as spelled in a safetensors header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafetensorsDtype {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    I64,
    U64,
    I32,
    U32,
    I16,
    U16,
    I8,
    U8,
    Bool,
}

impl SafetensorsDtype {
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" => Self::F8E8M0,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            other => bail!("unsupported safetensors dtype {other:?}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
        }
    }

    /// Bytes per stored element (fp4 experts are pre-packed two values per
    /// `I8` byte upstream, so the *stored* width is still 1).
    pub fn byte_width(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

/// One header entry: name, dtype, shape, and byte span in the data section.
#[derive(Clone, Debug)]
pub struct SafetensorsTensor {
    pub name: String,
    pub dtype: SafetensorsDtype,
    pub shape: Vec<usize>,
    /// Byte offset of the first data byte, relative to the data section
    /// (the byte immediately after the JSON header).
    pub begin: usize,
    /// One past the last data byte, relative to the data section.
    pub end: usize,
}

impl SafetensorsTensor {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.end - self.begin
    }
}

/// A memory-mapped safetensors file. Opening parses and validates the header
/// (dtype/shape/offset consistency and gap-free coverage of the data
/// section), so tensor byte access afterwards is infallible slicing.
#[derive(Debug)]
pub struct SafetensorsFile {
    path: PathBuf,
    mmap: Mmap,
    data_start: usize,
    metadata: Option<BTreeMap<String, String>>,
    /// Sorted by tensor name.
    tensors: Vec<SafetensorsTensor>,
    index: HashMap<String, usize>,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory-mapping {}", path.display()))?;
        Self::parse(path.to_path_buf(), mmap)
            .with_context(|| format!("parsing safetensors file {}", path.display()))
    }

    fn parse(path: PathBuf, mmap: Mmap) -> Result<Self> {
        if mmap.len() < 8 {
            bail!(
                "file is {} bytes; need at least an 8-byte header length",
                mmap.len()
            );
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().expect("slice length is 8"));
        let header_len = usize::try_from(header_len).context("header length does not fit usize")?;
        let data_start = header_len
            .checked_add(8)
            .ok_or_else(|| anyhow!("header length {header_len} overflows"))?;
        if data_start > mmap.len() {
            bail!(
                "header claims {header_len} bytes but only {} remain (truncated download?)",
                mmap.len() - 8
            );
        }
        let data_len = mmap.len() - data_start;
        let header: serde_json::Value =
            serde_json::from_slice(&mmap[8..data_start]).context("header is not valid JSON")?;
        let entries = header
            .as_object()
            .ok_or_else(|| anyhow!("header JSON is not an object"))?;

        let mut metadata = None;
        let mut tensors = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            if name == "__metadata__" {
                let map = value
                    .as_object()
                    .ok_or_else(|| anyhow!("__metadata__ is not an object"))?;
                let mut parsed = BTreeMap::new();
                for (key, val) in map {
                    let val = val
                        .as_str()
                        .ok_or_else(|| anyhow!("__metadata__[{key:?}] is not a string"))?;
                    parsed.insert(key.clone(), val.to_string());
                }
                metadata = Some(parsed);
                continue;
            }
            tensors.push(parse_tensor_entry(name, value, data_len)?);
        }

        validate_coverage(&tensors, data_len)?;

        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let index = tensors
            .iter()
            .enumerate()
            .map(|(idx, tensor)| (tensor.name.clone(), idx))
            .collect();
        Ok(Self {
            path,
            mmap,
            data_start,
            metadata,
            tensors,
            index,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The optional `"__metadata__"` string map.
    pub fn metadata(&self) -> Option<&BTreeMap<String, String>> {
        self.metadata.as_ref()
    }

    /// All tensors, sorted by name.
    pub fn tensors(&self) -> &[SafetensorsTensor] {
        &self.tensors
    }

    /// Tensor names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    pub fn info(&self, name: &str) -> Option<&SafetensorsTensor> {
        self.tensors.get(*self.index.get(name)?)
    }

    fn require(&self, name: &str) -> Result<&SafetensorsTensor> {
        self.info(name)
            .ok_or_else(|| anyhow!("tensor {name:?} not found in {}", self.path.display()))
    }

    /// Zero-copy view of a tensor's raw little-endian bytes.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.require(name)?;
        Ok(&self.mmap[self.data_start + info.begin..self.data_start + info.end])
    }

    /// Convert an F32 / F16 / BF16 tensor to f32 values. Block-scaled FP8 and
    /// fp4 tensors go through [`Self::fp8_block_scaled_f32`] /
    /// [`Self::fp4_block_scaled_f32`] instead (they need their scale sibling).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.require(name)?;
        let bytes = self.bytes(name)?;
        match info.dtype {
            SafetensorsDtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
                .collect()),
            SafetensorsDtype::F16 => Ok(le_u16s(bytes).map(f16_to_f32).collect()),
            SafetensorsDtype::BF16 => Ok(le_u16s(bytes).map(bf16_to_f32).collect()),
            other => bail!(
                "tensor {name:?} has dtype {}; tensor_f32 handles F32/F16/BF16",
                other.name()
            ),
        }
    }

    /// [`Self::tensor_f32`] rounded to IEEE f16 bits (round-to-nearest-even).
    pub fn tensor_f16(&self, name: &str) -> Result<Vec<u16>> {
        Ok(self
            .tensor_f32(name)?
            .into_iter()
            .map(f32_to_f16_bits)
            .collect())
    }

    /// Read an I64 tensor (e.g. the DFlash `d2t` draft-to-target vocab map).
    pub fn tensor_i64(&self, name: &str) -> Result<Vec<i64>> {
        let info = self.require(name)?;
        if info.dtype != SafetensorsDtype::I64 {
            bail!(
                "tensor {name:?} has dtype {}, expected I64",
                info.dtype.name()
            );
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunk length is 8")))
            .collect())
    }

    /// Dequantize a 2-D `F8_E4M3` weight with its ue8m0 block-scale sibling
    /// (`x.weight` + `x.scale`, e.g. DeepSeek-V4 128x128 blocks) to row-major
    /// f32. The block grid is inferred from the two shapes.
    pub fn fp8_block_scaled_f32(&self, weight_name: &str) -> Result<Vec<f32>> {
        let (weight, scale) = self.weight_and_scale(weight_name, SafetensorsDtype::F8E4M3)?;
        dequant_fp8_block_scaled(
            self.bytes(&weight.name)?,
            self.bytes(&scale.name)?,
            weight.shape[0],
            weight.shape[1],
            scale.shape[0],
            scale.shape[1],
        )
        .with_context(|| format!("dequantizing {weight_name:?}"))
    }

    /// [`Self::fp8_block_scaled_f32`] rounded to IEEE f16 bits.
    pub fn fp8_block_scaled_f16(&self, weight_name: &str) -> Result<Vec<u16>> {
        Ok(self
            .fp8_block_scaled_f32(weight_name)?
            .into_iter()
            .map(f32_to_f16_bits)
            .collect())
    }

    /// Dequantize a packed-fp4 expert weight (`I8` `[rows, cols/2]` + ue8m0
    /// `.scale` `[rows, cols/32]`) to row-major f32 — the CPU-reference view
    /// of what [`Self::fp4_to_gguf_mxfp4`] repacks.
    pub fn fp4_block_scaled_f32(&self, weight_name: &str) -> Result<Vec<f32>> {
        let (weight, scale, rows, logical_cols) = self.fp4_weight_and_scale(weight_name)?;
        dequant_fp4_groups(
            self.bytes(&weight.name)?,
            self.bytes(&scale.name)?,
            rows,
            logical_cols,
        )
        .with_context(|| format!("dequantizing {weight_name:?}"))
    }

    /// Losslessly repack a packed-fp4 expert weight into the GGUF `MXFP4`
    /// block stream (17 bytes per 32 values) that `hi_gguf` decodes and the
    /// DsV4 expert pool consumes. Concatenating the per-expert results in
    /// expert order reproduces the packed rank-3 GGUF expert tensor layout
    /// (innermost = the logical column/`in` dim in both formats).
    pub fn fp4_to_gguf_mxfp4(&self, weight_name: &str) -> Result<Vec<u8>> {
        let (weight, scale, rows, logical_cols) = self.fp4_weight_and_scale(weight_name)?;
        repack_fp4_to_gguf_mxfp4(
            self.bytes(&weight.name)?,
            self.bytes(&scale.name)?,
            rows,
            logical_cols,
        )
        .with_context(|| format!("repacking {weight_name:?}"))
    }

    /// Resolve `x.weight` + its `x.scale` sibling, checking dtypes and ranks.
    fn weight_and_scale(
        &self,
        weight_name: &str,
        dtype: SafetensorsDtype,
    ) -> Result<(&SafetensorsTensor, &SafetensorsTensor)> {
        let weight = self.require(weight_name)?;
        if weight.dtype != dtype {
            bail!(
                "tensor {weight_name:?} has dtype {}, expected {}",
                weight.dtype.name(),
                dtype.name()
            );
        }
        if weight.shape.len() != 2 {
            bail!(
                "tensor {weight_name:?} has shape {:?}, expected rank 2",
                weight.shape
            );
        }
        let scale_name = weight_name
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.scale"))
            .ok_or_else(|| {
                anyhow!("cannot derive scale name for {weight_name:?} (no .weight suffix)")
            })?;
        let scale = self.require(&scale_name)?;
        if scale.dtype != SafetensorsDtype::F8E8M0 {
            bail!(
                "scale tensor {scale_name:?} has dtype {}, expected F8_E8M0",
                scale.dtype.name()
            );
        }
        if scale.shape.len() != 2 {
            bail!(
                "scale tensor {scale_name:?} has shape {:?}, expected rank 2",
                scale.shape
            );
        }
        Ok((weight, scale))
    }

    /// [`Self::weight_and_scale`] for packed fp4, validating the packed and
    /// scale shapes against each other; returns `(weight, scale, rows,
    /// logical_cols)` where `logical_cols` is the unpacked column count.
    fn fp4_weight_and_scale(
        &self,
        weight_name: &str,
    ) -> Result<(&SafetensorsTensor, &SafetensorsTensor, usize, usize)> {
        let (weight, scale) = self.weight_and_scale(weight_name, SafetensorsDtype::I8)?;
        let rows = weight.shape[0];
        let logical_cols = weight.shape[1] * 2;
        if logical_cols % MXFP4_BLOCK_ELEMENTS != 0 {
            bail!(
                "fp4 tensor {weight_name:?} has {logical_cols} logical columns, not a multiple of {MXFP4_BLOCK_ELEMENTS}"
            );
        }
        let expected = [rows, logical_cols / MXFP4_BLOCK_ELEMENTS];
        if scale.shape != expected {
            bail!(
                "fp4 scale for {weight_name:?} has shape {:?}, expected {expected:?} (one ue8m0 scale per {MXFP4_BLOCK_ELEMENTS}-value group)",
                scale.shape
            );
        }
        Ok((weight, scale, rows, logical_cols))
    }
}

fn parse_tensor_entry(
    name: &str,
    value: &serde_json::Value,
    data_len: usize,
) -> Result<SafetensorsTensor> {
    let entry = value
        .as_object()
        .ok_or_else(|| anyhow!("tensor {name:?} entry is not an object"))?;
    let dtype = entry
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("tensor {name:?} has no dtype string"))?;
    let dtype = SafetensorsDtype::from_name(dtype).with_context(|| format!("tensor {name:?}"))?;
    let shape = entry
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("tensor {name:?} has no shape array"))?
        .iter()
        .map(|dim| {
            dim.as_u64()
                .and_then(|dim| usize::try_from(dim).ok())
                .ok_or_else(|| anyhow!("tensor {name:?} has a non-u64 shape entry"))
        })
        .collect::<Result<Vec<_>>>()?;
    let offsets = entry
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .filter(|v| v.len() == 2)
        .ok_or_else(|| anyhow!("tensor {name:?} has no [begin, end] data_offsets"))?;
    let begin = offsets[0]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| anyhow!("tensor {name:?} has a non-u64 begin offset"))?;
    let end = offsets[1]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| anyhow!("tensor {name:?} has a non-u64 end offset"))?;
    if begin > end || end > data_len {
        bail!(
            "tensor {name:?} data_offsets [{begin}, {end}] fall outside the {data_len}-byte data section"
        );
    }
    let element_count: usize = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("tensor {name:?} shape {shape:?} overflows"))
    })?;
    let expected = element_count * dtype.byte_width();
    if end - begin != expected {
        bail!(
            "tensor {name:?} spans {} bytes but dtype {} with shape {shape:?} needs {expected}",
            end - begin,
            dtype.name()
        );
    }
    Ok(SafetensorsTensor {
        name: name.to_string(),
        dtype,
        shape,
        begin,
        end,
    })
}

/// Require the tensor spans to tile the data section exactly: sorted by
/// offset they must be contiguous from 0 and end at `data_len` (the format
/// forbids gaps and overlaps).
fn validate_coverage(tensors: &[SafetensorsTensor], data_len: usize) -> Result<()> {
    let mut spans: Vec<(usize, usize, &str)> = tensors
        .iter()
        .map(|tensor| (tensor.begin, tensor.end, tensor.name.as_str()))
        .collect();
    spans.sort_unstable();
    let mut cursor = 0usize;
    for (begin, end, name) in spans {
        if begin != cursor {
            bail!(
                "tensor {name:?} starts at data offset {begin} but the previous tensor ended at {cursor} (gap or overlap)"
            );
        }
        cursor = end;
    }
    if cursor != data_len {
        bail!("tensors cover {cursor} bytes but the data section holds {data_len}");
    }
    Ok(())
}

fn le_u16s(bytes: &[u8]) -> impl Iterator<Item = u16> + '_ {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunk length is 2")))
}

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

/// BF16 bits -> f32 (exact: BF16 is the top half of an f32).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// IEEE binary16 bits -> f32 (exact).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let man = u32::from(bits & 0x03ff);
    let out = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal (value = man * 2^-24): renormalize into an f32
            // exponent. `shift` is 10 minus the leading-bit position, so the
            // exponent field is 127 - 24 + (10 - shift) = 113 - shift.
            let shift = man.leading_zeros() - 21;
            let exp = 113 - shift;
            let frac = (man << shift) & 0x03ff;
            sign | (exp << 23) | (frac << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((u32::from(exp) + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// f32 -> IEEE binary16 bits with round-to-nearest-even; overflow becomes
/// infinity, NaN stays NaN.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        // Inf stays inf; NaN keeps a nonzero (quiet) mantissa.
        return if abs > 0x7f80_0000 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }
    let half_exp = (abs >> 23) as i32 - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        // Subnormal (or zero) in f16: shift the implicit-1 mantissa into
        // place. Rounding may carry into the smallest normal, which is the
        // correct result of `sign | rounded` since that carry sets bit 10.
        if half_exp < -10 {
            return sign;
        }
        let man = (abs & 0x007f_ffff) | 0x0080_0000;
        return sign | round_shift_right(man, (14 - half_exp) as u32) as u16;
    }
    let man = abs & 0x007f_ffff;
    let mut out = ((half_exp as u32) << 10) | (man >> 13);
    let dropped = man & 0x1fff;
    if dropped > 0x1000 || (dropped == 0x1000 && out & 1 != 0) {
        // May carry into the exponent (e.g. 0x7bff -> 0x7c00 = inf): correct.
        out += 1;
    }
    sign | out as u16
}

/// `man >> shift` with round-to-nearest-even on the dropped bits.
fn round_shift_right(man: u32, shift: u32) -> u32 {
    let kept = man >> shift;
    let dropped = man & ((1 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if dropped > half || (dropped == half && kept & 1 != 0) {
        kept + 1
    } else {
        kept
    }
}

/// OCP FP8 E4M3FN (`F8_E4M3`, torch `float8_e4m3fn`) -> f32: bias 7,
/// subnormals `man/8 * 2^-6`, no infinities, `S.1111.111` is NaN (max normal
/// is `S.1111.110` = +-448).
pub fn f8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 3) & 0x0f;
    let man = bits & 0x07;
    if exp == 0x0f && man == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exp == 0 {
        f32::from(man) * 2.0f32.powi(-9)
    } else {
        (1.0 + f32::from(man) / 8.0) * 2.0f32.powi(i32::from(exp) - 7)
    };
    sign * magnitude
}

/// OCP ue8m0 (`F8_E8M0`) -> f32: a pure power of two `2^(bits - 127)`;
/// 0xff is NaN. Used as the multiplicative block scale for DeepSeek-V4 FP8
/// and fp4 tensors (`quantization_config.scale_fmt = "ue8m0"`).
pub fn e8m0_to_f32(bits: u8) -> f32 {
    match bits {
        0 => f32::from_bits(0x0040_0000), // 2^-127 (f32 subnormal)
        0xff => f32::NAN,
        _ => f32::from_bits(u32::from(bits) << 23),
    }
}

/// FP4 E2M1 code -> value; the nibble's bit 3 is the sign, matching both the
/// safetensors packing and GGUF `MXFP4` (whose table stores these doubled,
/// paired with a halved scale decode).
pub const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// ---------------------------------------------------------------------------
// Bulk conversions
// ---------------------------------------------------------------------------

/// Raw little-endian BF16 bytes -> f32 values.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("BF16 byte length {} is odd", bytes.len());
    }
    Ok(le_u16s(bytes).map(bf16_to_f32).collect())
}

/// Raw little-endian BF16 bytes -> IEEE f16 bits (round-to-nearest-even).
pub fn bf16_bytes_to_f16(bytes: &[u8]) -> Result<Vec<u16>> {
    if bytes.len() % 2 != 0 {
        bail!("BF16 byte length {} is odd", bytes.len());
    }
    Ok(le_u16s(bytes)
        .map(|bits| f32_to_f16_bits(bf16_to_f32(bits)))
        .collect())
}

/// Dequantize a row-major `[rows, cols]` E4M3FN weight against a
/// `[scale_rows, scale_cols]` ue8m0 scale grid (`value = fp8 * 2^(scale-127)`
/// with scale `[r / block_rows, c / block_cols]`). Block dims are inferred by
/// ceiling division and cross-checked, so uniform grids with partial edge
/// blocks (e.g. DeepSeek's 128x128) all work.
pub fn dequant_fp8_block_scaled(
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    scale_rows: usize,
    scale_cols: usize,
) -> Result<Vec<f32>> {
    if weight.len() != rows * cols {
        bail!(
            "fp8 weight has {} bytes, expected {rows}x{cols}",
            weight.len()
        );
    }
    if scales.len() != scale_rows * scale_cols {
        bail!(
            "fp8 scale has {} bytes, expected {scale_rows}x{scale_cols}",
            scales.len()
        );
    }
    if scale_rows == 0 || scale_cols == 0 || rows == 0 || cols == 0 {
        bail!("fp8 dequant requires non-empty weight and scale grids");
    }
    let block_rows = rows.div_ceil(scale_rows);
    let block_cols = cols.div_ceil(scale_cols);
    if rows.div_ceil(block_rows) != scale_rows || cols.div_ceil(block_cols) != scale_cols {
        bail!("scale grid {scale_rows}x{scale_cols} does not evenly tile a {rows}x{cols} weight");
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let scale_row = &scales[(r / block_rows) * scale_cols..][..scale_cols];
        let weight_row = &weight[r * cols..][..cols];
        let out_row = &mut out[r * cols..][..cols];
        for c in 0..cols {
            out_row[c] = f8_e4m3_to_f32(weight_row[c]) * e8m0_to_f32(scale_row[c / block_cols]);
        }
    }
    Ok(out)
}

/// Values per MXFP4 block (both the safetensors fp4 scale grouping and the
/// GGUF `MXFP4` block).
pub const MXFP4_BLOCK_ELEMENTS: usize = 32;
/// Bytes per GGUF `MXFP4` block: 1 e8m0 scale byte + 16 nibble bytes.
pub const MXFP4_BLOCK_BYTES: usize = 17;

/// The nibble holding logical element `elem` of a group in the safetensors
/// fp4 packing: two consecutive values per byte, low nibble first.
fn fp4_nibble(group: &[u8], elem: usize) -> u8 {
    (group[elem >> 1] >> ((elem & 1) * 4)) & 0x0f
}

fn check_fp4_dims(packed: &[u8], scales: &[u8], rows: usize, logical_cols: usize) -> Result<usize> {
    if logical_cols == 0 || logical_cols % MXFP4_BLOCK_ELEMENTS != 0 {
        bail!(
            "fp4 logical column count {logical_cols} is not a positive multiple of {MXFP4_BLOCK_ELEMENTS}"
        );
    }
    if packed.len() != rows * logical_cols / 2 {
        bail!(
            "fp4 packed weight has {} bytes, expected {rows}x{logical_cols}/2",
            packed.len()
        );
    }
    let groups_per_row = logical_cols / MXFP4_BLOCK_ELEMENTS;
    if scales.len() != rows * groups_per_row {
        bail!(
            "fp4 scale has {} bytes, expected {rows}x{groups_per_row} (one per {MXFP4_BLOCK_ELEMENTS}-value group)",
            scales.len()
        );
    }
    Ok(groups_per_row)
}

/// Dequantize packed fp4 (`[rows, logical_cols/2]` bytes, low nibble = even
/// element) with one ue8m0 scale per 32-value group along the columns.
pub fn dequant_fp4_groups(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    logical_cols: usize,
) -> Result<Vec<f32>> {
    let groups_per_row = check_fp4_dims(packed, scales, rows, logical_cols)?;
    let mut out = vec![0.0f32; rows * logical_cols];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let scale = e8m0_to_f32(scales[r * groups_per_row + g]);
            let group = &packed[r * logical_cols / 2 + g * MXFP4_BLOCK_ELEMENTS / 2..]
                [..MXFP4_BLOCK_ELEMENTS / 2];
            let out_group =
                &mut out[r * logical_cols + g * MXFP4_BLOCK_ELEMENTS..][..MXFP4_BLOCK_ELEMENTS];
            for (elem, slot) in out_group.iter_mut().enumerate() {
                *slot = FP4_E2M1[fp4_nibble(group, elem) as usize] * scale;
            }
        }
    }
    Ok(out)
}

/// Losslessly repack safetensors fp4 (packed pairs + per-32 ue8m0 scale
/// tensor) into the GGUF `MXFP4` byte stream: per 32-value block, the raw
/// scale byte followed by 16 nibble bytes where byte `j` holds element `j`
/// (low nibble) and element `j + 16` (high nibble). Exact for every scale
/// byte except the ue8m0 NaN encoding 0xff (which real weights never use):
/// GGUF decodes scale `s` as `2^(s-128)` against a doubled value table,
/// reproducing our `2^(s-127) * e2m1` bit-for-bit.
pub fn repack_fp4_to_gguf_mxfp4(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    logical_cols: usize,
) -> Result<Vec<u8>> {
    let groups_per_row = check_fp4_dims(packed, scales, rows, logical_cols)?;
    let mut out = Vec::with_capacity(rows * groups_per_row * MXFP4_BLOCK_BYTES);
    for r in 0..rows {
        for g in 0..groups_per_row {
            out.push(scales[r * groups_per_row + g]);
            let group = &packed[r * logical_cols / 2 + g * MXFP4_BLOCK_ELEMENTS / 2..]
                [..MXFP4_BLOCK_ELEMENTS / 2];
            for j in 0..MXFP4_BLOCK_ELEMENTS / 2 {
                out.push(fp4_nibble(group, j) | (fp4_nibble(group, j + 16) << 4));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-cuda-safetensors-{name}-{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    /// Write a syntactically valid safetensors file with contiguous offsets.
    fn write_safetensors(
        path: &Path,
        metadata: Option<&[(&str, &str)]>,
        tensors: &[(&str, &str, Vec<usize>, Vec<u8>)],
    ) {
        let mut entries = serde_json::Map::new();
        if let Some(metadata) = metadata {
            let map: serde_json::Map<String, serde_json::Value> = metadata
                .iter()
                .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
                .collect();
            entries.insert("__metadata__".to_string(), serde_json::Value::Object(map));
        }
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len();
            data.extend_from_slice(bytes);
            entries.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [begin, data.len()],
                }),
            );
        }
        let header = serde_json::Value::Object(entries).to_string().into_bytes();
        let mut out = Vec::with_capacity(8 + header.len() + data.len());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    fn write_raw(path: &Path, header_json: &str, data: &[u8]) {
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        out.extend_from_slice(data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn safetensors_scalar_conversions_match_hand_computed_values() {
        // BF16: top 16 bits of the f32.
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc020), -2.5);
        assert_eq!(bf16_to_f32(0x4049), 3.140625);
        assert_eq!(bf16_to_f32(0x0000), 0.0);

        // f16 decode.
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc100), -2.5);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14)); // smallest normal
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());

        // f16 encode: exact values, RNE ties, subnormals, overflow, NaN.
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.5), 0xc100);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00); // tie rounds up to inf
        assert_eq!(f32_to_f16_bits(65519.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-25)), 0x0000); // tie to even
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-25) * 1.5), 0x0001);
        assert_eq!(f32_to_f16_bits(0.1), 0x2e66);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7e00, 0x7e00);
        // Round-trip everything f16 can hold exactly.
        for bits in [0x3c01u16, 0x4b80, 0x0010, 0x83ff, 0xfbff] {
            assert_eq!(f32_to_f16_bits(f16_to_f32(bits)), bits);
        }

        // FP8 E4M3FN.
        assert_eq!(f8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(f8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(f8_e4m3_to_f32(0x3c), 1.5);
        assert_eq!(f8_e4m3_to_f32(0x40), 2.0);
        assert_eq!(f8_e4m3_to_f32(0xc0), -2.0);
        assert_eq!(f8_e4m3_to_f32(0x01), 2.0f32.powi(-9)); // smallest subnormal
        assert_eq!(f8_e4m3_to_f32(0x87), -7.0 * 2.0f32.powi(-9));
        assert_eq!(f8_e4m3_to_f32(0x7e), 448.0); // max normal
        assert!(f8_e4m3_to_f32(0x7f).is_nan());
        assert!(f8_e4m3_to_f32(0xff).is_nan());
        assert_eq!(f8_e4m3_to_f32(0x80), 0.0); // negative zero

        // ue8m0.
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(115), 2.0f32.powi(-12));
        assert_eq!(e8m0_to_f32(254), 2.0f32.powi(127));
        assert_eq!(e8m0_to_f32(1), 2.0f32.powi(-126));
        assert_eq!(e8m0_to_f32(0), 2.0f32.powi(-127));
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn safetensors_synthetic_file_reads_and_dequantizes_exactly() {
        // BF16 vector: 1.0, -2.5, 0.5, 3.0.
        let bf16: Vec<u8> = [0x3f80u16, 0xc020, 0x3f00, 0x4040]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();

        // FP8 4x4 weight with a 2x2 scale grid (2x2 blocks). Codes:
        //   1.0  2.0  1.5 -2.0        scales: 0.5 (126)  2.0 (128)
        //   0.0  2^-9 448 -3.5        ------- 1.0 (127)  4.0 (129)
        //   1.0  1.0  0.5  0.5
        //  -1.0  8.0  0.25 2.0
        let fp8 = vec![
            0x38, 0x40, 0x3c, 0xc0, // row 0
            0x00, 0x01, 0x7e, 0xbc, // row 1 (0xbc = -1.5)
            0x38, 0x38, 0x30, 0x30, // row 2 (0x30 = 0.5)
            0xb8, 0x50, 0x28, 0x40, // row 3 (0x50 = 8.0, 0x28 = 0.25)
        ];
        let fp8_scales = vec![126u8, 128, 127, 129];
        #[rustfmt::skip]
        let fp8_expected = vec![
            0.5, 1.0, 3.0, -4.0,
            0.0, 2.0f32.powi(-10), 896.0, -3.0,
            1.0, 1.0, 2.0, 2.0,
            -1.0, 8.0, 1.0, 8.0,
        ];

        // fp4 experts: 2 rows x 64 logical cols (2 groups per row). Row 0
        // group 0 walks all 16 codes twice; the rest exercise distinct scales.
        let mut fp4 = Vec::new();
        for _ in 0..2 {
            for j in 0..16u8 {
                // elements 2j = code j, 2j+1 = code 15-j
                fp4.push(j | ((15 - j) << 4));
            }
            for j in 0..16u8 {
                fp4.push((j % 8) | (((j % 8) | 0x8) << 4)); // +v then -v pairs
            }
        }
        let fp4_scales = vec![127u8, 121, 128, 0];
        let mut fp4_expected = Vec::new();
        for row in 0..2 {
            for group in 0..2 {
                let scale = e8m0_to_f32(fp4_scales[row * 2 + group]);
                for j in 0..16usize {
                    if group == 0 {
                        fp4_expected.push(FP4_E2M1[j] * scale);
                        fp4_expected.push(FP4_E2M1[15 - j] * scale);
                    } else {
                        fp4_expected.push(FP4_E2M1[j % 8] * scale);
                        fp4_expected.push(FP4_E2M1[(j % 8) | 0x8] * scale);
                    }
                }
            }
        }

        let path = tempfile_path("synthetic");
        write_safetensors(
            &path,
            Some(&[("format", "test"), ("who", "hi-cuda")]),
            &[
                ("norm.weight", "BF16", vec![4], bf16),
                ("proj.weight", "F8_E4M3", vec![4, 4], fp8),
                ("proj.scale", "F8_E8M0", vec![2, 2], fp8_scales),
                ("experts.0.w1.weight", "I8", vec![2, 32], fp4.clone()),
                (
                    "experts.0.w1.scale",
                    "F8_E8M0",
                    vec![2, 2],
                    fp4_scales.clone(),
                ),
                (
                    "ids",
                    "I64",
                    vec![3],
                    7i64.to_le_bytes()
                        .iter()
                        .chain(0i64.to_le_bytes().iter())
                        .chain((-2i64).to_le_bytes().iter())
                        .copied()
                        .collect(),
                ),
            ],
        );
        let file = SafetensorsFile::open(&path).unwrap();

        // Names, metadata, info, zero-copy bytes.
        assert_eq!(file.tensors().len(), 6);
        let names: Vec<&str> = file.names().collect();
        assert!(
            names.windows(2).all(|w| w[0] < w[1]),
            "names sorted: {names:?}"
        );
        assert_eq!(file.metadata().unwrap().get("format").unwrap(), "test");
        let info = file.info("proj.weight").unwrap();
        assert_eq!(info.dtype, SafetensorsDtype::F8E4M3);
        assert_eq!(info.shape, vec![4, 4]);
        assert_eq!(info.element_count(), 16);
        assert_eq!(file.bytes("proj.scale").unwrap(), &[126, 128, 127, 129]);
        assert!(file.info("missing").is_none());
        assert!(file.bytes("missing").is_err());

        // BF16 -> f32 / f16.
        assert_eq!(
            file.tensor_f32("norm.weight").unwrap(),
            vec![1.0, -2.5, 0.5, 3.0]
        );
        assert_eq!(
            file.tensor_f16("norm.weight").unwrap(),
            vec![0x3c00, 0xc100, 0x3800, 0x4200]
        );
        assert!(
            file.tensor_f32("proj.weight").is_err(),
            "fp8 needs the scale path"
        );

        // I64.
        assert_eq!(file.tensor_i64("ids").unwrap(), vec![7, 0, -2]);
        assert!(file.tensor_i64("norm.weight").is_err());

        // FP8 block dequant, f32 and f16.
        assert_eq!(
            file.fp8_block_scaled_f32("proj.weight").unwrap(),
            fp8_expected
        );
        assert_eq!(
            file.fp8_block_scaled_f16("proj.weight").unwrap(),
            fp8_expected
                .iter()
                .map(|&v| f32_to_f16_bits(v))
                .collect::<Vec<_>>()
        );

        // fp4 dequant against hand-computed values.
        assert_eq!(
            file.fp4_block_scaled_f32("experts.0.w1.weight").unwrap(),
            fp4_expected
        );

        // Repack to GGUF MXFP4 and decode with hi-gguf itself: the values the
        // expert pool will see must match the safetensors dequant exactly.
        let repacked = file.fp4_to_gguf_mxfp4("experts.0.w1.weight").unwrap();
        assert_eq!(repacked.len(), 4 * MXFP4_BLOCK_BYTES);
        assert_eq!(repacked[0], 127, "scale byte copies verbatim");
        assert_eq!(repacked[MXFP4_BLOCK_BYTES], 121);
        let via_gguf =
            hi_gguf::dequantize_tensor_as_f32(&repacked, hi_gguf::GgufTensorType::MXFP4, 2 * 64)
                .unwrap();
        assert_eq!(via_gguf, fp4_expected);

        // Free-function equivalents used on raw slices.
        assert_eq!(
            dequant_fp4_groups(&fp4, &fp4_scales, 2, 64).unwrap(),
            fp4_expected
        );
        assert_eq!(
            repack_fp4_to_gguf_mxfp4(&fp4, &fp4_scales, 2, 64).unwrap(),
            repacked
        );
        assert_eq!(bf16_bytes_to_f32(&[0x80, 0x3f]).unwrap(), vec![1.0]);
        assert_eq!(bf16_bytes_to_f16(&[0x80, 0x3f]).unwrap(), vec![0x3c00]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn safetensors_open_rejects_malformed_files() {
        // Gap between tensors.
        let path = tempfile_path("gap");
        write_raw(
            &path,
            r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}}"#,
            &[0; 6],
        );
        let err = SafetensorsFile::open(&path).unwrap_err().to_string();
        assert!(
            format!("{err:#}").contains("gap") || err.contains("gap"),
            "got: {err}"
        );
        std::fs::remove_file(&path).ok();

        // Overlapping tensors.
        let path = tempfile_path("overlap");
        write_raw(
            &path,
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#,
            &[0; 6],
        );
        assert!(SafetensorsFile::open(&path).is_err());
        std::fs::remove_file(&path).ok();

        // Trailing uncovered bytes.
        let path = tempfile_path("trailing");
        write_raw(
            &path,
            r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
            &[0; 4],
        );
        assert!(SafetensorsFile::open(&path).is_err());
        std::fs::remove_file(&path).ok();

        // Shape/byte-length mismatch.
        let path = tempfile_path("shapelen");
        write_raw(
            &path,
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            &[0; 8],
        );
        let err = SafetensorsFile::open(&path).unwrap_err();
        assert!(format!("{err:#}").contains("needs 12"), "got: {err:#}");
        std::fs::remove_file(&path).ok();

        // Header longer than the file (truncated download).
        let path = tempfile_path("truncated");
        std::fs::write(&path, 1_000_000u64.to_le_bytes()).unwrap();
        let err = SafetensorsFile::open(&path).unwrap_err();
        assert!(format!("{err:#}").contains("truncated"), "got: {err:#}");
        std::fs::remove_file(&path).ok();

        // Unknown dtype.
        let path = tempfile_path("dtype");
        write_raw(
            &path,
            r#"{"a":{"dtype":"Q4_K","shape":[2],"data_offsets":[0,2]}}"#,
            &[0; 2],
        );
        assert!(SafetensorsFile::open(&path).is_err());
        std::fs::remove_file(&path).ok();

        // Offsets past the data section.
        let path = tempfile_path("oob");
        write_raw(
            &path,
            r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            &[0; 2],
        );
        assert!(SafetensorsFile::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    // -----------------------------------------------------------------------
    // Real-artifact validation (skipped when the downloads are absent).
    // -----------------------------------------------------------------------

    fn real_artifact_path(env: &str, suffix: &str) -> Option<PathBuf> {
        if let Some(path) = std::env::var_os(env) {
            let path = PathBuf::from(path);
            return path.exists().then_some(path);
        }
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home)
            .join(".hi/models/deepseek-v4-flash")
            .join(suffix);
        path.exists().then_some(path)
    }

    #[test]
    fn safetensors_real_mtp_shard_census_and_spot_dequant() {
        let Some(path) =
            real_artifact_path("HI_DSV4_MTP_PATH", "mtp/model-00046-of-00046.safetensors")
        else {
            eprintln!("skipping: MTP shard not found");
            return;
        };
        // open() already validates dtype/shape/offset consistency and
        // gap-free coverage of the whole data section.
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(file.tensors().len(), 1575);
        assert!(file.names().all(|name| name.starts_with("mtp.0.")));

        // Attention: FP8 E4M3 weights with 128x128 ue8m0 block scales.
        for (name, shape, scale_shape) in [
            ("mtp.0.attn.wq_a", vec![1024, 4096], vec![8, 32]),
            ("mtp.0.attn.wq_b", vec![32768, 1024], vec![256, 8]),
            ("mtp.0.attn.wkv", vec![512, 4096], vec![4, 32]),
            ("mtp.0.attn.wo_a", vec![8192, 4096], vec![64, 32]),
            ("mtp.0.attn.wo_b", vec![4096, 8192], vec![32, 64]),
            ("mtp.0.e_proj", vec![4096, 4096], vec![32, 32]),
            ("mtp.0.h_proj", vec![4096, 4096], vec![32, 32]),
            (
                "mtp.0.ffn.shared_experts.w1",
                vec![2048, 4096],
                vec![16, 32],
            ),
            (
                "mtp.0.ffn.shared_experts.w2",
                vec![4096, 2048],
                vec![32, 16],
            ),
            (
                "mtp.0.ffn.shared_experts.w3",
                vec![2048, 4096],
                vec![16, 32],
            ),
        ] {
            let weight = file.info(&format!("{name}.weight")).unwrap();
            assert_eq!(weight.dtype, SafetensorsDtype::F8E4M3, "{name}");
            assert_eq!(weight.shape, shape, "{name}");
            let scale = file.info(&format!("{name}.scale")).unwrap();
            assert_eq!(scale.dtype, SafetensorsDtype::F8E8M0, "{name}");
            assert_eq!(scale.shape, scale_shape, "{name}");
        }

        // Norms and small vectors.
        for (name, dtype, shape) in [
            ("mtp.0.attn.attn_sink", SafetensorsDtype::F32, vec![64]),
            (
                "mtp.0.attn.q_norm.weight",
                SafetensorsDtype::BF16,
                vec![1024],
            ),
            (
                "mtp.0.attn.kv_norm.weight",
                SafetensorsDtype::BF16,
                vec![512],
            ),
            ("mtp.0.attn_norm.weight", SafetensorsDtype::BF16, vec![4096]),
            ("mtp.0.ffn_norm.weight", SafetensorsDtype::BF16, vec![4096]),
            ("mtp.0.enorm.weight", SafetensorsDtype::BF16, vec![4096]),
            ("mtp.0.hnorm.weight", SafetensorsDtype::BF16, vec![4096]),
            ("mtp.0.norm.weight", SafetensorsDtype::BF16, vec![4096]),
            (
                "mtp.0.ffn.gate.weight",
                SafetensorsDtype::BF16,
                vec![256, 4096],
            ),
            ("mtp.0.ffn.gate.bias", SafetensorsDtype::F32, vec![256]),
            ("mtp.0.hc_attn_fn", SafetensorsDtype::F32, vec![24, 16384]),
            ("mtp.0.hc_attn_base", SafetensorsDtype::F32, vec![24]),
            ("mtp.0.hc_attn_scale", SafetensorsDtype::F32, vec![3]),
            ("mtp.0.hc_ffn_fn", SafetensorsDtype::F32, vec![24, 16384]),
            ("mtp.0.hc_head_fn", SafetensorsDtype::F32, vec![4, 16384]),
            ("mtp.0.hc_head_base", SafetensorsDtype::F32, vec![4]),
            ("mtp.0.hc_head_scale", SafetensorsDtype::F32, vec![1]),
        ] {
            let info = file.info(name).unwrap_or_else(|| panic!("{name} missing"));
            assert_eq!(info.dtype, dtype, "{name}");
            assert_eq!(info.shape, shape, "{name}");
        }
        // No embedding / lm head in the shard: shared with the target.
        assert!(
            !file
                .names()
                .any(|n| n.contains("emb") && !n.contains("enorm"))
        );

        // Routed experts 0..=255: fp4-packed with per-32 ue8m0 scales
        // (MXFP4-32) — the direct-repack contract.
        for expert in [0usize, 128, 255] {
            for (proj, shape, scale_shape) in [
                ("w1", vec![2048, 2048], vec![2048, 128]),
                ("w2", vec![4096, 1024], vec![4096, 64]),
                ("w3", vec![2048, 2048], vec![2048, 128]),
            ] {
                let base = format!("mtp.0.ffn.experts.{expert}.{proj}");
                let weight = file.info(&format!("{base}.weight")).unwrap();
                assert_eq!(weight.dtype, SafetensorsDtype::I8, "{base}");
                assert_eq!(weight.shape, shape, "{base}");
                let scale = file.info(&format!("{base}.scale")).unwrap();
                assert_eq!(scale.dtype, SafetensorsDtype::F8E8M0, "{base}");
                assert_eq!(scale.shape, scale_shape, "{base}");
            }
        }

        // Spot dequants: norm vectors are O(1) and positive-mean.
        for name in [
            "mtp.0.norm.weight",
            "mtp.0.attn_norm.weight",
            "mtp.0.enorm.weight",
        ] {
            let values = file.tensor_f32(name).unwrap();
            assert!(values.iter().all(|v| v.is_finite()), "{name}");
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            assert!((0.001..10.0).contains(&mean), "{name} mean {mean}");
            assert!(values.iter().all(|v| v.abs() < 100.0), "{name}");
        }

        // FP8 shared expert dequants to sane weight magnitudes.
        let shared = file
            .fp8_block_scaled_f32("mtp.0.ffn.shared_experts.w1.weight")
            .unwrap();
        assert_eq!(shared.len(), 2048 * 4096);
        let rms = (shared
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / shared.len() as f64)
            .sqrt();
        assert!((1e-4..1.0).contains(&rms), "shared w1 rms {rms}");

        // fp4 expert: repack and decode through hi-gguf's own MXFP4 decoder;
        // must match the direct safetensors dequant exactly.
        let direct = file
            .fp4_block_scaled_f32("mtp.0.ffn.experts.0.w1.weight")
            .unwrap();
        let repacked = file
            .fp4_to_gguf_mxfp4("mtp.0.ffn.experts.0.w1.weight")
            .unwrap();
        assert_eq!(repacked.len(), 2048 * (4096 / 32) * MXFP4_BLOCK_BYTES);
        let via_gguf = hi_gguf::dequantize_tensor_as_f32(
            &repacked,
            hi_gguf::GgufTensorType::MXFP4,
            2048 * 4096,
        )
        .unwrap();
        assert_eq!(via_gguf, direct);
        let mean_abs = direct.iter().map(|v| f64::from(v.abs())).sum::<f64>() / direct.len() as f64;
        assert!(
            (1e-4..1.0).contains(&mean_abs),
            "expert w1 mean|w| {mean_abs}"
        );
    }

    #[test]
    fn safetensors_real_dflash_census() {
        let Some(path) =
            real_artifact_path("HI_DSV4_DFLASH_PATH", "dflash-redhat/model.safetensors")
        else {
            eprintln!("skipping: DFlash checkpoint not found");
            return;
        };
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(file.tensors().len(), 62);
        assert_eq!(file.metadata().unwrap().get("format").unwrap(), "pt");

        // fc combiner: in_features 81920 = 5 aux layers x flat hc_mult*4096
        // (resolves the plan's flat-vs-averaged conditioning question: FLAT).
        let fc = file.info("fc.weight").unwrap();
        assert_eq!(fc.dtype, SafetensorsDtype::BF16);
        assert_eq!(fc.shape, vec![4096, 81920]);

        // Vocab plumbing: reduced 32000-entry draft vocab with d2t map (and a
        // bonus t2d membership mask); embed_tokens spans the full target
        // vocab, lm_head only the draft vocab.
        assert_eq!(
            file.info("embed_tokens.weight").unwrap().shape,
            vec![129280, 4096]
        );
        assert_eq!(
            file.info("lm_head.weight").unwrap().shape,
            vec![32000, 4096]
        );
        let d2t_info = file.info("d2t").unwrap();
        assert_eq!(d2t_info.dtype, SafetensorsDtype::I64);
        assert_eq!(d2t_info.shape, vec![32000]);
        let t2d = file.info("t2d").unwrap();
        assert_eq!(t2d.dtype, SafetensorsDtype::Bool);
        assert_eq!(t2d.shape, vec![129280]);

        // d2t is an OFFSET map (vLLM speculators convention), not direct ids:
        // target_id = draft_id + d2t[draft_id]. Offsets are nondecreasing, so
        // mapped target ids are strictly increasing (32000 unique), and every
        // mapped id is marked in the t2d membership mask (which marks exactly
        // the draft vocab).
        let d2t = file.tensor_i64("d2t").unwrap();
        let t2d_bytes = file.bytes("t2d").unwrap();
        assert_eq!(t2d_bytes.iter().filter(|&&b| b != 0).count(), 32000);
        assert!(
            d2t.windows(2).all(|w| w[0] <= w[1]),
            "d2t offsets nondecreasing"
        );
        for (draft, &offset) in d2t.iter().enumerate() {
            let target = draft as i64 + offset;
            assert!((0..129280).contains(&target), "d2t[{draft}] = {offset}");
            assert_ne!(t2d_bytes[target as usize], 0, "d2t[{draft}] -> {target}");
        }

        // No dedicated mask-embedding tensor: the mask query embedding is
        // embed_tokens[mask_token_id = 1].
        assert!(file.info("mask_embedding").is_none());
        assert!(!file.names().any(|n| n.contains("mask")));

        // 5 llama-style layers with Qwen3-style per-head-dim q/k norms,
        // 64 Q heads x head_dim 256, 1 KV head, SwiGLU intermediate 2048.
        for layer in 0..5 {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![4096usize]),
                ("post_attention_layernorm.weight", vec![4096]),
                ("self_attn.q_proj.weight", vec![16384, 4096]),
                ("self_attn.k_proj.weight", vec![256, 4096]),
                ("self_attn.v_proj.weight", vec![256, 4096]),
                ("self_attn.o_proj.weight", vec![4096, 16384]),
                ("self_attn.q_norm.weight", vec![256]),
                ("self_attn.k_norm.weight", vec![256]),
                ("mlp.gate_proj.weight", vec![2048, 4096]),
                ("mlp.up_proj.weight", vec![2048, 4096]),
                ("mlp.down_proj.weight", vec![4096, 2048]),
            ] {
                let name = format!("layers.{layer}.{suffix}");
                let info = file.info(&name).unwrap_or_else(|| panic!("{name} missing"));
                assert_eq!(info.dtype, SafetensorsDtype::BF16, "{name}");
                assert_eq!(info.shape, shape, "{name}");
            }
        }
        assert_eq!(file.info("norm.weight").unwrap().shape, vec![4096]);
        assert_eq!(file.info("hidden_norm.weight").unwrap().shape, vec![4096]);

        // Spot dequant: final norm is O(1).
        let norm = file.tensor_f32("norm.weight").unwrap();
        assert!(norm.iter().all(|v| v.is_finite() && v.abs() < 100.0));
        let mean = norm.iter().sum::<f32>() / norm.len() as f32;
        assert!((0.001..10.0).contains(&mean), "norm mean {mean}");
    }
}
