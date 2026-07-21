//! GGUF tensor dtype definitions and dequantization to f32.
//!
//! Extracted from `lib.rs` as a pure code move; all public items are
//! re-exported from the crate root so `hi_gguf::X` paths are unchanged.

use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GgufTensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4_0_4_4,
    Q4_0_4_8,
    Q4_0_8_8,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ2_S,
    IQ3_S,
    IQ4_NL,
    IQ4_NL_4_4,
    IQ4_NL_4_8,
    IQ4_NL_8_8,
    IQ4_XS,
    IQ1_M,
    I8,
    I16,
    I32,
    I64,
    F64,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    TQ1_0,
    TQ2_0,
    BF16,
}

impl GgufTensorType {
    pub(crate) fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            22 => Ok(Self::IQ2_S),
            21 => Ok(Self::IQ3_S),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            20 => Ok(Self::IQ4_NL),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            31 => Ok(Self::Q4_0_4_4),
            32 => Ok(Self::Q4_0_4_8),
            33 => Ok(Self::Q4_0_8_8),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            36 => Ok(Self::IQ4_NL_4_4),
            37 => Ok(Self::IQ4_NL_4_8),
            38 => Ok(Self::IQ4_NL_8_8),
            39 => Ok(Self::MXFP4),
            40 => Ok(Self::NVFP4),
            41 => Ok(Self::Q1_0),
            other => Err(unsupported_tensor_type_error(other, None)),
        }
    }

    pub fn element_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::MXFP4
            | Self::NVFP4
            | Self::Q1_0
            | Self::Q4_0
            | Self::Q4_0_4_4
            | Self::Q4_0_4_8
            | Self::Q4_0_8_8
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ2_S
            | Self::IQ3_S
            | Self::IQ4_NL
            | Self::IQ4_NL_4_4
            | Self::IQ4_NL_4_8
            | Self::IQ4_NL_8_8
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::TQ1_0
            | Self::TQ2_0 => 0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q4_0_4_4 => "Q4_0_4_4",
            Self::Q4_0_4_8 => "Q4_0_4_8",
            Self::Q4_0_8_8 => "Q4_0_8_8",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ3_S => "IQ3_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ4_NL_4_4 => "IQ4_NL_4_4",
            Self::IQ4_NL_4_8 => "IQ4_NL_4_8",
            Self::IQ4_NL_8_8 => "IQ4_NL_8_8",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ1_M => "IQ1_M",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::MXFP4 => "MXFP4",
            Self::NVFP4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::BF16 => "BF16",
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::MXFP4
                | Self::NVFP4
                | Self::Q1_0
                | Self::Q4_0
                | Self::Q4_0_4_4
                | Self::Q4_0_4_8
                | Self::Q4_0_8_8
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::IQ2_XXS
                | Self::IQ2_XS
                | Self::IQ3_XXS
                | Self::IQ1_S
                | Self::IQ2_S
                | Self::IQ3_S
                | Self::IQ4_NL
                | Self::IQ4_NL_4_4
                | Self::IQ4_NL_4_8
                | Self::IQ4_NL_8_8
                | Self::IQ4_XS
                | Self::IQ1_M
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_K
                | Self::TQ1_0
                | Self::TQ2_0
        )
    }

    pub fn block_element_count(self) -> Option<u64> {
        match self {
            Self::Q4_0
            | Self::Q4_1
            | Self::Q4_0_4_4
            | Self::Q4_0_4_8
            | Self::Q4_0_8_8
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::IQ4_NL_4_4
            | Self::IQ4_NL_4_8
            | Self::IQ4_NL_8_8
            | Self::MXFP4 => Some(32),
            Self::NVFP4 => Some(64),
            Self::Q1_0 => Some(128),
            Self::IQ4_XS
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ2_S
            | Self::IQ3_S
            | Self::Q2_K
            | Self::IQ1_M
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::TQ1_0
            | Self::TQ2_0 => Some(256),
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => None,
        }
    }

    pub fn block_byte_len(self) -> Option<u64> {
        match self {
            Self::MXFP4 => Some(17),
            Self::NVFP4 => Some(36),
            Self::Q1_0 => Some(18),
            Self::Q4_0 | Self::Q4_0_4_4 | Self::Q4_0_4_8 | Self::Q4_0_8_8 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            Self::IQ2_XXS => Some(66),
            Self::IQ2_XS => Some(74),
            Self::IQ3_XXS => Some(98),
            Self::IQ1_S => Some(50),
            Self::IQ2_S => Some(82),
            Self::IQ3_S => Some(110),
            Self::IQ4_NL | Self::IQ4_NL_4_4 | Self::IQ4_NL_4_8 | Self::IQ4_NL_8_8 => Some(18),
            Self::IQ4_XS => Some(136),
            Self::IQ1_M => Some(56),
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            Self::TQ1_0 => Some(54),
            Self::TQ2_0 => Some(66),
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => None,
        }
    }

    pub fn byte_len(self, element_count: u64) -> Result<u64> {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => element_count
                .checked_mul(self.element_size())
                .ok_or_else(|| anyhow!("dense tensor byte length overflows u64")),
            Self::MXFP4
            | Self::NVFP4
            | Self::Q1_0
            | Self::Q4_0
            | Self::Q4_0_4_4
            | Self::Q4_0_4_8
            | Self::Q4_0_8_8
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ2_S
            | Self::IQ3_S
            | Self::IQ4_NL
            | Self::IQ4_NL_4_4
            | Self::IQ4_NL_4_8
            | Self::IQ4_NL_8_8
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::TQ1_0
            | Self::TQ2_0 => {
                let block_elements = self.block_element_count().expect("quantized block size");
                let block_bytes = self.block_byte_len().expect("quantized block byte length");
                if !element_count.is_multiple_of(block_elements) {
                    bail!(
                        "{} tensor element count {element_count} is not divisible by block size {block_elements}",
                        self.label()
                    );
                }
                element_count
                    .checked_div(block_elements)
                    .and_then(|blocks| blocks.checked_mul(block_bytes))
                    .ok_or_else(|| anyhow!("quantized tensor byte length overflows u64"))
            }
        }
    }
}

fn unsupported_tensor_type_name(raw: u32) -> Option<&'static str> {
    match raw {
        other if other > 41 => Some("future GGUF tensor type"),
        _ => None,
    }
}

fn supported_tensor_type_list() -> &'static str {
    "F32, F16, BF16, I8, I16, I32, I64, F64, Q8_0, Q8_1, Q5_0, Q5_1, Q4_0, Q4_1, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, MXFP4, NVFP4, Q1_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ2_S, IQ3_S, IQ4_NL, IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8, IQ4_XS, IQ1_M, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, TQ1_0, and TQ2_0"
}

fn tensor_type_label(raw: u32) -> String {
    match unsupported_tensor_type_name(raw) {
        Some(name) => format!("{raw} ({name})"),
        None => raw.to_string(),
    }
}

pub(crate) fn unsupported_tensor_type_error(raw: u32, tensor_name: Option<&str>) -> anyhow::Error {
    let supported = supported_tensor_type_list();
    let label = tensor_type_label(raw);
    match tensor_name {
        Some(tensor_name) => anyhow!(
            "tensor {tensor_name} has unsupported GGUF tensor type {label}; hi-local currently accepts {supported} tensors"
        ),
        None => anyhow!(
            "unsupported GGUF tensor type {label}; hi-local currently accepts {supported} tensors"
        ),
    }
}
pub fn dequantize_tensor_as_f32(
    bytes: &[u8],
    dtype: GgufTensorType,
    element_count: usize,
) -> Result<Vec<f32>> {
    match dtype {
        GgufTensorType::F32 => read_f32_tensor(bytes, element_count),
        GgufTensorType::F16 => read_f16_tensor(bytes, element_count),
        GgufTensorType::BF16 => read_bf16_tensor(bytes, element_count),
        GgufTensorType::I8 => read_i8_tensor(bytes, element_count),
        GgufTensorType::I16 => read_i16_tensor(bytes, element_count),
        GgufTensorType::I32 => read_i32_tensor(bytes, element_count),
        GgufTensorType::I64 => read_i64_tensor(bytes, element_count),
        GgufTensorType::F64 => read_f64_tensor(bytes, element_count),
        GgufTensorType::MXFP4 => dequantize_mxfp4(bytes, element_count),
        GgufTensorType::NVFP4 => dequantize_nvfp4(bytes, element_count),
        GgufTensorType::Q1_0 => dequantize_q1_0(bytes, element_count),
        GgufTensorType::Q4_0
        | GgufTensorType::Q4_0_4_4
        | GgufTensorType::Q4_0_4_8
        | GgufTensorType::Q4_0_8_8 => dequantize_q4_0(bytes, element_count),
        GgufTensorType::Q4_1 => dequantize_q4_1(bytes, element_count),
        GgufTensorType::Q5_0 => dequantize_q5_0(bytes, element_count),
        GgufTensorType::Q5_1 => dequantize_q5_1(bytes, element_count),
        GgufTensorType::Q8_0 => dequantize_q8_0(bytes, element_count),
        GgufTensorType::Q8_1 => dequantize_q8_1(bytes, element_count),
        GgufTensorType::IQ2_XXS => dequantize_iq2_xxs(bytes, element_count),
        GgufTensorType::IQ2_XS => dequantize_iq2_xs(bytes, element_count),
        GgufTensorType::IQ3_XXS => dequantize_iq3_xxs(bytes, element_count),
        GgufTensorType::IQ1_S => dequantize_iq1_s(bytes, element_count),
        GgufTensorType::IQ2_S => dequantize_iq2_s(bytes, element_count),
        GgufTensorType::IQ3_S => dequantize_iq3_s(bytes, element_count),
        GgufTensorType::IQ4_NL
        | GgufTensorType::IQ4_NL_4_4
        | GgufTensorType::IQ4_NL_4_8
        | GgufTensorType::IQ4_NL_8_8 => dequantize_iq4_nl(bytes, element_count),
        GgufTensorType::IQ4_XS => dequantize_iq4_xs(bytes, element_count),
        GgufTensorType::IQ1_M => dequantize_iq1_m(bytes, element_count),
        GgufTensorType::Q2_K => dequantize_q2_k(bytes, element_count),
        GgufTensorType::Q3_K => dequantize_q3_k(bytes, element_count),
        GgufTensorType::Q4_K => dequantize_q4_k(bytes, element_count),
        GgufTensorType::Q5_K => dequantize_q5_k(bytes, element_count),
        GgufTensorType::Q6_K => dequantize_q6_k(bytes, element_count),
        GgufTensorType::Q8_K => dequantize_q8_k(bytes, element_count),
        GgufTensorType::TQ1_0 => dequantize_tq1_0(bytes, element_count),
        GgufTensorType::TQ2_0 => dequantize_tq2_0(bytes, element_count),
    }
}
fn read_f32_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 4 {
        bail!(
            "F32 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            Ok(f32::from_le_bytes(
                chunk.try_into().expect("chunk length is 4"),
            ))
        })
        .collect()
}

fn read_f16_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 2 {
        bail!(
            "F16 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let raw = u16::from_le_bytes(chunk.try_into().expect("chunk length is 2"));
            Ok(f16_to_f32(raw))
        })
        .collect()
}

fn read_bf16_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 2 {
        bail!(
            "BF16 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let raw = u16::from_le_bytes(chunk.try_into().expect("chunk length is 2"));
            Ok(f32::from_bits(u32::from(raw) << 16))
        })
        .collect()
}

fn read_i8_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count {
        bail!(
            "I8 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    Ok(bytes
        .iter()
        .map(|value| f32::from(i8::from_le_bytes([*value])))
        .collect())
}

fn read_i16_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 2 {
        bail!(
            "I16 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            Ok(f32::from(i16::from_le_bytes(
                chunk.try_into().expect("chunk length is 2"),
            )))
        })
        .collect()
}

fn read_i32_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 4 {
        bail!(
            "I32 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| Ok(i32::from_le_bytes(chunk.try_into().expect("chunk length is 4")) as f32))
        .collect()
}

fn read_i64_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 8 {
        bail!(
            "I64 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(8)
        .map(|chunk| Ok(i64::from_le_bytes(chunk.try_into().expect("chunk length is 8")) as f32))
        .collect()
}

fn read_f64_tensor(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    if bytes.len() != element_count * 8 {
        bail!(
            "F64 tensor byte length {} does not match element count {element_count}",
            bytes.len()
        );
    }
    bytes
        .chunks_exact(8)
        .map(|chunk| Ok(f64::from_le_bytes(chunk.try_into().expect("chunk length is 8")) as f32))
        .collect()
}

const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn e8m0_to_f32_half(value: u8) -> f32 {
    let bits = if value < 2 {
        0x0020_0000u32 << u32::from(value)
    } else {
        (u32::from(value) - 1) << 23
    };
    f32::from_bits(bits)
}

fn ue4m3_to_f32(value: u8) -> f32 {
    if value == 0 || value == 0x7f {
        return 0.0;
    }
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    let raw = if exponent == 0 {
        f32::from(mantissa) * 2.0f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    raw * 0.5
}

fn dequantize_mxfp4(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 17;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "MXFP4")?;
    let mut out = vec![0.0f32; element_count];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = e8m0_to_f32_half(block[0]);
        let qs = &block[1..];
        let base = block_idx * BLOCK_ELEMENTS;
        for j in 0..(BLOCK_ELEMENTS / 2) {
            let packed = qs[j];
            out[base + j] = d * f32::from(MXFP4_VALUES[(packed & 0x0f) as usize]);
            out[base + j + BLOCK_ELEMENTS / 2] =
                d * f32::from(MXFP4_VALUES[(packed >> 4) as usize]);
        }
    }
    Ok(out)
}

fn dequantize_nvfp4(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 64;
    const BLOCK_BYTES: usize = 36;
    const SUB_BLOCK_ELEMENTS: usize = 16;
    const SUB_BLOCK_QS: usize = 8;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "NVFP4")?;
    let mut out = vec![0.0f32; element_count];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let qs = &block[4..];
        let base = block_idx * BLOCK_ELEMENTS;
        for sub in 0..(BLOCK_ELEMENTS / SUB_BLOCK_ELEMENTS) {
            let d = ue4m3_to_f32(block[sub]);
            let sub_base = base + sub * SUB_BLOCK_ELEMENTS;
            let sub_qs = &qs[sub * SUB_BLOCK_QS..sub * SUB_BLOCK_QS + SUB_BLOCK_QS];
            for j in 0..SUB_BLOCK_QS {
                let packed = sub_qs[j];
                out[sub_base + j] = d * f32::from(MXFP4_VALUES[(packed & 0x0f) as usize]);
                out[sub_base + j + SUB_BLOCK_QS] =
                    d * f32::from(MXFP4_VALUES[(packed >> 4) as usize]);
            }
        }
    }
    Ok(out)
}

fn dequantize_q1_0(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 128;
    const BLOCK_BYTES: usize = 18;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q1_0")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        for j in 0..BLOCK_ELEMENTS {
            let bit = (qs[j / 8] >> (j % 8)) & 1;
            out.push(if bit != 0 { d } else { -d });
        }
    }
    Ok(out)
}

fn dequantize_q8_0(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q8_0")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for value in &block[2..] {
            out.push(d * f32::from(*value as i8));
        }
    }
    Ok(out)
}

fn dequantize_q8_1(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 36;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q8_1")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for value in &block[4..] {
            out.push(d * f32::from(*value as i8));
        }
    }
    Ok(out)
}

fn dequantize_q4_0(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q4_0")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        for byte in qs {
            out.push(d * f32::from((byte & 0x0f) as i8 - 8));
        }
        for byte in qs {
            out.push(d * f32::from((byte >> 4) as i8 - 8));
        }
    }
    Ok(out)
}

fn dequantize_q4_1(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 20;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q4_1")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qs = &block[4..];
        for byte in qs {
            out.push(d * f32::from(byte & 0x0f) + m);
        }
        for byte in qs {
            out.push(d * f32::from(byte >> 4) + m);
        }
    }
    Ok(out)
}

const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn dequantize_iq4_nl(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ4_NL")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        for byte in qs {
            out.push(d * f32::from(IQ4_NL_VALUES[(byte & 0x0f) as usize]));
        }
        for byte in qs {
            out.push(d * f32::from(IQ4_NL_VALUES[(byte >> 4) as usize]));
        }
    }
    Ok(out)
}

fn dequantize_iq4_xs(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 136;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ4_XS")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        for ib in 0..8 {
            let scale_low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
            let scale_high = ((scales_h >> (2 * ib)) & 0x03) as u8;
            let scale = (scale_low | (scale_high << 4)) as i32 - 32;
            let dl = d * scale as f32;
            let q = &qs[ib * 16..ib * 16 + 16];
            for byte in q {
                out.push(dl * f32::from(IQ4_NL_VALUES[(byte & 0x0f) as usize]));
            }
            for byte in q {
                out.push(dl * f32::from(IQ4_NL_VALUES[(byte >> 4) as usize]));
            }
        }
    }
    Ok(out)
}

const IQ2_XXS_GRID: [u16; 256] = [
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130, 138, 162, 257, 260,
    272, 277, 320, 388, 408, 512, 514, 546, 642, 1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096,
    1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288, 1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600,
    1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580,
    4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240, 4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644,
    4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400, 5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273,
    6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217,
    9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872, 10248, 10272, 10388, 10820, 16385, 16388,
    16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516, 16528, 16640, 16672,
    16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013,
    19088, 20480, 20488, 20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648,
    21650, 21770, 22017, 22100, 22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592, 24640,
    24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874, 25988, 26690, 32768, 32770,
    32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856, 33888,
    34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928,
    37025, 37142, 37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093,
    41225, 41472, 42008, 43088, 43268,
];

const IQ2_XXS_VALUES: [u8; 4] = [8, 25, 43, 0];

const IQ2_XS_GRID: [u16; 512] = [
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70, 73, 80, 82, 85, 88, 97, 100, 128,
    130, 133, 136, 145, 148, 153, 160, 257, 260, 262, 265, 272, 274, 277, 280, 282, 289, 292, 320,
    322, 325, 328, 337, 340, 352, 360, 385, 388, 400, 512, 514, 517, 520, 529, 532, 544, 577, 580,
    592, 597, 640, 650, 1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1088, 1090,
    1093, 1096, 1105, 1108, 1110, 1120, 1153, 1156, 1168, 1280, 1282, 1285, 1288, 1297, 1300, 1312,
    1345, 1348, 1360, 1377, 1408, 1537, 1540, 1552, 1574, 1600, 1602, 1668, 2048, 2050, 2053, 2056,
    2058, 2065, 2068, 2080, 2085, 2113, 2116, 2128, 2136, 2176, 2208, 2218, 2305, 2308, 2320, 2368,
    2433, 2441, 2560, 2592, 2600, 2710, 2720, 4097, 4100, 4102, 4105, 4112, 4114, 4117, 4120, 4129,
    4132, 4160, 4162, 4165, 4168, 4177, 4180, 4192, 4202, 4225, 4228, 4240, 4352, 4354, 4357, 4360,
    4369, 4372, 4384, 4417, 4420, 4432, 4480, 4500, 4502, 4609, 4612, 4614, 4624, 4672, 4704, 5120,
    5122, 5125, 5128, 5137, 5140, 5152, 5185, 5188, 5193, 5200, 5220, 5248, 5377, 5380, 5392, 5440,
    5632, 5652, 5705, 6145, 6148, 6160, 6162, 6208, 6228, 6278, 6400, 6405, 6502, 6737, 6825, 8192,
    8194, 8197, 8200, 8202, 8209, 8212, 8224, 8257, 8260, 8272, 8320, 8352, 8449, 8452, 8464, 8512,
    8520, 8549, 8704, 8738, 8832, 8872, 9217, 9220, 9232, 9257, 9280, 9472, 9537, 9554, 9625, 9729,
    9754, 9894, 10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882,
    10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448,
    16450, 16453, 16456, 16458, 16465, 16468, 16480, 16485, 16513, 16516, 16528, 16640, 16642,
    16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773, 16802, 16897, 16900,
    16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
    17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988,
    18000, 18433, 18436, 18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793,
    18948, 20480, 20482, 20485, 20488, 20497, 20500, 20512, 20520, 20545, 20548, 20560, 20608,
    20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162, 21505, 21508, 21520, 21537,
    21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528, 22548,
    22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674,
    24710, 24745, 24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625,
    26730, 26917, 27142, 27220, 27234, 32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810,
    32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025, 33028, 33030, 33040, 33088, 33105,
    33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810, 33813, 33856,
    33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946,
    34984, 35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996,
    37120, 37154, 37220, 37462, 37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290,
    38465, 38993, 39078, 39241, 39445, 39520, 40960, 40962, 40968, 40970, 40992, 41002, 41120,
    41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048, 42133, 42597, 42648,
    43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
];

const IQ3_XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

const IQ1_S_GRID_HEX: &str = concat!(
    "00000200050008000a00110015002000220028002a00450051005400560065008000820088008a009500a000a200a800",
    "aa000401050111011401160119011a012501410146014901520155015a0161016401660168018501910194019601a501",
    "0002020208020a0215022002220228022a02450251025902640269028002820288028a02910295029902a002a202a802",
    "aa0211041404160425044104490455045a046404650491049904a5040105040505050605150518051a05290540054505",
    "4a0550055105540555055605590560056205650568056a0581059105950598059a05a105a405a505a605a90514061906",
    "410644065006520655065806600661066606690685069106940699060008020808080a0815082008220828082a084508",
    "5108560865088008820888088a089508a008a208a808aa08050911091409190924092509410950095109550961096409",
    "69099109940996099909a509000a020a080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a",
    "8a0a950aa00aa20aa80aaa0a101011101410191024102510411044105010551058106110641065106910911094109610",
    "a110a5100111041106110911101112111511181121112411291145114a11501151115211541155115611591160116511",
    "841192119511a111a41111121412161225124012461249125212551258125a12641266128512911294129612a5120114",
    "0614091414141514181419142114261441144514461448144a1451145414551456145914621465146814841489149014",
    "94149514981499149a14a114a414a514a914021505150a151115141515151615191520152215251528152a1541154415",
    "451546155115521554155515561559155a1561156415651566156915801582158415851588158a159015911594159515",
    "961599159a15a015a215a51501160416051606161516161618161a1621162616401642164416451648164a1651165516",
    "561658165916611664166516681669166a1686168a1692169516a416a916111816182518411844184618491850185518",
    "58185a1860186118641866186918851891189418a5181019121915191a19211925194219441945194819511954195519",
    "561959195a19601965196a1989199119921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a",
    "581a611a661a691a851a911a961a9a1a0020022008200a20152020202220252028202a20452051205920612065208020",
    "822088208a209520a020a220a520a820aa2005211121142119212521422144214921552158215a216121642165216621",
    "8521902196219921a521012208220a22112215222022222228222a2245225122562259226522812288228a2291229522",
    "a022a222a822aa220524142416241924252444244524462449245224552458245a2466248524912494249924a124a524",
    "09251525212529254025452548255125542555255925622565256825892590259425952598259a25a125a425a625a925",
    "052610261226192625264126492655266026612669268426862690269a260028022808280a2815282028222828282a28",
    "45285128542865288028822888288a28a028a228a828aa28092911291429192925294629492952295529612964296629",
    "69298529902996299929a429a529002a022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a",
    "952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440664094409940a140a640004101410441",
    "0641094112411541164118411a41214126412941454148414a41514154415541564159415a41654168416a4181418441",
    "8641904192419541a041a141a241054211421442164225424142524255425a426442694289429442a542014415441944",
    "2944454448444a44514454445544564461446244654468446a44814486448944904492449544a044a144a94401450245",
    "05450a4511451445154516451945204525452a4541454445454546454945504551455445554556455845594561456445",
    "6545664569458245844585458845914594459545964599459a45a545a845aa450146054609461446154618461a462146",
    "244629464046424645464846504651465246554656465946624665466846814685468a4694469546a146a446a6460548",
    "114815481a48254842484948504855485848614864486648694885489148944896489948a5480149054906490a491049",
    "144915491849214924492649404945494a4951495249544955495649594960496249654966496a498649894992499549",
    "96499849a149a449a649a949164a444a464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550",
    "1a5021502450295040504550485051505450555056505950655068508650895095509850a050a150a650a95005510851",
    "09510a5111511451155116511851195120512551265128512a5141514451455146514951505151515251545155515651",
    "585159515a51615164516551665169518251855191519451955196519951a051a551aa5101520652125215521a522152",
    "2452425245524a525152545255525652595262526552855290529252955299529a52a452045405541154145415541654",
    "185419542154255428542a54415444544554465449544a5450545154545455545654585459545a546154625464546554",
    "66546954805488548a5491549454955496549954a154a454a554aa540155025504550555065509551055115512551455",
    "1555165519551a5521552455255526552955405541554255445545554655485549555055515552555455555556555855",
    "59555a5560556155645565556655685569556a5581558455855589558a559055915594559555965598559955a155a455",
    "a555a655a955005601560256045606560856095611561456155618561956205621562256245625562656285629564156",
    "45564656485649564a56505651565256545655565656585659565a566156645665566956825685568656885689568a56",
    "915695569a56a256a556a656a856a956045805580658095810581558185821582a58455848584a585158545855585658",
    "585859586058625864586558825889589058925895589858a158a9580159025905590a59115914591559165919592559",
    "41594459455946594959505951595259545955595659585959595a596159645965596659695981598559895991599459",
    "9559965998599959a559045a085a155a1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a",
    "6a5a815a8a5a925a955a965a985a9a5aa15a05601460166019602560446050605560566058605a606160646066606960",
    "81609660a5600161046106610961126115612161226126612961456149615161556156615961656166616a6184618a61",
    "92619561a161a661a9611162166219624062416246625562566258626062856291629662a56211641264156416641a64",
    "21642664296440644264456448644a64516454645564566459645a646064626465648464856489649064926494649564",
    "966498649a64a164a464a964056508650a65116515651665196544654565466549655065516554655565566559656165",
    "6465656566656965866589658a6591659565966599659a65a265a565a665a86502660966156620662666286629664066",
    "456648664a66516654665566566658665a666066656668668066826685668a669466966698669966a066a466a666aa66",
    "1668196825684168526855685a6861686968856891689868a66801690469106915692169246926692969406941694569",
    "4669486951695469556956695969606965696a69826984698a699569a169a469a569a969116a166a186a416a446a496a",
    "506a556a586a5a6a646a656a696a866a946a986a9a6aa66a0080028008800a802080228028802a804580508051805480",
    "5680598065808080828088808a809580a080a280a880aa80058111811481168119812581418144814981508152815581",
    "56815881598164816681698185818981948196819981a5810082028208820a8215822082228228822a82518254825982",
    "65828082828288828a829582a082a282a882aa821484198441844484518455845a846184648469849484998401850985",
    "128515851a85268529854085418545854885518554855585568559855a856585668568856a8581858485868589859085",
    "928595859885a68511861686198625864186448649864a865086558659865a86618666866a86858691869a86a4860088",
    "028808880a8815882088228828882a8841884588518854885988658869888088828888888a889588a088a288a888aa88",
    "05890689118914891689258941894489468949895089528955895a8961896489858996899989a589008a028a088a0a8a",
    "158a208a228a288a2a8a458a518a548a568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590",
    "419046904990559058905a9069906a9085909190949096909990a59001910491069109911091159118911a9121912491",
    "26912991409145915091519154915591569159916291659184918691929195919891a191a491a691a991059211921492",
    "19922592449246924992509252925592589266926992859294929692a992019404940694109415941894269440944a94",
    "5194549455945694589459946094619462946594849486949294949495949894a194a9940095059508950a9510951195",
    "14951595169519952195259529952a9541954495459546954995509551955295549555955695589559955a9561956495",
    "6595669569958195859588959195929594959595969599959a95a095a295a595a895aa95019604961096159619962096",
    "2696299645964896499651965296559656965996659668968296849689968a96929694969596a496a696a99605981698",
    "199825984198469850985298559856985a98649865988598919896989998a59804990699099910991299159918991a99",
    "209921992499269940994299459948994a99519954995599569959996299659966996a99819984999099929995999a99",
    "a199a699059a159a259a449a469a499a509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0",
    "28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa005a109a111a114a116a119a11aa146a1",
    "49a151a155a158a15aa161a164a185a190a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2",
    "65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a455a458a45aa461a465a466a468a469a4",
    "85a406a509a510a512a515a518a526a529a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5",
    "95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a662a686a690a695a696a699a6a1a6a4a6",
    "a6a600a802a808a80aa820a822a828a82aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9",
    "19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa20aa22aa28aa2aaa51aa54aa56aa80aa",
    "82aa88aa8aaa95aaa0aaa2aaa8aaaaaa",
);

const IQ3_S_GRID_HEX: &str = concat!(
    "000001000200050007001000110012001400160020002100250033004000420045004700510053006000620071007400",
    "770000010101020104011001110115012001230127013101350144016101650172010002010205020702100213021602",
    "210225023002340242024502470251025302700273020303110315032003220331033303360344035003520367037103",
    "750300041304170421042404320440044304510470040205040520052205260533054105450547056605730506061106",
    "130631065206710600070207040720072207260733075007540700100110021004101010111013101510171020102210",
    "311034103610541056106110721000110111031106111011141121113011331141115011521170117611001212121512",
    "171220122412321240124312551260127212011304130713101313132113271330133413411362137013031405141214",
    "141431143314421446145014541401151015131521153015321551152016241627164416461601170317101712172117",
    "351741176217701700200120032005200720102012201420162021202320272030203220412043204520502052206720",
    "702073207520002102211021132117212221252131213421422151210122042207222122232230223722412253225722",
    "712274220023022305231123222324233123332342235023662301240724202423243224352441247224752404251125",
    "222537254025532570250026022607262126552661260527112726273027432750270230113013301530173022303130",
    "333035304230443047305130633071300131033105311431213123314031603172317631003212322032323234325032",
    "013310331433213323332733303341334333473355337333033411341634223431345234603464340135103512352535",
    "323544355635733516364136013703372037223735370040044012402040244027403240414050407040024107411141",
    "134122413041354143415141554101420342104215422142334240425742624270420443114313432043224331433543",
    "004402442444374440447144054507452145624513463446604610471547304743475147025010501450225040504450",
    "475052506650745001510351055112512151325172510052115223523052365253520253075310532753445351536553",
    "735301540454205432544654125526555155535542560257045722571160136015603160336060600061206127616461",
    "126234624262556262627062006314632163406325644364626400650365346560650566406611671367007004700770",
    "207022703670407054706270027111712471437145710172047210721672217230725172027332733573537301740574",
    "13742074507422754275027631760077",
);

const IQ2_S_GRID_HEX: &str = concat!(
    "00000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100640066006900800082008500880091009400a000",
    "a500aa0001010401060109011001120115011801210124014001420145014801510154015601590160016501680181018401900192019501a101a40100020202",
    "050208021102140220022a02410244024602490250025502800285028a029402a202010404040604090410041204150418042104240426042904400442044504",
    "48044a0451045404560459046004620465048104840486048904900495049804a104a40400050205050508050a05110514051605190520052505280541054405",
    "46054905500552055505580561056405800582058505880591059405a00501060406060609061006150640064506480651065406600681068406900600080208",
    "050808081108140816081908200825082a084108440846084908500852085508580861086408800885089408aa08010904091009120915091809210940094509",
    "480951095409600981099009000a110a140a220a280a2a0a500a990a011004100610091010101210151018102110241026104010421045104810511054105610",
    "59106010621065106810811084108610901095109810a110a41000110211051108110a1111111411161119112011221125112811411144114611491150115211",
    "55115811611164118011821185118811911194110112041209121012151221122412401245125112541281128412901200140214051408141114141416141914",
    "2014251428144114441446144914501452145514581461146414801482148514881491149414a014011504150615091510151215151518152115241540154215",
    "451548155115541560158115841590150016051608161116141620164116441650168016aa160118041806180918101815181818211840184218451848185118",
    "541860188118841800190219051908191119141920194119441950196919a219041a101a401a561a00200220052008201120142016201920202025202a204120",
    "4420502052205520642080208a209420aa2001210421102112211521212140214221452151215421602181218421902100220a22222228222a22442250228822",
    "8a22a822012404240624092410241524182421242424402442244524482451245424602481248424902400250525082511251425202541254425502566258025",
    "0126042610264026592600280528112814284128442850288a28aa2801290429102995290a2a222a642a882a8a2a014004400640094010401240154018401a40",
    "21402440264040404240454048404a40514054405640594060406240654081408440904095409840a140a4400041024105410841114114411641194120412241",
    "25414141444146414941504152415541584161416441804182418541884191419441a04101420442104212421542184224424042454248425142544260428142",
    "844200440244054408440a44114414441644194420442244254428444144444446444944504452445544584461446444804482448544884491449444a0440145",
    "04450645094510451245154518452145244540454245454548455145544560456a4581458445904500460246054608461146144620464146444650468046a546",
    "014804480948104812481548184821482448404842484548484851485448604884489048004902490549084911491449204941494449504980499649014a044a",
    "104a404a005002500550085011501450165019502050225025502850415044504650495050505250555058506150645080508250855088509150945001510451",
    "06510951105112511551185121512451405142514551485151515451605181518451905100520552085211521452205241524452505269528052015404540654",
    "09541054125415541854215424544054425445544854515454546054815484549054005502550555085511551455205541554455505580550156045610562656",
    "405600580258055808581158145820584158445850585a5880580159045910594059005a195a855aa85a01600460066010601260156018602160246040604560",
    "4860516054606060846090600061026105610861116114612061416144615061806199610462106240625662a162006405640864116414642064416444645064",
    "806401650465106540654a6568659265006694660168046810686568986800692a69426aa16a0080028005800880118014801980208025804180448050805280",
    "5580588061808080858091809480018104810981108112811581188121812481408142814581488151815481818184819081a981008205820a82118214824182",
    "44825082018404840684098410841284158418842184408442844584488451845484608481848484908400850285058508851185148520854185448550858085",
    "8a85018604861086298640860088058811881488418844885088a2880189048940896589228a588a5a8a828aa28a019004900990109012901590189024904090",
    "42904590489051905490609081908490909000910591119114914191449150915a910192049210924092a6920094029405940894119414942094419444945094",
    "8094969401950495109540959895a19500964696649601980498109826984098a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0",
    "40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa",
);

fn dequantize_iq2_xxs(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 66;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ2_XXS")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..66];
        for ib32 in 0..8 {
            let q = &qs[ib32 * 8..ib32 * 8 + 8];
            let aux32 = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.25;
            for lane in 0..4 {
                let grid_index = q[lane] as usize;
                let signs = iq2_xxs_signs(((aux32 >> (7 * lane)) & 0x7f) as u8);
                for j in 0..8 {
                    let value =
                        IQ2_XXS_VALUES[((IQ2_XXS_GRID[grid_index] >> (2 * j)) & 0x03) as usize];
                    let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out.push(db * f32::from(value) * sign);
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_iq2_xs(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 74;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ2_XS")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..66];
        let scales = &block[66..74];
        for ib32 in 0..8 {
            let scale = scales[ib32];
            let db = [
                d * (0.5 + f32::from(scale & 0x0f)) * 0.25,
                d * (0.5 + f32::from(scale >> 4)) * 0.25,
            ];
            for lane in 0..4 {
                let q_offset = (4 * ib32 + lane) * 2;
                let q = u16::from_le_bytes([qs[q_offset], qs[q_offset + 1]]);
                let grid_index = (q & 0x01ff) as usize;
                let signs = iq2_xxs_signs((q >> 9) as u8);
                let dl = db[lane / 2];
                for j in 0..8 {
                    let value =
                        IQ2_XXS_VALUES[((IQ2_XS_GRID[grid_index] >> (2 * j)) & 0x03) as usize];
                    let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out.push(dl * f32::from(value) * sign);
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_iq3_xxs(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 98;
    const QS_BYTES: usize = 64;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ3_XXS")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..2 + QS_BYTES];
        let scales_and_signs = &block[2 + QS_BYTES..BLOCK_BYTES];
        for ib32 in 0..8 {
            let aux_offset = 4 * ib32;
            let aux32 = u32::from_le_bytes([
                scales_and_signs[aux_offset],
                scales_and_signs[aux_offset + 1],
                scales_and_signs[aux_offset + 2],
                scales_and_signs[aux_offset + 3],
            ]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for lane in 0..4 {
                let signs = iq2_xxs_signs(((aux32 >> (7 * lane)) & 0x7f) as u8);
                let grid1 = IQ3_XXS_GRID[qs[8 * ib32 + 2 * lane] as usize];
                let grid2 = IQ3_XXS_GRID[qs[8 * ib32 + 2 * lane + 1] as usize];
                for j in 0..4 {
                    let value = ((grid1 >> (8 * j)) & 0xff) as u8;
                    let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out.push(db * f32::from(value) * sign);
                }
                for j in 0..4 {
                    let value = ((grid2 >> (8 * j)) & 0xff) as u8;
                    let sign = if signs & (1 << (j + 4)) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * f32::from(value) * sign);
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_iq1_s(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 50;
    const QS_BYTES: usize = 32;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ1_S")?;
    let mut out = Vec::with_capacity(element_count);
    let grid = iq1_s_grid();
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..2 + QS_BYTES];
        let qh = &block[2 + QS_BYTES..BLOCK_BYTES];
        for ib32 in 0..8 {
            let qh_offset = 2 * ib32;
            let qh_word = u16::from_le_bytes([qh[qh_offset], qh[qh_offset + 1]]);
            let dl = d * f32::from(2 * ((qh_word >> 12) & 7) + 1);
            let delta = if qh_word & 0x8000 != 0 { -0.125 } else { 0.125 };
            for lane in 0..4 {
                let grid_index =
                    qs[4 * ib32 + lane] as usize | ((((qh_word >> (3 * lane)) & 7) as usize) << 8);
                let grid_offset = grid_index * 8;
                for j in 0..8 {
                    out.push(dl * (f32::from(grid[grid_offset + j]) + delta));
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_iq2_s(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 82;
    const QS_BYTES: usize = 64;
    const QH_BYTES: usize = 8;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ2_S")?;
    let mut out = Vec::with_capacity(element_count);
    let grid = iq2_s_grid();
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs_all = &block[2..2 + QS_BYTES];
        let qs = &qs_all[..QS_BYTES / 2];
        let signs = &qs_all[QS_BYTES / 2..];
        let qh = &block[2 + QS_BYTES..2 + QS_BYTES + QH_BYTES];
        let scales = &block[2 + QS_BYTES + QH_BYTES..BLOCK_BYTES];
        for ib32 in 0..8 {
            let db = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
            let qh_byte = qh[ib32];
            for lane in 0..4 {
                let grid_index = qs[4 * ib32 + lane] as usize
                    | ((((qh_byte >> (2 * lane)) & 0x03) as usize) << 8);
                let signs_byte = signs[4 * ib32 + lane];
                let grid_offset = grid_index * 8;
                for j in 0..8 {
                    let sign = if signs_byte & (1 << j) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * f32::from(grid[grid_offset + j]) * sign);
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_iq1_m(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 56;
    const QS_BYTES: usize = 32;
    const QH_BYTES: usize = 16;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ1_M")?;
    let mut out = Vec::with_capacity(element_count);
    let grid = iq1_s_grid();
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let qs = &block[..QS_BYTES];
        let qh = &block[QS_BYTES..QS_BYTES + QH_BYTES];
        let scales = &block[QS_BYTES + QH_BYTES..BLOCK_BYTES];
        let sc = [
            u16::from_le_bytes([scales[0], scales[1]]),
            u16::from_le_bytes([scales[2], scales[3]]),
            u16::from_le_bytes([scales[4], scales[5]]),
            u16::from_le_bytes([scales[6], scales[7]]),
        ];
        let d_bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16_to_f32(d_bits);

        for ib32 in 0..8 {
            let scale_word = sc[ib32 / 2];
            let scale_shift = 6 * (ib32 % 2);
            let dl = [
                d * f32::from(2 * ((scale_word >> scale_shift) & 0x07) + 1),
                d * f32::from(2 * ((scale_word >> (scale_shift + 3)) & 0x07) + 1),
            ];
            for lane in 0..4 {
                let qh_byte = qh[2 * ib32 + lane / 2];
                let qh_shift = 4 * (lane % 2);
                let grid_index =
                    qs[4 * ib32 + lane] as usize | ((((qh_byte >> qh_shift) & 0x07) as usize) << 8);
                let delta = if qh_byte & (0x08u8 << qh_shift) != 0 {
                    -0.125
                } else {
                    0.125
                };
                let grid_offset = grid_index * 8;
                for j in 0..8 {
                    out.push(dl[lane / 2] * (f32::from(grid[grid_offset + j]) + delta));
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_iq3_s(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 110;
    const QS_BYTES: usize = 64;
    const QH_BYTES: usize = 8;
    const SIGNS_BYTES: usize = 32;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "IQ3_S")?;
    let mut out = Vec::with_capacity(element_count);
    let grid = iq3_s_grid();
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..2 + QS_BYTES];
        let qh = &block[2 + QS_BYTES..2 + QS_BYTES + QH_BYTES];
        let signs = &block[2 + QS_BYTES + QH_BYTES..2 + QS_BYTES + QH_BYTES + SIGNS_BYTES];
        let scales = &block[2 + QS_BYTES + QH_BYTES + SIGNS_BYTES..BLOCK_BYTES];
        for ib32 in 0..8 {
            let scale_byte = scales[ib32 / 2];
            let scale = if ib32 % 2 == 0 {
                scale_byte & 0x0f
            } else {
                scale_byte >> 4
            };
            let db = d * f32::from(1 + 2 * scale);
            let qh_byte = qh[ib32];
            for lane in 0..4 {
                let signs_byte = signs[4 * ib32 + lane];
                let grid1_index = qs[8 * ib32 + 2 * lane] as usize
                    | (((qh_byte >> (2 * lane)) as usize & 1) << 8);
                let grid2_index = qs[8 * ib32 + 2 * lane + 1] as usize
                    | ((((qh_byte >> (2 * lane + 1)) as usize) & 1) << 8);
                let grid1_offset = grid1_index * 4;
                let grid2_offset = grid2_index * 4;
                for j in 0..4 {
                    let sign = if signs_byte & (1 << j) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * f32::from(grid[grid1_offset + j]) * sign);
                }
                for j in 0..4 {
                    let sign = if signs_byte & (1 << (j + 4)) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * f32::from(grid[grid2_offset + j]) * sign);
                }
            }
        }
    }
    Ok(out)
}

fn iq1_s_grid() -> &'static [i8] {
    static GRID: OnceLock<Vec<i8>> = OnceLock::new();
    GRID.get_or_init(decode_iq1_s_grid).as_slice()
}

fn iq3_s_grid() -> &'static [u8] {
    static GRID: OnceLock<Vec<u8>> = OnceLock::new();
    GRID.get_or_init(decode_iq3_s_grid).as_slice()
}

fn iq2_s_grid() -> &'static [u8] {
    static GRID: OnceLock<Vec<u8>> = OnceLock::new();
    GRID.get_or_init(decode_iq2_s_grid).as_slice()
}

fn decode_iq1_s_grid() -> Vec<i8> {
    let hex = IQ1_S_GRID_HEX.as_bytes();
    debug_assert_eq!(hex.len(), 8192);
    let mut grid = Vec::with_capacity(2048 * 8);
    for pair in hex.chunks_exact(2) {
        let byte = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
        for shift in [0, 2, 4, 6] {
            let value = match (byte >> shift) & 0x03 {
                0 => -1,
                1 => 0,
                2 => 1,
                other => panic!("invalid IQ1_S grid code {other}"),
            };
            grid.push(value);
        }
    }
    debug_assert_eq!(grid.len(), 2048 * 8);
    grid
}

fn decode_iq3_s_grid() -> Vec<u8> {
    let hex = IQ3_S_GRID_HEX.as_bytes();
    debug_assert_eq!(hex.len(), 2048);
    let mut grid = Vec::with_capacity(512 * 4);
    for pair in hex.chunks_exact(2) {
        let byte = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
        for shift in [0, 4] {
            grid.push(1 + 2 * ((byte >> shift) & 0x07));
        }
    }
    debug_assert_eq!(grid.len(), 512 * 4);
    grid
}

fn decode_iq2_s_grid() -> Vec<u8> {
    let hex = IQ2_S_GRID_HEX.as_bytes();
    debug_assert_eq!(hex.len(), 4096);
    let mut grid = Vec::with_capacity(1024 * 8);
    const VALUES: [u8; 3] = [0x08, 0x19, 0x2b];
    for pair in hex.chunks_exact(2) {
        let byte = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
        for shift in [0, 2, 4, 6] {
            let code = ((byte >> shift) & 0x03) as usize;
            grid.push(VALUES[code]);
        }
    }
    debug_assert_eq!(grid.len(), 1024 * 8);
    grid
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        other => panic!("invalid hex byte {other}"),
    }
}

fn iq2_xxs_signs(index: u8) -> u8 {
    index | if index.count_ones() % 2 == 1 { 0x80 } else { 0 }
}

const TQ_POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

fn dequantize_tq1_0(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 54;
    const QS_BYTES: usize = 48;
    const QH_BYTES: usize = 4;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "TQ1_0")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let qs = &block[0..QS_BYTES];
        let qh = &block[QS_BYTES..QS_BYTES + QH_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[52], block[53]]));

        for j in (0..QS_BYTES - QS_BYTES % 32).step_by(32) {
            for pow in TQ_POW3.iter().take(5) {
                for m in 0..32 {
                    out.push(dequantize_tq_value(qs[j + m], *pow, d));
                }
            }
        }
        for j in (QS_BYTES - QS_BYTES % 32..QS_BYTES).step_by(16) {
            for pow in TQ_POW3.iter().take(5) {
                for m in 0..16 {
                    out.push(dequantize_tq_value(qs[j + m], *pow, d));
                }
            }
        }
        for pow in TQ_POW3.iter().take(4) {
            for value in qh {
                out.push(dequantize_tq_value(*value, *pow, d));
            }
        }
    }
    Ok(out)
}

fn dequantize_tq2_0(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 66;
    const QS_BYTES: usize = 64;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "TQ2_0")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let qs = &block[0..QS_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));
        for j in (0..QS_BYTES).step_by(32) {
            for shift in [0, 2, 4, 6] {
                for m in 0..32 {
                    let quant = ((qs[j + m] >> shift) & 0x03) as i8;
                    out.push(f32::from(quant - 1) * d);
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_tq_value(value: u8, pow: u8, d: f32) -> f32 {
    let q = value.wrapping_mul(pow);
    let xi = ((u16::from(q) * 3) >> 8) as i16;
    f32::from(xi - 1) * d
}

fn dequantize_q5_0(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 22;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q5_0")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..];
        for idx in 0..BLOCK_ELEMENTS {
            let packed = qs[idx % 16];
            let low = if idx < 16 { packed & 0x0f } else { packed >> 4 };
            let high = ((qh >> idx) & 1) as u8;
            let quant = low | (high << 4);
            out.push(d * (i32::from(quant) - 16) as f32);
        }
    }
    Ok(out)
}

fn dequantize_q5_1(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 24;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q5_1")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..];
        for idx in 0..BLOCK_ELEMENTS {
            let packed = qs[idx % 16];
            let low = if idx < 16 { packed & 0x0f } else { packed >> 4 };
            let high = ((qh >> idx) & 1) as u8;
            let quant = low | (high << 4);
            out.push(d * f32::from(quant) + m);
        }
    }
    Ok(out)
}

fn dequantize_q2_k(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 84;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q2_K")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut scale_idx = 0usize;
        let mut q_offset = 0usize;
        for _ in 0..2 {
            let q = &qs[q_offset..q_offset + 32];
            let mut shift = 0u8;
            for _ in 0..4 {
                let sc = scales[scale_idx];
                scale_idx += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = dmin * f32::from(sc >> 4);
                for byte in &q[0..16] {
                    out.push(dl * f32::from((byte >> shift) & 0x03) - ml);
                }

                let sc = scales[scale_idx];
                scale_idx += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = dmin * f32::from(sc >> 4);
                for byte in &q[16..32] {
                    out.push(dl * f32::from((byte >> shift) & 0x03) - ml);
                }

                shift += 2;
            }
            q_offset += 32;
        }
    }
    Ok(out)
}

fn dequantize_q3_k(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 110;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q3_K")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales = &block[96..108];
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        let mut scale_idx = 0usize;
        let mut q_offset = 0usize;
        let mut mask = 1u8;
        for _ in 0..2 {
            let q = &qs[q_offset..q_offset + 32];
            let mut shift = 0u8;
            for _ in 0..4 {
                let dl = d * f32::from(q3_k_scale(scale_idx, scales));
                scale_idx += 1;
                for l in 0..16 {
                    let low = ((q[l] >> shift) & 0x03) as i8;
                    let quant = low - if hmask[l] & mask != 0 { 0 } else { 4 };
                    out.push(dl * f32::from(quant));
                }

                let dl = d * f32::from(q3_k_scale(scale_idx, scales));
                scale_idx += 1;
                for l in 0..16 {
                    let low = ((q[l + 16] >> shift) & 0x03) as i8;
                    let quant = low - if hmask[l + 16] & mask != 0 { 0 } else { 4 };
                    out.push(dl * f32::from(quant));
                }

                shift += 2;
                mask <<= 1;
            }
            q_offset += 32;
        }
    }
    Ok(out)
}

fn dequantize_q4_k(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 144;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q4_K")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut scale_idx = 0usize;
        for chunk in qs.chunks_exact(32) {
            let (sc1, min1) = q4_k_scale_min(scale_idx, scales);
            let (sc2, min2) = q4_k_scale_min(scale_idx + 1, scales);
            let d1 = d * f32::from(sc1);
            let d2 = d * f32::from(sc2);
            let m1 = dmin * f32::from(min1);
            let m2 = dmin * f32::from(min2);
            for byte in chunk {
                out.push(d1 * f32::from(byte & 0x0f) - m1);
            }
            for byte in chunk {
                out.push(d2 * f32::from(byte >> 4) - m2);
            }
            scale_idx += 2;
        }
    }
    Ok(out)
}

fn dequantize_q5_k(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 176;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q5_K")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        for group64 in 0..4 {
            let scale_idx = group64 * 2;
            let (sc1, min1) = q4_k_scale_min(scale_idx, scales);
            let (sc2, min2) = q4_k_scale_min(scale_idx + 1, scales);
            let d1 = d * f32::from(sc1);
            let d2 = d * f32::from(sc2);
            let m1 = dmin * f32::from(min1);
            let m2 = dmin * f32::from(min2);
            let ql = &qs[group64 * 32..group64 * 32 + 32];
            let u1 = 1u8 << (2 * group64);
            let u2 = 2u8 << (2 * group64);
            for (idx, byte) in ql.iter().enumerate() {
                let high = if qh[idx] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * f32::from((byte & 0x0f) + high) - m1);
            }
            for (idx, byte) in ql.iter().enumerate() {
                let high = if qh[idx] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * f32::from((byte >> 4) + high) - m2);
            }
        }
    }
    Ok(out)
}

fn dequantize_q6_k(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 210;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q6_K")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        let mut ql_offset = 0usize;
        let mut qh_offset = 0usize;
        let mut scale_offset = 0usize;
        for _ in 0..2 {
            let base = out.len();
            out.resize(base + 128, 0.0);
            for l in 0..32 {
                let is = l / 16;
                let qh_byte = qh[qh_offset + l];
                let q1 = (((ql[ql_offset + l] & 0x0f) | ((qh_byte & 0x03) << 4)) as i8) - 32;
                let q2 =
                    (((ql[ql_offset + l + 32] & 0x0f) | (((qh_byte >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[ql_offset + l] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 =
                    (((ql[ql_offset + l + 32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) as i8) - 32;
                out[base + l] = d * f32::from(scales[scale_offset + is] as i8) * f32::from(q1);
                out[base + l + 32] =
                    d * f32::from(scales[scale_offset + is + 2] as i8) * f32::from(q2);
                out[base + l + 64] =
                    d * f32::from(scales[scale_offset + is + 4] as i8) * f32::from(q3);
                out[base + l + 96] =
                    d * f32::from(scales[scale_offset + is + 6] as i8) * f32::from(q4);
            }
            ql_offset += 64;
            qh_offset += 32;
            scale_offset += 8;
        }
    }
    Ok(out)
}

fn dequantize_q8_k(bytes: &[u8], element_count: usize) -> Result<Vec<f32>> {
    const BLOCK_ELEMENTS: usize = 256;
    const BLOCK_BYTES: usize = 292;
    require_quantized_len(bytes, element_count, BLOCK_ELEMENTS, BLOCK_BYTES, "Q8_K")?;
    let mut out = Vec::with_capacity(element_count);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for value in &block[4..260] {
            out.push(d * f32::from(*value as i8));
        }
    }
    Ok(out)
}

fn q4_k_scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
    debug_assert_eq!(scales.len(), 12);
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        let scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        let min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        (scale, min)
    }
}

fn q3_k_scale(index: usize, scales: &[u8]) -> i8 {
    debug_assert_eq!(scales.len(), 12);
    let low = if index < 8 {
        scales[index] & 0x0f
    } else {
        scales[index - 8] >> 4
    };
    let high = (scales[8 + index % 4] >> (2 * (index / 4))) & 0x03;
    ((low | (high << 4)) as i8) - 32
}

fn require_quantized_len(
    bytes: &[u8],
    element_count: usize,
    block_elements: usize,
    block_bytes: usize,
    label: &str,
) -> Result<()> {
    if !element_count.is_multiple_of(block_elements) {
        bail!(
            "{label} tensor element count {element_count} is not divisible by block size {block_elements}"
        );
    }
    let expected = element_count
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| anyhow!("{label} tensor byte length overflows usize"))?;
    if bytes.len() != expected {
        bail!(
            "{label} tensor byte length {} does not match expected {expected}",
            bytes.len()
        );
    }
    Ok(())
}

fn f16_to_f32(raw: u16) -> f32 {
    let sign = (u32::from(raw & 0x8000)) << 16;
    let exp = (raw >> 10) & 0x1f;
    let frac = u32::from(raw & 0x03ff);
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14i32;
                while frac & 0x0400 == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                let exp_bits = u32::try_from(exp + 127).expect("subnormal exponent") << 23;
                sign | exp_bits | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp_bits = (u32::from(exp) + 112) << 23;
            sign | exp_bits | (frac << 13)
        }
    };
    f32::from_bits(bits)
}
