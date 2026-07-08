use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use hi_local_core::model::{
    DEFAULT_MAX_OUTPUT_TOKENS, ModelFamily, ModelInfo, TokenizerInfo, WeightShard,
};
use memmap2::Mmap;
use serde::Serialize;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    mmap: Mmap,
    version: u32,
    alignment: u64,
    data_start: u64,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory-mapping {}", path.display()))?;
        parse_mmap(path.to_path_buf(), mmap)
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

    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let info = self.tensors.iter().find(|tensor| tensor.name == name)?;
        self.tensor_view(info).ok()
    }

    pub fn tensor_view<'a>(&'a self, info: &'a TensorInfo) -> Result<TensorView<'a>> {
        let byte_len = info.byte_len()?;
        let start = checked_add(self.data_start, info.offset, "tensor data offset")?;
        let end = checked_add(start, byte_len, "tensor data length")?;
        let start = usize::try_from(start).context("tensor data offset does not fit usize")?;
        let end = usize::try_from(end).context("tensor data length does not fit usize")?;
        let bytes = self
            .mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("tensor {} points outside GGUF data section", info.name))?;
        Ok(TensorView { info, bytes })
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
    fn from_raw(raw: u32) -> Result<Self> {
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
                if element_count % block_elements != 0 {
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

fn unsupported_tensor_type_error(raw: u32, tensor_name: Option<&str>) -> anyhow::Error {
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

pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
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

#[derive(Clone, Debug)]
pub struct GgufTokenizer {
    model: Option<String>,
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), usize>,
    scores: Option<Vec<f32>>,
    token_types: Option<Vec<i32>>,
    special_ids: BTreeSet<u32>,
    special_tokens: Vec<(String, u32)>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unknown_token_id: Option<u32>,
    padding_token_id: Option<u32>,
    add_bos_token: bool,
    add_eos_token: bool,
}

impl GgufTokenizer {
    fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model = gguf
            .metadata_string("tokenizer.ggml.model")
            .map(ToString::to_string);
        let tokens = gguf
            .metadata_string_array("tokenizer.ggml.tokens")?
            .ok_or_else(|| anyhow!("GGUF metadata missing tokenizer.ggml.tokens"))?;
        if tokens.is_empty() {
            bail!("GGUF tokenizer has no tokens");
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).context("GGUF tokenizer token count exceeds u32")?;
            if token_to_id.insert(token.clone(), id).is_some() {
                bail!("GGUF tokenizer contains duplicate token {token:?}");
            }
        }

        let merges = gguf
            .metadata_string_array("tokenizer.ggml.merges")?
            .unwrap_or_default();
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| anyhow!("invalid GGUF tokenizer merge {merge:?}"))?;
            if left.is_empty() || right.is_empty() {
                bail!("invalid GGUF tokenizer merge {merge:?}");
            }
            merge_ranks.insert((left.to_string(), right.to_string()), rank);
        }

        let scores = gguf.metadata_f32_array("tokenizer.ggml.scores")?;
        if let Some(scores) = &scores {
            if scores.len() != tokens.len() {
                bail!(
                    "GGUF tokenizer score count {} does not match token count {}",
                    scores.len(),
                    tokens.len()
                );
            }
        }
        let token_types = gguf.metadata_i32_array("tokenizer.ggml.token_type")?;
        if let Some(token_types) = &token_types {
            if token_types.len() != tokens.len() {
                bail!(
                    "GGUF tokenizer token_type count {} does not match token count {}",
                    token_types.len(),
                    tokens.len()
                );
            }
        }

        let bos_token_id = gguf.metadata_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = gguf.metadata_u32("tokenizer.ggml.eos_token_id");
        let unknown_token_id = gguf.metadata_u32("tokenizer.ggml.unknown_token_id");
        let padding_token_id = gguf.metadata_u32("tokenizer.ggml.padding_token_id");
        let mut special_ids = BTreeSet::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.separator_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.mask_token_id",
        ] {
            if let Some(id) = gguf.metadata_u32(key) {
                ensure_token_id_in_range(id, tokens.len(), key)?;
                special_ids.insert(id);
            }
        }
        if let Some(token_types) = &token_types {
            for (idx, token_type) in token_types.iter().enumerate() {
                if matches!(*token_type, 2 | 3 | 4 | 5) {
                    special_ids.insert(idx as u32);
                }
            }
        }
        for (idx, token) in tokens.iter().enumerate() {
            if token.starts_with("<|") && token.ends_with("|>") {
                special_ids.insert(idx as u32);
            }
        }
        let mut special_tokens = special_ids
            .iter()
            .filter_map(|id| tokens.get(*id as usize).map(|token| (token.clone(), *id)))
            .filter(|(token, _)| !token.is_empty())
            .collect::<Vec<_>>();
        special_tokens.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()));

        Ok(Self {
            model,
            tokens,
            token_to_id,
            merge_ranks,
            scores,
            token_types,
            special_ids,
            special_tokens,
            bos_token_id,
            eos_token_id,
            unknown_token_id,
            padding_token_id,
            add_bos_token: gguf
                .metadata_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(false),
            add_eos_token: gguf
                .metadata_bool("tokenizer.ggml.add_eos_token")
                .unwrap_or(false),
        })
    }

    pub fn summary(&self) -> GgufTokenizerSummary {
        GgufTokenizerSummary {
            model: self.model.clone(),
            token_count: self.tokens.len(),
            merge_count: self.merge_ranks.len(),
            has_scores: self.scores.is_some(),
            has_token_types: self.token_types.is_some(),
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            unknown_token_id: self.unknown_token_id,
            padding_token_id: self.padding_token_id,
            add_bos_token: self.add_bos_token,
            add_eos_token: self.add_eos_token,
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if self.add_bos_token {
            if let Some(id) = self.bos_token_id {
                ids.push(id);
            }
        }

        ids.extend(self.encode_with_special_tokens(text)?);

        if self.add_eos_token {
            if let Some(id) = self.eos_token_id {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.decode_with_options(token_ids, true)
    }

    pub fn decode_with_options(&self, token_ids: &[u32], skip_special: bool) -> Result<String> {
        let mut text = String::new();
        for id in token_ids {
            ensure_token_id_in_range(*id, self.tokens.len(), "token id")?;
            if skip_special && self.special_ids.contains(id) {
                continue;
            }
            text.push_str(&self.tokens[*id as usize]);
        }

        if self.is_sentencepiece_unigram() {
            decode_sentencepiece_text(&text)
        } else if self.is_byte_level_bpe() {
            decode_byte_level_text(&text)
        } else {
            Ok(text.replace('\u{2581}', " "))
        }
    }

    fn is_byte_level_bpe(&self) -> bool {
        self.model
            .as_deref()
            .is_some_and(|model| matches!(model, "gpt2" | "bpe" | "qwen2"))
            || !self.merge_ranks.is_empty()
    }

    fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        if self.special_tokens.is_empty() {
            return self.encode_plain(text);
        }
        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            let Some((relative_idx, token_len, token_id)) = self.find_next_special(remaining)
            else {
                ids.extend(self.encode_plain(remaining)?);
                break;
            };
            if relative_idx > 0 {
                ids.extend(self.encode_plain(&remaining[..relative_idx])?);
            }
            ids.push(token_id);
            offset = offset
                .checked_add(relative_idx)
                .and_then(|value| value.checked_add(token_len))
                .context("special token encoding offset overflows usize")?;
        }
        Ok(ids)
    }

    fn find_next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (token, id) in &self.special_tokens {
            let Some(index) = text.find(token) else {
                continue;
            };
            let candidate = (index, token.len(), *id);
            if best.is_none_or(|current| {
                candidate.0 < current.0 || (candidate.0 == current.0 && candidate.1 > current.1)
            }) {
                best = Some(candidate);
            }
        }
        best
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if self.is_sentencepiece_unigram() {
            self.encode_sentencepiece_unigram(text)
        } else if self.is_byte_level_bpe() {
            self.encode_byte_level_bpe(text)
        } else {
            self.encode_greedy(text)
        }
    }

    fn is_sentencepiece_unigram(&self) -> bool {
        self.model
            .as_deref()
            .is_some_and(|model| matches!(model, "llama" | "spm"))
    }

    fn encode_sentencepiece_unigram(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = sentencepiece_normalize(text);
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Clone)]
        struct Node {
            score: f32,
            next: usize,
            ids: Vec<u32>,
        }

        let mut dp: Vec<Option<Node>> = vec![None; normalized.len() + 1];
        dp[normalized.len()] = Some(Node {
            score: 0.0,
            next: normalized.len(),
            ids: Vec::new(),
        });
        let offsets = normalized
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let scores = self.scores.as_ref();

        for offset in offsets.iter().rev().copied() {
            let remaining = &normalized[offset..];
            let mut best: Option<Node> = None;
            for (token, id) in &self.token_to_id {
                let id_usize = *id as usize;
                if self.special_ids.contains(id)
                    || byte_fallback_value(token).is_some()
                    || token.is_empty()
                    || !remaining.starts_with(token)
                {
                    continue;
                }
                let next = offset + token.len();
                let Some(tail) = dp.get(next).and_then(Option::as_ref) else {
                    continue;
                };
                let score_bias = scores
                    .and_then(|scores| scores.get(id_usize))
                    .copied()
                    .unwrap_or_default()
                    * 0.000_001;
                let score = tail.score - 1.0 + score_bias;
                if best.as_ref().is_none_or(|current| score > current.score) {
                    best = Some(Node {
                        score,
                        next,
                        ids: vec![*id],
                    });
                }
            }

            if let Some((ch, next)) = remaining
                .chars()
                .next()
                .map(|ch| (ch, offset + ch.len_utf8()))
            {
                let Some(tail) = dp.get(next).and_then(Option::as_ref) else {
                    continue;
                };
                let fallback_ids = self.sentencepiece_byte_fallback(ch)?;
                let score = tail.score - 100.0 * fallback_ids.len() as f32;
                if best.as_ref().is_none_or(|current| score > current.score) {
                    best = Some(Node {
                        score,
                        next,
                        ids: fallback_ids,
                    });
                }
            }
            dp[offset] = best;
        }

        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < normalized.len() {
            let node = dp[offset].as_ref().ok_or_else(|| {
                anyhow!("GGUF llama tokenizer cannot encode text at byte offset {offset}")
            })?;
            ids.extend_from_slice(&node.ids);
            offset = node.next;
        }
        Ok(ids)
    }

    fn sentencepiece_byte_fallback(&self, ch: char) -> Result<Vec<u32>> {
        let mut buffer = [0; 4];
        let mut ids = Vec::new();
        for byte in ch.encode_utf8(&mut buffer).as_bytes() {
            let token = format!("<0x{byte:02X}>");
            match self.token_to_id.get(&token).copied() {
                Some(id) => ids.push(id),
                None => match self.unknown_token_id {
                    Some(id) => {
                        ids.push(id);
                        break;
                    }
                    None => bail!("GGUF llama tokenizer has no byte fallback token {token}"),
                },
            }
        }
        Ok(ids)
    }

    fn encode_byte_level_bpe(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = encode_byte_level_text(text.as_bytes());
        let mut ids = Vec::new();
        for symbol in self.apply_bpe(&encoded) {
            match self.token_to_id.get(&symbol).copied() {
                Some(id) => ids.push(id),
                None => match self.unknown_token_id {
                    Some(id) => ids.push(id),
                    None => bail!("BPE produced token {symbol:?} that is missing from GGUF vocab"),
                },
            }
        }
        Ok(ids)
    }

    fn apply_bpe(&self, encoded: &str) -> Vec<String> {
        let mut symbols = encoded.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if symbols.len() < 2 || self.merge_ranks.is_empty() {
            return symbols;
        }

        loop {
            let mut best: Option<(usize, usize)> = None;
            for idx in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[idx].clone(), symbols[idx + 1].clone());
                let Some(rank) = self.merge_ranks.get(&pair).copied() else {
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

    fn encode_greedy(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            let mut best: Option<(&str, u32)> = None;
            for (token, id) in &self.token_to_id {
                if remaining.starts_with(token)
                    && best.is_none_or(|(best_token, _)| token.len() > best_token.len())
                {
                    best = Some((token.as_str(), *id));
                }
            }

            let Some((token, id)) = best else {
                if let Some(id) = self.unknown_token_id {
                    ids.push(id);
                    offset += remaining
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or_default();
                    continue;
                }
                bail!("GGUF tokenizer cannot encode text at byte offset {offset}");
            };
            ids.push(id);
            offset += token.len();
        }
        Ok(ids)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GgufTokenizerSummary {
    pub model: Option<String>,
    pub token_count: usize,
    pub merge_count: usize,
    pub has_scores: bool,
    pub has_token_types: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct QwenGgufConfig {
    pub architecture: String,
    pub family: ModelFamily,
    pub context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: Option<u32>,
    pub expert_feed_forward_length: Option<u32>,
    pub block_count: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub attention_key_length: Option<u32>,
    pub attention_value_length: Option<u32>,
    pub attention_q_lora_rank: Option<u32>,
    pub attention_kv_lora_rank: Option<u32>,
    pub attention_qk_rope_head_dim: Option<u32>,
    pub attention_qk_nope_head_dim: Option<u32>,
    pub attention_v_head_dim: Option<u32>,
    pub attention_qk_head_dim: Option<u32>,
    pub attention_mla_tensor_layout: bool,
    pub recurrent_ssm_tensor_layout: bool,
    pub ssm_conv_kernel: Option<u32>,
    pub ssm_inner_size: Option<u32>,
    pub ssm_state_size: Option<u32>,
    pub ssm_time_step_rank: Option<u32>,
    pub ssm_group_count: Option<u32>,
    pub ssm_dt_b_c_rms: Option<bool>,
    pub full_attention_interval: Option<u32>,
    pub attention_recurrent_layers: Option<Vec<bool>>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub expert_weights_norm: bool,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub rope_dimension_sections: Option<[u32; 4]>,
    pub rms_norm_eps: Option<f32>,
    pub vocab_size: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub file_type: Option<u32>,
    pub tensor_dtypes: Vec<String>,
    pub total_tensor_bytes: u64,
}

impl QwenGgufConfig {
    fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("GGUF metadata missing general.architecture"))?
            .to_ascii_lowercase();
        let family = ModelFamily::from_gguf_architecture(&architecture).ok_or_else(|| {
            anyhow!(
                "unsupported GGUF architecture '{architecture}'; only Qwen/Llama/Mistral/Mixtral/Gemma/Phi/DeepSeek/GLM GGUF is accepted"
            )
        })?;
        let prefix = architecture.clone();
        let context_length = gguf.required_u32(&format!("{prefix}.context_length"))?;
        let embedding_length = gguf.required_u32(&format!("{prefix}.embedding_length"))?;
        let block_count = gguf.required_u32(&format!("{prefix}.block_count"))?;
        reject_unsupported_mla_layout(gguf, family, &prefix, block_count)?;
        reject_unsupported_qwen_ssm_layout(gguf, family, &prefix)?;
        let attention_head_count = gguf.required_u32(&format!("{prefix}.attention.head_count"))?;
        let attention_key_length = gguf.metadata_u32(&format!("{prefix}.attention.key_length"));
        let attention_value_length = gguf.metadata_u32(&format!("{prefix}.attention.value_length"));
        let attention_q_lora_rank = gguf.metadata_u32(&format!("{prefix}.attention.q_lora_rank"));
        let attention_kv_lora_rank = gguf.metadata_u32(&format!("{prefix}.attention.kv_lora_rank"));
        let attention_qk_rope_head_dim =
            gguf.metadata_u32(&format!("{prefix}.attention.qk_rope_head_dim"));
        let attention_qk_nope_head_dim =
            gguf.metadata_u32(&format!("{prefix}.attention.qk_nope_head_dim"));
        let attention_v_head_dim = gguf.metadata_u32(&format!("{prefix}.attention.v_head_dim"));
        let attention_qk_head_dim = gguf.metadata_u32(&format!("{prefix}.attention.qk_head_dim"));
        let attention_mla_tensor_layout = qwen_mla_decoder_tensors_present(gguf, block_count);
        let recurrent_ssm_tensor_layout =
            qwen_any_recurrent_ssm_layer_tensors_present(gguf, block_count);
        let attention_recurrent_layers =
            qwen_attention_recurrent_layers(gguf, &prefix, block_count)?;
        reject_unsupported_custom_attention_lengths(
            family,
            &prefix,
            embedding_length,
            attention_head_count,
            attention_key_length,
            attention_value_length.or(attention_v_head_dim),
        )?;
        let attention_head_count_kv = gguf
            .metadata_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(attention_head_count);
        let expert_count = gguf.metadata_u32(&format!("{prefix}.expert_count"));
        let expert_used_count = gguf.metadata_u32(&format!("{prefix}.expert_used_count"));
        let vocab_size = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(array_len_as_u32);

        let mut tensor_dtypes = gguf
            .tensors
            .iter()
            .map(|tensor| tensor.dtype.label().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        tensor_dtypes.sort();
        let total_tensor_bytes = gguf.tensors.iter().try_fold(0u64, |acc, tensor| {
            Ok::<_, anyhow::Error>(acc + tensor.byte_len()?)
        })?;

        let rope_dimension_sections = match gguf
            .metadata_i32_array(&format!("{prefix}.rope.dimension_sections"))?
        {
            Some(values) => {
                if values.len() != 4 {
                    bail!(
                        "GGUF metadata {prefix}.rope.dimension_sections must contain exactly 4 integers, got {}",
                        values.len()
                    );
                }
                let mut sections = [0u32; 4];
                for (idx, value) in values.into_iter().enumerate() {
                    if value < 0 {
                        bail!(
                            "GGUF metadata {prefix}.rope.dimension_sections contains negative value {value}"
                        );
                    }
                    sections[idx] = u32::try_from(value).with_context(|| {
                        format!("GGUF metadata {prefix}.rope.dimension_sections is out of range")
                    })?;
                }
                Some(sections)
            }
            None => None,
        };

        let feed_forward_length = gguf.metadata_u32(&format!("{prefix}.feed_forward_length"));
        let expert_feed_forward_length = gguf
            .metadata_u32(&format!("{prefix}.expert_feed_forward_length"))
            .or_else(|| gguf.metadata_u32(&format!("{prefix}.moe_feed_forward_length")))
            .or_else(|| gguf.metadata_u32(&format!("{prefix}.moe_intermediate_size")))
            .or_else(|| {
                if expert_count.is_some() {
                    feed_forward_length
                } else {
                    None
                }
            });

        Ok(Self {
            architecture,
            family,
            context_length,
            embedding_length,
            feed_forward_length,
            expert_feed_forward_length,
            block_count,
            attention_head_count,
            attention_head_count_kv,
            attention_key_length,
            attention_value_length,
            attention_q_lora_rank,
            attention_kv_lora_rank,
            attention_qk_rope_head_dim,
            attention_qk_nope_head_dim,
            attention_v_head_dim,
            attention_qk_head_dim,
            attention_mla_tensor_layout,
            recurrent_ssm_tensor_layout,
            ssm_conv_kernel: gguf.metadata_u32(&format!("{prefix}.ssm.conv_kernel")),
            ssm_inner_size: gguf.metadata_u32(&format!("{prefix}.ssm.inner_size")),
            ssm_state_size: gguf.metadata_u32(&format!("{prefix}.ssm.state_size")),
            ssm_time_step_rank: gguf.metadata_u32(&format!("{prefix}.ssm.time_step_rank")),
            ssm_group_count: gguf.metadata_u32(&format!("{prefix}.ssm.group_count")),
            ssm_dt_b_c_rms: gguf.metadata_bool(&format!("{prefix}.ssm.dt_b_c_rms")),
            full_attention_interval: gguf
                .metadata_u32(&format!("{prefix}.full_attention_interval")),
            attention_recurrent_layers,
            expert_count,
            expert_used_count,
            expert_weights_norm: gguf
                .metadata_bool(&format!("{prefix}.expert_weights_norm"))
                .unwrap_or(true),
            rope_freq_base: gguf.metadata_f32(&format!("{prefix}.rope.freq_base")),
            rope_freq_scale: gguf.metadata_f32(&format!("{prefix}.rope.freq_scale")),
            rope_dimension_sections,
            rms_norm_eps: gguf.metadata_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")),
            vocab_size,
            eos_token_id: gguf.metadata_u32("tokenizer.ggml.eos_token_id"),
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            file_type: gguf.metadata_u32("general.file_type"),
            tensor_dtypes,
            total_tensor_bytes,
        })
    }

    pub fn quantization_label(&self) -> String {
        let mut labels = self.tensor_dtypes.clone();
        if labels.iter().any(|label| label == "Q4_K") {
            let replacement = match self.file_type {
                Some(14) => Some("Q4_K_S"),
                Some(15) => Some("Q4_K_M"),
                _ => None,
            };
            if let Some(replacement) = replacement {
                for label in &mut labels {
                    if label == "Q4_K" {
                        *label = replacement.to_string();
                    }
                }
            }
        }
        labels.sort();
        labels.dedup();
        labels.join(",")
    }

    pub fn default_rope_freq_base(&self) -> f32 {
        match self.family {
            ModelFamily::Llama
            | ModelFamily::Mistral
            | ModelFamily::Mixtral
            | ModelFamily::Gemma
            | ModelFamily::Phi
            | ModelFamily::DeepSeek => 10_000.0,
            _ => 1_000_000.0,
        }
    }

    pub fn attention_head_dim(&self) -> Option<u32> {
        let key = self.attention_key_head_dim()?;
        let value = self.attention_value_head_dim()?;
        if key == 0 || key != value {
            return None;
        }
        Some(key)
    }

    pub fn attention_key_head_dim(&self) -> Option<u32> {
        if let Some(qk) = self.attention_qk_head_dim {
            return (qk != 0).then_some(qk);
        }
        if let (Some(nope), Some(rope)) = (
            self.attention_qk_nope_head_dim,
            self.attention_qk_rope_head_dim,
        ) {
            return nope.checked_add(rope).filter(|value| *value != 0);
        }
        let dense = dense_attention_head_dim(self.embedding_length, self.attention_head_count);
        let key = self.attention_key_length.or(dense)?;
        (key != 0).then_some(key)
    }

    pub fn attention_value_head_dim(&self) -> Option<u32> {
        let dense = dense_attention_head_dim(self.embedding_length, self.attention_head_count);
        let value = self
            .attention_value_length
            .or(self.attention_v_head_dim)
            .or(dense)?;
        (value != 0).then_some(value)
    }
}

fn dense_attention_head_dim(embedding_length: u32, attention_head_count: u32) -> Option<u32> {
    if attention_head_count == 0 {
        return None;
    }
    embedding_length
        .checked_rem(attention_head_count)
        .filter(|remainder| *remainder == 0)
        .map(|_| embedding_length / attention_head_count)
}

fn cuda_model_family_label(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen2 | ModelFamily::Qwen3 => "Qwen",
        ModelFamily::Llama => "Llama",
        ModelFamily::Mistral => "Mistral",
        ModelFamily::Mixtral => "Mixtral",
        ModelFamily::Gemma => "Gemma",
        ModelFamily::Phi => "Phi",
        ModelFamily::DeepSeek => "DeepSeek",
        ModelFamily::GlmFlash => "GLM",
        ModelFamily::Hy3 => "Hy3",
        ModelFamily::NemotronH => "NemotronH",
        // MLX-only families (never reach the CUDA/GGUF path) — arms kept explicit so a new
        // ModelFamily variant is a compile error here rather than a silent fallthrough.
        ModelFamily::MiniMax => "MiniMax",
    }
}

fn reject_unsupported_custom_attention_lengths(
    family: ModelFamily,
    prefix: &str,
    embedding_length: u32,
    attention_head_count: u32,
    attention_key_length: Option<u32>,
    attention_value_length: Option<u32>,
) -> Result<()> {
    let expected_head_dim = dense_attention_head_dim(embedding_length, attention_head_count);
    let effective_key = attention_key_length.or(expected_head_dim);
    let effective_value = attention_value_length.or(expected_head_dim);
    if attention_key_length.is_none() && attention_value_length.is_none() {
        return Ok(());
    }
    if let (Some(key), Some(value)) = (effective_key, effective_value)
        && key != 0
        && value != 0
    {
        return Ok(());
    }

    let family_label = cuda_model_family_label(family);
    let expected = match expected_head_dim {
        Some(head_dim) => format!(
            "expected dense per-head length {head_dim} from {prefix}.embedding_length={embedding_length} / {prefix}.attention.head_count={attention_head_count}"
        ),
        None => format!(
            "could not derive a dense per-head length from {prefix}.embedding_length={embedding_length} / {prefix}.attention.head_count={attention_head_count}"
        ),
    };
    let mut unsupported = Vec::new();
    if let Some(value) = attention_key_length {
        unsupported.push(format!("{prefix}.attention.key_length={value}"));
    }
    if let Some(value) = attention_value_length {
        unsupported.push(format!("{prefix}.attention.value_length={value}"));
    }
    if attention_key_length.is_none()
        && let Some(value) = effective_key
    {
        unsupported.push(format!("{prefix}.attention.key_length=<dense {value}>"));
    }
    if attention_value_length.is_none()
        && let Some(value) = effective_value
    {
        unsupported.push(format!("{prefix}.attention.value_length=<dense {value}>"));
    }
    bail!(
        "unsupported {family_label} GGUF metadata {}: attention key/value lengths must resolve to non-zero per-head dimensions for CUDA {family_label} support; {expected}",
        unsupported.join(", ")
    );
}

fn reject_unsupported_mla_layout(
    gguf: &GgufFile,
    family: ModelFamily,
    prefix: &str,
    block_count: u32,
) -> Result<()> {
    if family != ModelFamily::DeepSeek && family != ModelFamily::GlmFlash {
        return Ok(());
    }
    let family_label = cuda_model_family_label(family);
    let dense_decoder_present = decoder_dense_attention_tensors_present(gguf, block_count);
    let mla_decoder_present = qwen_mla_decoder_tensors_present(gguf, block_count);
    let mut mla_metadata_keys = Vec::new();

    for suffix in [
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.qk_rope_head_dim",
        "attention.qk_nope_head_dim",
        "attention.v_head_dim",
        "attention.qk_head_dim",
    ] {
        let key = format!("{prefix}.{suffix}");
        if gguf.metadata.contains_key(&key) {
            mla_metadata_keys.push(key);
        }
    }

    for layer in 0..block_count {
        for layer_prefix in layer_prefix_variants(&format!("blk.{layer}")) {
            for suffix in [
                "attn_q_a.weight",
                "attn_q_a_norm.weight",
                "attn_q_b.weight",
                "attn_q_a_proj.weight",
                "attn_q_a_layernorm.weight",
                "attn_q_b_proj.weight",
                "attn_kv_a_mqa.weight",
                "attn_kv_a_norm.weight",
                "attn_kv_b.weight",
                "attn_kv_a_proj_with_mqa.weight",
                "attn_kv_a_layernorm.weight",
                "attn_kv_b_proj.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_proj.weight",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_a_norm.weight",
                "self_attn.kv_b_proj.weight",
                "self_attention.q_a_proj.weight",
                "self_attention.q_a_layernorm.weight",
                "self_attention.q_a_norm.weight",
                "self_attention.q_b_proj.weight",
                "self_attention.kv_a_proj_with_mqa.weight",
                "self_attention.kv_a_proj.weight",
                "self_attention.kv_a_layernorm.weight",
                "self_attention.kv_a_norm.weight",
                "self_attention.kv_b_proj.weight",
                "attention.q_a_proj.weight",
                "attention.q_a_layernorm.weight",
                "attention.q_a_norm.weight",
                "attention.q_b_proj.weight",
                "attention.kv_a_proj_with_mqa.weight",
                "attention.kv_a_proj.weight",
                "attention.kv_a_layernorm.weight",
                "attention.kv_a_norm.weight",
                "attention.kv_b_proj.weight",
                "attn.wkv.weight",
            ] {
                let name = format!("{layer_prefix}.{suffix}");
                if gguf.tensor(&name).is_some() {
                    if !dense_decoder_present && !mla_decoder_present {
                        bail!(
                            "unsupported {family_label} GGUF tensor layout: tensor {name} uses incomplete MLA attention; CUDA {family_label} support requires either split attn_q/attn_k/attn_v decoder tensors or a complete MLA tensor set attn_q_a/attn_q_a_norm/attn_q_b/attn_kv_a_mqa/attn_kv_a_norm/attn_kv_b"
                        );
                    }
                    break;
                }
            }
        }
    }

    if !mla_metadata_keys.is_empty() && !dense_decoder_present && !mla_decoder_present {
        bail!(
            "unsupported {family_label} GGUF metadata {}: MLA attention metadata is present without split attn_q/attn_k/attn_v decoder tensors or a complete MLA tensor set; CUDA {family_label} support can load split decoder tensors and complete low-rank MLA tensor layouts",
            mla_metadata_keys.join(", ")
        );
    }

    Ok(())
}

fn decoder_dense_attention_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).all(|layer| {
        let prefix = format!("blk.{layer}");
        decoder_split_attention_tensors_present(gguf, &prefix)
            || qwen_dense_packed_qkv_weight_names(&prefix)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
    })
}

fn qwen_mla_decoder_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).all(|layer| {
        let prefix = format!("blk.{layer}");
        qwen_mla_attention_tensors_present(gguf, &prefix)
    })
}

fn qwen_recurrent_ssm_decoder_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    block_count != 0
        && (0..block_count).all(|layer| {
            let prefix = format!("blk.{layer}");
            let dense_attention = decoder_split_attention_tensors_present(gguf, &prefix)
                || qwen_dense_packed_qkv_weight_names(&prefix)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some());
            let recurrent_ssm = qwen_ssm_layer_tensors_present(gguf, &prefix);
            let split_ffn = ["gate", "up", "down"].iter().all(|kind| {
                qwen_dense_ffn_weight_names(&prefix, kind)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some())
            });
            let packed_ffn = qwen_dense_ffn_weight_names(&prefix, "down")
                .iter()
                .any(|name| gguf.tensor(name).is_some())
                && qwen_dense_packed_ffn_gate_up_weight_names(&prefix)
                    .into_iter()
                    .chain(qwen_dense_packed_ffn_up_gate_weight_names(&prefix))
                    .any(|name| gguf.tensor(&name).is_some());
            (dense_attention || recurrent_ssm) && (split_ffn || packed_ffn)
        })
}

fn qwen_any_recurrent_ssm_layer_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).any(|layer| {
        let prefix = format!("blk.{layer}");
        qwen_ssm_layer_tensors_present(gguf, &prefix)
    })
}

fn reject_unsupported_qwen_ssm_layout(
    gguf: &GgufFile,
    family: ModelFamily,
    prefix: &str,
) -> Result<()> {
    if !matches!(family, ModelFamily::Qwen2 | ModelFamily::Qwen3) {
        return Ok(());
    }

    let block_count = gguf
        .metadata_u32(&format!("{prefix}.block_count"))
        .unwrap_or(0);
    let dense_decoder_present = qwen_dense_decoder_tensors_present(gguf, block_count);
    let recurrent_ssm_decoder_present =
        qwen_recurrent_ssm_decoder_tensors_present(gguf, block_count);
    let mut ssm_metadata = Vec::new();
    for key in gguf.metadata.keys() {
        if key.starts_with(prefix)
            && let Some(feature) = qwen_ssm_metadata_feature(key)
        {
            ssm_metadata.push((key.clone(), feature));
        }
    }

    for tensor in &gguf.tensors {
        if let Some(feature) = qwen_ssm_tensor_feature(&tensor.name) {
            if !dense_decoder_present && !recurrent_ssm_decoder_present {
                bail!(
                    "unsupported Qwen GGUF tensor layout: tensor {} uses {feature}; CUDA Qwen support requires either dense attention/MLP decoder tensors or a complete recurrent SSM tensor set ssm_in or attn_qkv+attn_gate plus ssm_conv1d/ssm_dt/ssm_a/ssm_ba/ssm_norm/ssm_out",
                    tensor.name
                );
            }
        }
    }

    if !ssm_metadata.is_empty() && !dense_decoder_present && !recurrent_ssm_decoder_present {
        let details = ssm_metadata
            .iter()
            .map(|(key, feature)| format!("{key} ({feature})"))
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "unsupported Qwen GGUF metadata {details}: recurrent/SSM metadata is present without dense decoder tensors or a complete recurrent SSM tensor set; CUDA Qwen support requires dense attn_q/attn_k/attn_v decoder tensors or complete Qwen3-Next SSM tensors"
        );
    }

    Ok(())
}

fn qwen_dense_decoder_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).all(|layer| {
        let prefix = format!("blk.{layer}");
        let dense_attention = decoder_split_attention_tensors_present(gguf, &prefix)
            || qwen_dense_packed_qkv_weight_names(&prefix)
                .iter()
                .any(|name| gguf.tensor(name).is_some());
        let split_ffn = ["gate", "up", "down"].iter().all(|kind| {
            qwen_dense_ffn_weight_names(&prefix, kind)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
        });
        let packed_ffn = qwen_dense_ffn_weight_names(&prefix, "down")
            .iter()
            .any(|name| gguf.tensor(name).is_some())
            && qwen_dense_packed_ffn_gate_up_weight_names(&prefix)
                .into_iter()
                .chain(qwen_dense_packed_ffn_up_gate_weight_names(&prefix))
                .any(|name| gguf.tensor(&name).is_some());
        dense_attention && (split_ffn || packed_ffn)
    })
}

fn decoder_split_attention_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
    ["q", "k", "v"].iter().all(|suffix| {
        qwen_dense_attention_weight_names(prefix, suffix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
    })
}

pub fn qwen_dense_output_norm_weight_names() -> Vec<String> {
    [
        "output_norm.weight",
        "norm.weight",
        "model.norm.weight",
        "language_model.norm.weight",
        "language_model.model.norm.weight",
        "transformer.norm_f.weight",
        "model.transformer.norm_f.weight",
        "transformer.encoder.final_layernorm.weight",
        "transformer.final_layernorm.weight",
        "transformer.final_layer_norm.weight",
        "model.final_layernorm.weight",
        "model.final_layer_norm.weight",
        "model.transformer.final_layernorm.weight",
        "model.transformer.final_layer_norm.weight",
        "language_model.final_layernorm.weight",
        "language_model.final_layer_norm.weight",
        "language_model.model.final_layernorm.weight",
        "language_model.model.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.weight",
        "final_layernorm.weight",
        "final_layer_norm.weight",
        "transformer.ln_f.weight",
        "model.transformer.ln_f.weight",
        "ln_f.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn qwen_dense_token_embd_weight_names() -> Vec<String> {
    [
        "token_embd.weight",
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.transformer.embed_tokens.weight",
        "model.transformer.wte.weight",
        "model.tok_embeddings.weight",
        "tok_embeddings.weight",
        "transformer.wte.weight",
        "wte.weight",
        "gpt_neox.embed_in.weight",
        "model.embed_in.weight",
        "embed_in.weight",
        "transformer.embedding.word_embeddings.weight",
        "transformer.word_embeddings.weight",
        "model.word_embeddings.weight",
        "word_embeddings.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn qwen_dense_output_weight_names() -> Vec<String> {
    [
        "output.weight",
        "lm_head.weight",
        "model.lm_head.weight",
        "transformer.lm_head.weight",
        "model.transformer.lm_head.weight",
        "language_model.lm_head.weight",
        "language_model.model.lm_head.weight",
        "transformer.output_layer.weight",
        "model.transformer.output_layer.weight",
        "gpt_neox.embed_out.weight",
        "model.output.weight",
        "language_model.output.weight",
        "language_model.model.output.weight",
        "output_layer.weight",
        "model.embed_out.weight",
        "model.transformer.embed_out.weight",
        "embed_out.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn qwen_dense_output_bias_names() -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_output_weight_names())
}

pub fn qwen_dense_attention_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_norm.weight"),
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.input_layer_norm.weight"),
            format!("{prefix}.pre_attention_layernorm.weight"),
            format!("{prefix}.pre_attention_layer_norm.weight"),
            format!("{prefix}.self_attn_layer_norm.weight"),
            format!("{prefix}.attention_layernorm.weight"),
            format!("{prefix}.attention_norm.weight"),
            format!("{prefix}.ln_1.weight"),
            format!("{prefix}.ln1.weight"),
        ]
    })
}

pub fn qwen_dense_ffn_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_norm.weight"),
            format!("{prefix}.attn_post_norm.weight"),
            format!("{prefix}.post_attention_norm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.post_attention_layer_norm.weight"),
            format!("{prefix}.pre_feedforward_layernorm.weight"),
            format!("{prefix}.pre_feedforward_layer_norm.weight"),
            format!("{prefix}.post_feedforward_layernorm.weight"),
            format!("{prefix}.post_feedforward_layer_norm.weight"),
            format!("{prefix}.post_feedforward_norm.weight"),
            format!("{prefix}.post_feed_forward_layernorm.weight"),
            format!("{prefix}.post_feed_forward_layer_norm.weight"),
            format!("{prefix}.post_feed_forward_norm.weight"),
            format!("{prefix}.post_ffw_norm.weight"),
            format!("{prefix}.ffw_norm.weight"),
            format!("{prefix}.ffn_post_norm.weight"),
            format!("{prefix}.post_ffn_norm.weight"),
            format!("{prefix}.mlp_layer_norm.weight"),
            format!("{prefix}.ffn_layernorm.weight"),
            format!("{prefix}.ffn_layer_norm.weight"),
            format!("{prefix}.ln_2.weight"),
            format!("{prefix}.ln2.weight"),
        ]
    })
}

pub fn qwen_dense_attention_weight_names(prefix: &str, suffix: &str) -> Vec<String> {
    let Some((canonical, hf, llama)) = attention_projection_alias_parts(suffix) else {
        return layer_prefix_aliases(prefix, |prefix| {
            vec![format!("{prefix}.attn_{suffix}.weight")]
        });
    };
    layer_prefix_aliases(prefix, |prefix| {
        let mut names = vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.{hf}.weight"),
            format!("{prefix}.self_attn.{hf}.weight"),
            format!("{prefix}.self_attention.{hf}.weight"),
            format!("{prefix}.attention.{hf}.weight"),
            format!("{prefix}.attn.{hf}.weight"),
            format!("{prefix}.attention.{llama}.weight"),
            format!("{prefix}.self_attn.{llama}.weight"),
            format!("{prefix}.self_attention.{llama}.weight"),
            format!("{prefix}.attn.{llama}.weight"),
            format!("{prefix}.{llama}.weight"),
        ];
        match suffix {
            "q" => names.extend([
                format!("{prefix}.query.weight"),
                format!("{prefix}.self_attn.query.weight"),
                format!("{prefix}.self_attention.query.weight"),
                format!("{prefix}.attention.query.weight"),
                format!("{prefix}.attn.query.weight"),
                format!("{prefix}.Wq.weight"),
                format!("{prefix}.self_attn.Wq.weight"),
                format!("{prefix}.self_attention.Wq.weight"),
                format!("{prefix}.attention.Wq.weight"),
                format!("{prefix}.attn.Wq.weight"),
                format!("{prefix}.w_q.weight"),
                format!("{prefix}.self_attn.w_q.weight"),
                format!("{prefix}.self_attention.w_q.weight"),
                format!("{prefix}.attention.w_q.weight"),
                format!("{prefix}.attn.w_q.weight"),
            ]),
            "k" => names.extend([
                format!("{prefix}.key.weight"),
                format!("{prefix}.self_attn.key.weight"),
                format!("{prefix}.self_attention.key.weight"),
                format!("{prefix}.attention.key.weight"),
                format!("{prefix}.attn.key.weight"),
                format!("{prefix}.Wk.weight"),
                format!("{prefix}.self_attn.Wk.weight"),
                format!("{prefix}.self_attention.Wk.weight"),
                format!("{prefix}.attention.Wk.weight"),
                format!("{prefix}.attn.Wk.weight"),
                format!("{prefix}.w_k.weight"),
                format!("{prefix}.self_attn.w_k.weight"),
                format!("{prefix}.self_attention.w_k.weight"),
                format!("{prefix}.attention.w_k.weight"),
                format!("{prefix}.attn.w_k.weight"),
            ]),
            "v" => names.extend([
                format!("{prefix}.value.weight"),
                format!("{prefix}.self_attn.value.weight"),
                format!("{prefix}.self_attention.value.weight"),
                format!("{prefix}.attention.value.weight"),
                format!("{prefix}.attn.value.weight"),
                format!("{prefix}.Wv.weight"),
                format!("{prefix}.self_attn.Wv.weight"),
                format!("{prefix}.self_attention.Wv.weight"),
                format!("{prefix}.attention.Wv.weight"),
                format!("{prefix}.attn.Wv.weight"),
                format!("{prefix}.w_v.weight"),
                format!("{prefix}.self_attn.w_v.weight"),
                format!("{prefix}.self_attention.w_v.weight"),
                format!("{prefix}.attention.w_v.weight"),
                format!("{prefix}.attn.w_v.weight"),
            ]),
            _ => {}
        }
        if suffix == "output" {
            names.extend([
                format!("{prefix}.dense.weight"),
                format!("{prefix}.self_attn.dense.weight"),
                format!("{prefix}.self_attention.dense.weight"),
                format!("{prefix}.attention.dense.weight"),
                format!("{prefix}.attn.dense.weight"),
                format!("{prefix}.out_proj.weight"),
                format!("{prefix}.self_attn.out_proj.weight"),
                format!("{prefix}.self_attention.out_proj.weight"),
                format!("{prefix}.attention.out_proj.weight"),
                format!("{prefix}.attn.out_proj.weight"),
                format!("{prefix}.mixer.out_proj.weight"),
                format!("{prefix}.c_proj.weight"),
                format!("{prefix}.attn.c_proj.weight"),
                format!("{prefix}.self_attn.c_proj.weight"),
                format!("{prefix}.self_attention.c_proj.weight"),
                format!("{prefix}.Wo.weight"),
                format!("{prefix}.self_attn.Wo.weight"),
                format!("{prefix}.self_attention.Wo.weight"),
                format!("{prefix}.attention.Wo.weight"),
                format!("{prefix}.attn.Wo.weight"),
                format!("{prefix}.w_o.weight"),
                format!("{prefix}.self_attn.w_o.weight"),
                format!("{prefix}.self_attention.w_o.weight"),
                format!("{prefix}.attention.w_o.weight"),
                format!("{prefix}.attn.w_o.weight"),
                format!("{prefix}.out.weight"),
                format!("{prefix}.self_attn.out.weight"),
                format!("{prefix}.self_attention.out.weight"),
                format!("{prefix}.attention.out.weight"),
                format!("{prefix}.attn.out.weight"),
                format!("{prefix}.proj.weight"),
                format!("{prefix}.self_attn.proj.weight"),
                format!("{prefix}.self_attention.proj.weight"),
                format!("{prefix}.attention.proj.weight"),
                format!("{prefix}.attn.proj.weight"),
            ]);
        }
        names
    })
}

pub fn qwen_dense_attention_bias_names(prefix: &str, suffix: &str) -> Vec<String> {
    let Some((canonical, hf, llama)) = attention_projection_alias_parts(suffix) else {
        return layer_prefix_aliases(prefix, |prefix| {
            vec![format!("{prefix}.attn_{suffix}.bias")]
        });
    };
    layer_prefix_aliases(prefix, |prefix| {
        let mut names = vec![
            format!("{prefix}.{canonical}.bias"),
            format!("{prefix}.{hf}.bias"),
            format!("{prefix}.self_attn.{hf}.bias"),
            format!("{prefix}.self_attention.{hf}.bias"),
            format!("{prefix}.attention.{hf}.bias"),
            format!("{prefix}.attn.{hf}.bias"),
            format!("{prefix}.attention.{llama}.bias"),
            format!("{prefix}.self_attn.{llama}.bias"),
            format!("{prefix}.self_attention.{llama}.bias"),
            format!("{prefix}.attn.{llama}.bias"),
            format!("{prefix}.{llama}.bias"),
        ];
        match suffix {
            "q" => names.extend([
                format!("{prefix}.query.bias"),
                format!("{prefix}.self_attn.query.bias"),
                format!("{prefix}.self_attention.query.bias"),
                format!("{prefix}.attention.query.bias"),
                format!("{prefix}.attn.query.bias"),
                format!("{prefix}.Wq.bias"),
                format!("{prefix}.self_attn.Wq.bias"),
                format!("{prefix}.self_attention.Wq.bias"),
                format!("{prefix}.attention.Wq.bias"),
                format!("{prefix}.attn.Wq.bias"),
                format!("{prefix}.w_q.bias"),
                format!("{prefix}.self_attn.w_q.bias"),
                format!("{prefix}.self_attention.w_q.bias"),
                format!("{prefix}.attention.w_q.bias"),
                format!("{prefix}.attn.w_q.bias"),
            ]),
            "k" => names.extend([
                format!("{prefix}.key.bias"),
                format!("{prefix}.self_attn.key.bias"),
                format!("{prefix}.self_attention.key.bias"),
                format!("{prefix}.attention.key.bias"),
                format!("{prefix}.attn.key.bias"),
                format!("{prefix}.Wk.bias"),
                format!("{prefix}.self_attn.Wk.bias"),
                format!("{prefix}.self_attention.Wk.bias"),
                format!("{prefix}.attention.Wk.bias"),
                format!("{prefix}.attn.Wk.bias"),
                format!("{prefix}.w_k.bias"),
                format!("{prefix}.self_attn.w_k.bias"),
                format!("{prefix}.self_attention.w_k.bias"),
                format!("{prefix}.attention.w_k.bias"),
                format!("{prefix}.attn.w_k.bias"),
            ]),
            "v" => names.extend([
                format!("{prefix}.value.bias"),
                format!("{prefix}.self_attn.value.bias"),
                format!("{prefix}.self_attention.value.bias"),
                format!("{prefix}.attention.value.bias"),
                format!("{prefix}.attn.value.bias"),
                format!("{prefix}.Wv.bias"),
                format!("{prefix}.self_attn.Wv.bias"),
                format!("{prefix}.self_attention.Wv.bias"),
                format!("{prefix}.attention.Wv.bias"),
                format!("{prefix}.attn.Wv.bias"),
                format!("{prefix}.w_v.bias"),
                format!("{prefix}.self_attn.w_v.bias"),
                format!("{prefix}.self_attention.w_v.bias"),
                format!("{prefix}.attention.w_v.bias"),
                format!("{prefix}.attn.w_v.bias"),
            ]),
            _ => {}
        }
        if suffix == "output" {
            names.extend([
                format!("{prefix}.dense.bias"),
                format!("{prefix}.self_attn.dense.bias"),
                format!("{prefix}.self_attention.dense.bias"),
                format!("{prefix}.attention.dense.bias"),
                format!("{prefix}.attn.dense.bias"),
                format!("{prefix}.out_proj.bias"),
                format!("{prefix}.self_attn.out_proj.bias"),
                format!("{prefix}.self_attention.out_proj.bias"),
                format!("{prefix}.attention.out_proj.bias"),
                format!("{prefix}.attn.out_proj.bias"),
                format!("{prefix}.mixer.out_proj.bias"),
                format!("{prefix}.c_proj.bias"),
                format!("{prefix}.attn.c_proj.bias"),
                format!("{prefix}.self_attn.c_proj.bias"),
                format!("{prefix}.self_attention.c_proj.bias"),
                format!("{prefix}.Wo.bias"),
                format!("{prefix}.self_attn.Wo.bias"),
                format!("{prefix}.self_attention.Wo.bias"),
                format!("{prefix}.attention.Wo.bias"),
                format!("{prefix}.attn.Wo.bias"),
                format!("{prefix}.w_o.bias"),
                format!("{prefix}.self_attn.w_o.bias"),
                format!("{prefix}.self_attention.w_o.bias"),
                format!("{prefix}.attention.w_o.bias"),
                format!("{prefix}.attn.w_o.bias"),
                format!("{prefix}.out.bias"),
                format!("{prefix}.self_attn.out.bias"),
                format!("{prefix}.self_attention.out.bias"),
                format!("{prefix}.attention.out.bias"),
                format!("{prefix}.attn.out.bias"),
                format!("{prefix}.proj.bias"),
                format!("{prefix}.self_attn.proj.bias"),
                format!("{prefix}.self_attention.proj.bias"),
                format!("{prefix}.attention.proj.bias"),
                format!("{prefix}.attn.proj.bias"),
            ]);
        }
        names
    })
}

pub fn qwen_dense_gated_attention_q_weight_name(
    gguf: &GgufFile,
    prefix: &str,
    q_dim: u64,
    embed: u64,
) -> Option<String> {
    let gated_q_dim = q_dim.checked_mul(2)?;
    qwen_dense_attention_weight_names(prefix, "q")
        .into_iter()
        .find(|name| {
            gguf.tensor(name).is_some_and(|tensor| {
                tensor_dimensions_match_matrix(&tensor.info.dimensions, embed, gated_q_dim)
            })
        })
}

pub fn qwen_dense_gated_attention_q_bias_name(
    gguf: &GgufFile,
    prefix: &str,
    q_dim: u64,
) -> Option<String> {
    let gated_q_dim = q_dim.checked_mul(2)?;
    qwen_dense_attention_bias_names(prefix, "q")
        .into_iter()
        .find(|name| {
            gguf.tensor(name)
                .is_some_and(|tensor| tensor.info.dimensions == [gated_q_dim])
        })
}

pub fn qwen_dense_packed_qkv_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_qkv.weight"),
            format!("{prefix}.qkv.weight"),
            format!("{prefix}.qkv_proj.weight"),
            format!("{prefix}.self_attn.qkv.weight"),
            format!("{prefix}.self_attn.qkv_proj.weight"),
            format!("{prefix}.self_attention.qkv.weight"),
            format!("{prefix}.self_attention.qkv_proj.weight"),
            format!("{prefix}.attention.qkv.weight"),
            format!("{prefix}.attention.qkv_proj.weight"),
            format!("{prefix}.attn.qkv.weight"),
            format!("{prefix}.attn.qkv_proj.weight"),
            format!("{prefix}.query_key_value.weight"),
            format!("{prefix}.self_attn.query_key_value.weight"),
            format!("{prefix}.self_attention.query_key_value.weight"),
            format!("{prefix}.attention.query_key_value.weight"),
            format!("{prefix}.attn.query_key_value.weight"),
            format!("{prefix}.mixer.query_key_value.weight"),
            format!("{prefix}.Wqkv.weight"),
            format!("{prefix}.self_attn.Wqkv.weight"),
            format!("{prefix}.self_attention.Wqkv.weight"),
            format!("{prefix}.attention.Wqkv.weight"),
            format!("{prefix}.attn.Wqkv.weight"),
            format!("{prefix}.wqkv.weight"),
            format!("{prefix}.self_attn.wqkv.weight"),
            format!("{prefix}.self_attention.wqkv.weight"),
            format!("{prefix}.attention.wqkv.weight"),
            format!("{prefix}.attn.wqkv.weight"),
            format!("{prefix}.W_pack.weight"),
            format!("{prefix}.self_attn.W_pack.weight"),
            format!("{prefix}.self_attention.W_pack.weight"),
            format!("{prefix}.attention.W_pack.weight"),
            format!("{prefix}.attn.W_pack.weight"),
            format!("{prefix}.c_attn.weight"),
            format!("{prefix}.attn.c_attn.weight"),
            format!("{prefix}.attention.c_attn.weight"),
            format!("{prefix}.self_attn.c_attn.weight"),
            format!("{prefix}.self_attention.c_attn.weight"),
            format!("{prefix}.mixer.c_attn.weight"),
            format!("{prefix}.mixer.W_pack.weight"),
            format!("{prefix}.mixer.w_pack.weight"),
            format!("{prefix}.mixer.Wqkv.weight"),
            format!("{prefix}.mixer.wqkv.weight"),
        ]
    })
}

pub fn qwen_dense_packed_qkv_bias_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_qkv.bias"),
            format!("{prefix}.qkv.bias"),
            format!("{prefix}.qkv_proj.bias"),
            format!("{prefix}.self_attn.qkv.bias"),
            format!("{prefix}.self_attn.qkv_proj.bias"),
            format!("{prefix}.self_attention.qkv.bias"),
            format!("{prefix}.self_attention.qkv_proj.bias"),
            format!("{prefix}.attention.qkv.bias"),
            format!("{prefix}.attention.qkv_proj.bias"),
            format!("{prefix}.attn.qkv.bias"),
            format!("{prefix}.attn.qkv_proj.bias"),
            format!("{prefix}.query_key_value.bias"),
            format!("{prefix}.self_attn.query_key_value.bias"),
            format!("{prefix}.self_attention.query_key_value.bias"),
            format!("{prefix}.attention.query_key_value.bias"),
            format!("{prefix}.attn.query_key_value.bias"),
            format!("{prefix}.mixer.query_key_value.bias"),
            format!("{prefix}.Wqkv.bias"),
            format!("{prefix}.self_attn.Wqkv.bias"),
            format!("{prefix}.self_attention.Wqkv.bias"),
            format!("{prefix}.attention.Wqkv.bias"),
            format!("{prefix}.attn.Wqkv.bias"),
            format!("{prefix}.wqkv.bias"),
            format!("{prefix}.self_attn.wqkv.bias"),
            format!("{prefix}.self_attention.wqkv.bias"),
            format!("{prefix}.attention.wqkv.bias"),
            format!("{prefix}.attn.wqkv.bias"),
            format!("{prefix}.W_pack.bias"),
            format!("{prefix}.self_attn.W_pack.bias"),
            format!("{prefix}.self_attention.W_pack.bias"),
            format!("{prefix}.attention.W_pack.bias"),
            format!("{prefix}.attn.W_pack.bias"),
            format!("{prefix}.c_attn.bias"),
            format!("{prefix}.attn.c_attn.bias"),
            format!("{prefix}.attention.c_attn.bias"),
            format!("{prefix}.self_attn.c_attn.bias"),
            format!("{prefix}.self_attention.c_attn.bias"),
            format!("{prefix}.mixer.c_attn.bias"),
            format!("{prefix}.mixer.W_pack.bias"),
            format!("{prefix}.mixer.w_pack.bias"),
            format!("{prefix}.mixer.Wqkv.bias"),
            format!("{prefix}.mixer.wqkv.bias"),
        ]
    })
}

pub fn qwen_mla_q_a_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_q_a.weight"),
            format!("{prefix}.attn_q_a_proj.weight"),
            format!("{prefix}.self_attn.q_a_proj.weight"),
            format!("{prefix}.self_attention.q_a_proj.weight"),
            format!("{prefix}.attention.q_a_proj.weight"),
        ]
    })
}

pub fn qwen_mla_q_a_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_q_a_norm.weight"),
            format!("{prefix}.attn_q_a_layernorm.weight"),
            format!("{prefix}.self_attn.q_a_layernorm.weight"),
            format!("{prefix}.self_attn.q_a_norm.weight"),
            format!("{prefix}.self_attention.q_a_layernorm.weight"),
            format!("{prefix}.self_attention.q_a_norm.weight"),
            format!("{prefix}.attention.q_a_layernorm.weight"),
            format!("{prefix}.attention.q_a_norm.weight"),
        ]
    })
}

pub fn qwen_mla_q_b_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_q_b.weight"),
            format!("{prefix}.attn_q_b_proj.weight"),
            format!("{prefix}.self_attn.q_b_proj.weight"),
            format!("{prefix}.self_attention.q_b_proj.weight"),
            format!("{prefix}.attention.q_b_proj.weight"),
        ]
    })
}

pub fn qwen_mla_kv_a_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_kv_a_mqa.weight"),
            format!("{prefix}.attn_kv_a_proj_with_mqa.weight"),
            format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            format!("{prefix}.self_attn.kv_a_proj.weight"),
            format!("{prefix}.self_attention.kv_a_proj_with_mqa.weight"),
            format!("{prefix}.self_attention.kv_a_proj.weight"),
            format!("{prefix}.attention.kv_a_proj_with_mqa.weight"),
            format!("{prefix}.attention.kv_a_proj.weight"),
        ]
    })
}

pub fn qwen_mla_kv_a_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_kv_a_norm.weight"),
            format!("{prefix}.attn_kv_a_layernorm.weight"),
            format!("{prefix}.self_attn.kv_a_layernorm.weight"),
            format!("{prefix}.self_attn.kv_a_norm.weight"),
            format!("{prefix}.self_attention.kv_a_layernorm.weight"),
            format!("{prefix}.self_attention.kv_a_norm.weight"),
            format!("{prefix}.attention.kv_a_layernorm.weight"),
            format!("{prefix}.attention.kv_a_norm.weight"),
        ]
    })
}

pub fn qwen_mla_kv_b_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_kv_b.weight"),
            format!("{prefix}.attn_kv_b_proj.weight"),
            format!("{prefix}.self_attn.kv_b_proj.weight"),
            format!("{prefix}.self_attention.kv_b_proj.weight"),
            format!("{prefix}.attention.kv_b_proj.weight"),
        ]
    })
}

pub fn qwen_mla_attention_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
    qwen_mla_q_a_weight_names(prefix)
        .iter()
        .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_q_a_norm_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_q_b_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_kv_a_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_kv_a_norm_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_kv_b_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
}

pub fn qwen_ssm_in_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_in.weight"),
            format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
            format!("{prefix}.gated_delta.in_proj_qkvz.weight"),
        ]
    })
}

pub fn qwen_ssm_qkv_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_qkv_weight_names(prefix)
}

pub fn qwen_ssm_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_gate.weight"),
            format!("{prefix}.wqkv_gate.weight"),
            format!("{prefix}.linear_attn.gate_proj.weight"),
            format!("{prefix}.linear_attn.in_proj_gate.weight"),
            format!("{prefix}.gated_delta.gate_proj.weight"),
        ]
    })
}

pub fn qwen_ssm_conv1d_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_conv1d.weight"),
            format!("{prefix}.conv1d.weight"),
            format!("{prefix}.linear_attn.conv1d.weight"),
            format!("{prefix}.gated_delta.conv1d.weight"),
        ]
    })
}

pub fn qwen_ssm_dt_bias_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_dt.bias"),
            format!("{prefix}.dt_bias"),
            format!("{prefix}.linear_attn.dt_bias"),
            format!("{prefix}.gated_delta.dt_bias"),
        ]
    })
}

pub fn qwen_ssm_a_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_a"),
            format!("{prefix}.ssm_a.weight"),
            format!("{prefix}.A_log"),
            format!("{prefix}.linear_attn.A_log"),
            format!("{prefix}.gated_delta.A_log"),
        ]
    })
}

pub fn qwen_ssm_ba_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_ba.weight"),
            format!("{prefix}.ssm_beta_alpha.weight"),
            format!("{prefix}.linear_attn.in_proj_ba.weight"),
            format!("{prefix}.gated_delta.in_proj_ba.weight"),
        ]
    })
}

pub fn qwen_ssm_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_norm.weight"),
            format!("{prefix}.linear_attn.norm.weight"),
            format!("{prefix}.gated_delta.norm.weight"),
        ]
    })
}

pub fn qwen_ssm_out_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_out.weight"),
            format!("{prefix}.linear_attn.out_proj.weight"),
            format!("{prefix}.gated_delta.out_proj.weight"),
        ]
    })
}

pub fn qwen_ssm_layer_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
    qwen_ssm_layer_tensors_present_with(|name| gguf.tensor(name).is_some(), prefix)
}

pub fn qwen_dense_packed_ffn_gate_up_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_up.weight"),
            format!("{prefix}.gate_up_proj.weight"),
            format!("{prefix}.mlp.gate_up_proj.weight"),
            format!("{prefix}.ffn.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.mlp.gate_up_proj.weight"),
            format!("{prefix}.gate_up.weight"),
            format!("{prefix}.mlp.gate_up.weight"),
            format!("{prefix}.ffn.gate_up.weight"),
            format!("{prefix}.feed_forward.gate_up.weight"),
            format!("{prefix}.feed_forward.mlp.gate_up.weight"),
            format!("{prefix}.fc1.weight"),
            format!("{prefix}.mlp.fc1.weight"),
            format!("{prefix}.ffn.fc1.weight"),
            format!("{prefix}.feed_forward.fc1.weight"),
            format!("{prefix}.feed_forward.mlp.fc1.weight"),
            format!("{prefix}.dense_h_to_4h.weight"),
            format!("{prefix}.mlp.dense_h_to_4h.weight"),
            format!("{prefix}.ffn.dense_h_to_4h.weight"),
            format!("{prefix}.feed_forward.dense_h_to_4h.weight"),
            format!("{prefix}.feed_forward.mlp.dense_h_to_4h.weight"),
            format!("{prefix}.c_fc.weight"),
            format!("{prefix}.mlp.c_fc.weight"),
            format!("{prefix}.ffn.c_fc.weight"),
            format!("{prefix}.feed_forward.c_fc.weight"),
            format!("{prefix}.feed_forward.mlp.c_fc.weight"),
            format!("{prefix}.w1w3.weight"),
            format!("{prefix}.mlp.w1w3.weight"),
            format!("{prefix}.ffn.w1w3.weight"),
            format!("{prefix}.feed_forward.w1w3.weight"),
            format!("{prefix}.feed_forward.mlp.w1w3.weight"),
        ]
    })
}

pub fn qwen_dense_packed_ffn_gate_up_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_packed_ffn_gate_up_weight_names(prefix))
}

pub fn qwen_dense_packed_ffn_up_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_up_gate.weight"),
            format!("{prefix}.up_gate_proj.weight"),
            format!("{prefix}.mlp.up_gate_proj.weight"),
            format!("{prefix}.ffn.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.mlp.up_gate_proj.weight"),
            format!("{prefix}.up_gate.weight"),
            format!("{prefix}.mlp.up_gate.weight"),
            format!("{prefix}.ffn.up_gate.weight"),
            format!("{prefix}.feed_forward.up_gate.weight"),
            format!("{prefix}.feed_forward.mlp.up_gate.weight"),
            format!("{prefix}.w3w1.weight"),
            format!("{prefix}.mlp.w3w1.weight"),
            format!("{prefix}.ffn.w3w1.weight"),
            format!("{prefix}.feed_forward.w3w1.weight"),
            format!("{prefix}.feed_forward.mlp.w3w1.weight"),
        ]
    })
}

pub fn qwen_dense_packed_ffn_up_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_packed_ffn_up_gate_weight_names(prefix))
}

pub fn qwen_phi_packed_qkv_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_qkv_weight_names(prefix)
}

pub fn qwen_phi_packed_qkv_bias_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_qkv_bias_names(prefix)
}

pub fn qwen_phi_packed_ffn_gate_up_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_ffn_gate_up_weight_names(prefix)
}

pub fn qwen_phi_packed_ffn_up_gate_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_ffn_up_gate_weight_names(prefix)
}

pub fn qwen_dense_attention_head_norm_weight_names(prefix: &str, suffix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_{suffix}_norm.weight"),
            format!("{prefix}.attn_{suffix}_layernorm.weight"),
            format!("{prefix}.attn_{suffix}_layer_norm.weight"),
            format!("{prefix}.{suffix}_norm.weight"),
            format!("{prefix}.{suffix}_layernorm.weight"),
            format!("{prefix}.{suffix}_layer_norm.weight"),
            format!("{prefix}.self_attn.{suffix}_norm.weight"),
            format!("{prefix}.self_attn.{suffix}_layernorm.weight"),
            format!("{prefix}.self_attn.{suffix}_layer_norm.weight"),
            format!("{prefix}.self_attention.{suffix}_norm.weight"),
            format!("{prefix}.self_attention.{suffix}_layernorm.weight"),
            format!("{prefix}.self_attention.{suffix}_layer_norm.weight"),
            format!("{prefix}.attention.{suffix}_norm.weight"),
            format!("{prefix}.attention.{suffix}_layernorm.weight"),
            format!("{prefix}.attention.{suffix}_layer_norm.weight"),
        ]
    })
}

pub fn qwen_dense_ffn_weight_names(prefix: &str, kind: &str) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate", "gate_proj", "w1"),
        "up" => ("ffn_up", "up_proj", "w3"),
        "down" => ("ffn_down", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        let mut names = vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.{hf}.weight"),
            format!("{prefix}.mlp.{hf}.weight"),
            format!("{prefix}.ffn.{hf}.weight"),
            format!("{prefix}.feed_forward.{hf}.weight"),
            format!("{prefix}.feed_forward.mlp.{hf}.weight"),
            format!("{prefix}.feed_forward.{llama}.weight"),
            format!("{prefix}.mlp.{llama}.weight"),
            format!("{prefix}.ffn.{llama}.weight"),
            format!("{prefix}.feed_forward.mlp.{llama}.weight"),
            format!("{prefix}.{llama}.weight"),
        ];
        if kind == "down" {
            names.extend([
                format!("{prefix}.dense_4h_to_h.weight"),
                format!("{prefix}.mlp.dense_4h_to_h.weight"),
                format!("{prefix}.ffn.dense_4h_to_h.weight"),
                format!("{prefix}.feed_forward.dense_4h_to_h.weight"),
                format!("{prefix}.feed_forward.mlp.dense_4h_to_h.weight"),
                format!("{prefix}.fc2.weight"),
                format!("{prefix}.mlp.fc2.weight"),
                format!("{prefix}.ffn.fc2.weight"),
                format!("{prefix}.feed_forward.fc2.weight"),
                format!("{prefix}.feed_forward.mlp.fc2.weight"),
                format!("{prefix}.mlp.c_proj.weight"),
                format!("{prefix}.ffn.c_proj.weight"),
                format!("{prefix}.feed_forward.c_proj.weight"),
                format!("{prefix}.feed_forward.mlp.c_proj.weight"),
                format!("{prefix}.mlp.proj.weight"),
                format!("{prefix}.ffn.proj.weight"),
                format!("{prefix}.feed_forward.proj.weight"),
                format!("{prefix}.feed_forward.mlp.proj.weight"),
            ]);
        }
        names
    })
}

pub fn qwen_dense_ffn_bias_names(prefix: &str, kind: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_ffn_weight_names(prefix, kind))
}

fn weight_aliases_to_bias_names(names: Vec<String>) -> Vec<String> {
    names
        .into_iter()
        .map(|name| {
            name.strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.bias"))
                .unwrap_or_else(|| format!("{name}.bias"))
        })
        .collect()
}

pub fn qwen_moe_router_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_inp.weight"),
            format!("{prefix}.router.weight"),
            format!("{prefix}.gate.weight"),
            format!("{prefix}.mlp.router.weight"),
            format!("{prefix}.mlp.gate.weight"),
            format!("{prefix}.mlp.moe.router.weight"),
            format!("{prefix}.mlp.moe.gate.weight"),
            format!("{prefix}.mlp.block_sparse_moe.router.weight"),
            format!("{prefix}.mlp.block_sparse_moe.gate.weight"),
            format!("{prefix}.moe.router.weight"),
            format!("{prefix}.moe.gate.weight"),
            format!("{prefix}.block_sparse_moe.router.weight"),
            format!("{prefix}.block_sparse_moe.gate.weight"),
            format!("{prefix}.feed_forward.router.weight"),
            format!("{prefix}.feed_forward.gate.weight"),
            format!("{prefix}.feed_forward.moe.router.weight"),
            format!("{prefix}.feed_forward.moe.gate.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.router.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.gate.weight"),
        ]
    })
}

pub fn qwen_moe_router_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_router_weight_names(prefix))
}

pub fn qwen_moe_packed_expert_weight_names(prefix: &str, kind: &str) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate_exps", "gate_proj", "w1"),
        "up" => ("ffn_up_exps", "up_proj", "w3"),
        "down" => ("ffn_down_exps", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}_exps.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.experts.{hf}.weight"),
            format!("{prefix}.mlp.experts.{hf}.weight"),
            format!("{prefix}.mlp.moe.experts.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{hf}.weight"),
            format!("{prefix}.moe.experts.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{hf}.weight"),
            format!("{prefix}.feed_forward.experts.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{hf}.weight"),
            format!("{prefix}.experts.{llama}.weight"),
            format!("{prefix}.mlp.experts.{llama}.weight"),
            format!("{prefix}.mlp.moe.experts.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{llama}.weight"),
            format!("{prefix}.moe.experts.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{llama}.weight"),
            format!("{prefix}.feed_forward.experts.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{llama}.weight"),
        ]
    })
}

pub fn qwen_moe_packed_expert_bias_names(prefix: &str, kind: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_packed_expert_weight_names(prefix, kind))
}

pub fn qwen_moe_packed_expert_gate_up_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_up_exps.weight"),
            format!("{prefix}.experts.gate_up_proj.weight"),
            format!("{prefix}.mlp.experts.gate_up_proj.weight"),
            format!("{prefix}.mlp.moe.experts.gate_up_proj.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.gate_up_proj.weight"),
            format!("{prefix}.moe.experts.gate_up_proj.weight"),
            format!("{prefix}.block_sparse_moe.experts.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.experts.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.moe.experts.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.gate_up_proj.weight"),
            format!("{prefix}.experts.w1w3.weight"),
            format!("{prefix}.mlp.experts.w1w3.weight"),
            format!("{prefix}.mlp.moe.experts.w1w3.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.w1w3.weight"),
            format!("{prefix}.moe.experts.w1w3.weight"),
            format!("{prefix}.block_sparse_moe.experts.w1w3.weight"),
            format!("{prefix}.feed_forward.experts.w1w3.weight"),
            format!("{prefix}.feed_forward.moe.experts.w1w3.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.w1w3.weight"),
        ]
    })
}

pub fn qwen_moe_packed_expert_gate_up_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_packed_expert_gate_up_weight_names(prefix))
}

pub fn qwen_moe_packed_expert_up_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_up_gate_exps.weight"),
            format!("{prefix}.experts.up_gate_proj.weight"),
            format!("{prefix}.mlp.experts.up_gate_proj.weight"),
            format!("{prefix}.mlp.moe.experts.up_gate_proj.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.up_gate_proj.weight"),
            format!("{prefix}.moe.experts.up_gate_proj.weight"),
            format!("{prefix}.block_sparse_moe.experts.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.experts.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.moe.experts.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.up_gate_proj.weight"),
            format!("{prefix}.experts.w3w1.weight"),
            format!("{prefix}.mlp.experts.w3w1.weight"),
            format!("{prefix}.mlp.moe.experts.w3w1.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.w3w1.weight"),
            format!("{prefix}.moe.experts.w3w1.weight"),
            format!("{prefix}.block_sparse_moe.experts.w3w1.weight"),
            format!("{prefix}.feed_forward.experts.w3w1.weight"),
            format!("{prefix}.feed_forward.moe.experts.w3w1.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.w3w1.weight"),
        ]
    })
}

pub fn qwen_moe_packed_expert_up_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_packed_expert_up_gate_weight_names(prefix))
}

pub fn qwen_moe_per_expert_gate_up_weight_names(prefix: &str, expert: u64) -> Vec<String> {
    qwen_moe_per_expert_packed_gate_up_weight_names(
        prefix,
        expert,
        "ffn_gate_up",
        "gate_up_proj",
        "w1w3",
    )
}

pub fn qwen_moe_per_expert_gate_up_bias_names(prefix: &str, expert: u64) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_per_expert_gate_up_weight_names(prefix, expert))
}

pub fn qwen_moe_per_expert_up_gate_weight_names(prefix: &str, expert: u64) -> Vec<String> {
    qwen_moe_per_expert_packed_gate_up_weight_names(
        prefix,
        expert,
        "ffn_up_gate",
        "up_gate_proj",
        "w3w1",
    )
}

pub fn qwen_moe_per_expert_up_gate_bias_names(prefix: &str, expert: u64) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_per_expert_up_gate_weight_names(prefix, expert))
}

fn qwen_moe_per_expert_packed_gate_up_weight_names(
    prefix: &str,
    expert: u64,
    canonical: &str,
    hf: &str,
    llama: &str,
) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.{expert}.weight"),
            format!("{prefix}.{canonical}_exps.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{hf}.weight"),
            format!("{prefix}.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{llama}.weight"),
            format!("{prefix}.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{llama}.{expert}.weight"),
        ]
    })
}

pub fn qwen_moe_per_expert_weight_names(prefix: &str, kind: &str, expert: u64) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate", "gate_proj", "w1"),
        "up" => ("ffn_up", "up_proj", "w3"),
        "down" => ("ffn_down", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}.{expert}.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{hf}.weight"),
            format!("{prefix}.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{llama}.weight"),
            format!("{prefix}.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{llama}.{expert}.weight"),
        ]
    })
}

pub fn qwen_moe_per_expert_bias_names(prefix: &str, kind: &str, expert: u64) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_per_expert_weight_names(prefix, kind, expert))
}

pub fn qwen_moe_shared_expert_weight_names(prefix: &str, kind: &str) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate_shexp", "gate_proj", "w1"),
        "up" => ("ffn_up_shexp", "up_proj", "w3"),
        "down" => ("ffn_down_shexp", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}_shexp.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.shared_expert.{hf}.weight"),
            format!("{prefix}.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.shared_expert.{llama}.weight"),
            format!("{prefix}.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{llama}.weight"),
        ]
    })
}

pub fn qwen_moe_shared_expert_bias_names(prefix: &str, kind: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_weight_names(prefix, kind))
}

pub fn qwen_moe_shared_expert_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_inp_shexp.weight"),
            format!("{prefix}.shared_expert_gate.weight"),
            format!("{prefix}.shared_experts_gate.weight"),
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            format!("{prefix}.mlp.shared_experts_gate.weight"),
            format!("{prefix}.mlp.moe.shared_expert_gate.weight"),
            format!("{prefix}.mlp.moe.shared_experts_gate.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert_gate.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts_gate.weight"),
            format!("{prefix}.moe.shared_expert_gate.weight"),
            format!("{prefix}.moe.shared_experts_gate.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert_gate.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts_gate.weight"),
            format!("{prefix}.feed_forward.shared_expert_gate.weight"),
            format!("{prefix}.feed_forward.shared_experts_gate.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert_gate.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts_gate.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert_gate.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts_gate.weight"),
        ]
    })
}

pub fn qwen_moe_shared_expert_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_gate_weight_names(prefix))
}

pub fn qwen_moe_shared_expert_gate_up_weight_names(prefix: &str) -> Vec<String> {
    qwen_moe_shared_expert_packed_gate_up_weight_names(
        prefix,
        "ffn_gate_up_shexp",
        "gate_up_proj",
        "w1w3",
    )
}

pub fn qwen_moe_shared_expert_gate_up_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_gate_up_weight_names(prefix))
}

pub fn qwen_moe_shared_expert_up_gate_weight_names(prefix: &str) -> Vec<String> {
    qwen_moe_shared_expert_packed_gate_up_weight_names(
        prefix,
        "ffn_up_gate_shexp",
        "up_gate_proj",
        "w3w1",
    )
}

pub fn qwen_moe_shared_expert_up_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_up_gate_weight_names(prefix))
}

fn qwen_moe_shared_expert_packed_gate_up_weight_names(
    prefix: &str,
    canonical: &str,
    hf: &str,
    llama: &str,
) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.shared_expert.{hf}.weight"),
            format!("{prefix}.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.shared_expert.{llama}.weight"),
            format!("{prefix}.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{llama}.weight"),
        ]
    })
}

fn layer_prefix_aliases<F>(prefix: &str, build: F) -> Vec<String>
where
    F: Fn(&str) -> Vec<String>,
{
    let mut names = Vec::new();
    for prefix in layer_prefix_variants(prefix) {
        for name in build(&prefix) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

fn layer_prefix_variants(prefix: &str) -> Vec<String> {
    let mut prefixes = vec![prefix.to_string()];
    let Some(layer) = prefix.strip_prefix("blk.") else {
        return prefixes;
    };
    if layer.is_empty() || layer.contains('.') {
        return prefixes;
    }
    for alias in [
        format!("model.layers.{layer}"),
        format!("language_model.model.layers.{layer}"),
        format!("language_model.layers.{layer}"),
        format!("layers.{layer}"),
        format!("model.decoder.layers.{layer}"),
        format!("language_model.model.decoder.layers.{layer}"),
        format!("language_model.decoder.layers.{layer}"),
        format!("decoder.layers.{layer}"),
        format!("encoder.layers.{layer}"),
        format!("transformer.encoder.layers.{layer}"),
        format!("transformer.decoder.layers.{layer}"),
        format!("transformer.layers.{layer}"),
        format!("transformer.blocks.{layer}"),
        format!("model.transformer.layers.{layer}"),
        format!("model.transformer.blocks.{layer}"),
        format!("transformer.h.{layer}"),
        format!("model.transformer.h.{layer}"),
        format!("h.{layer}"),
        format!("model.h.{layer}"),
        format!("gpt_neox.layers.{layer}"),
        format!("model.gpt_neox.layers.{layer}"),
    ] {
        if !prefixes.contains(&alias) {
            prefixes.push(alias);
        }
    }
    prefixes
}

fn attention_projection_alias_parts(
    suffix: &str,
) -> Option<(&'static str, &'static str, &'static str)> {
    match suffix {
        "q" => Some(("attn_q", "q_proj", "wq")),
        "k" => Some(("attn_k", "k_proj", "wk")),
        "v" => Some(("attn_v", "v_proj", "wv")),
        "output" => Some(("attn_output", "o_proj", "wo")),
        _ => None,
    }
}

fn qwen_ssm_metadata_feature(key: &str) -> Option<&'static str> {
    let lower = key.to_ascii_lowercase();
    if lower.contains("ssm") {
        Some("unsupported feature SSM")
    } else if lower.contains("mamba") {
        Some("unsupported feature Mamba/SSM")
    } else if lower.contains("delta") {
        Some("unsupported feature DeltaNet/recurrent decoder")
    } else if lower.contains("conv1d") || lower.contains("conv_1d") {
        Some("unsupported feature SSM convolution")
    } else if lower.contains("recurrent") {
        Some("unsupported feature recurrent decoder")
    } else {
        None
    }
}

fn qwen_ssm_tensor_feature(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("ssm") {
        Some("unsupported feature SSM")
    } else if lower.contains("mamba") {
        Some("unsupported feature Mamba/SSM")
    } else if lower.contains("delta") {
        Some("unsupported feature DeltaNet/recurrent decoder")
    } else if lower.contains("conv1d") || lower.contains("conv_1d") {
        Some("unsupported feature SSM convolution")
    } else if lower.contains("time_mix") || lower.contains("time-mix") {
        Some("unsupported feature time-mix recurrent decoder")
    } else if lower.contains("recurrent") {
        Some("unsupported feature recurrent decoder")
    } else {
        None
    }
}

fn qwen_attention_recurrent_layers(
    gguf: &GgufFile,
    prefix: &str,
    block_count: u32,
) -> Result<Option<Vec<bool>>> {
    let key = format!("{prefix}.attention.recurrent_layers");
    let Some(value) = gguf.metadata.get(&key) else {
        return Ok(None);
    };
    let MetadataValue::Array(values) = value else {
        bail!("GGUF metadata {key} must be an array of booleans or integers");
    };
    if values.len()
        != usize::try_from(block_count).context("qwen block_count does not fit usize")?
    {
        bail!(
            "GGUF metadata {key} must contain {block_count} entries, got {}",
            values.len()
        );
    }
    values
        .iter()
        .map(|value| match value {
            MetadataValue::Bool(value) => Ok(*value),
            MetadataValue::Uint8(value) => Ok(*value != 0),
            MetadataValue::Int8(value) => Ok(*value != 0),
            MetadataValue::Uint16(value) => Ok(*value != 0),
            MetadataValue::Int16(value) => Ok(*value != 0),
            MetadataValue::Uint32(value) => Ok(*value != 0),
            MetadataValue::Int32(value) => Ok(*value != 0),
            MetadataValue::Uint64(value) => Ok(*value != 0),
            MetadataValue::Int64(value) => Ok(*value != 0),
            _ => bail!("GGUF metadata {key} must be an array of booleans or integers"),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn qwen_ssm_layer_tensors_present_with<F>(has_tensor: F, prefix: &str) -> bool
where
    F: Fn(&str) -> bool,
{
    let legacy_in = qwen_ssm_in_weight_names(prefix)
        .iter()
        .any(|name| has_tensor(name));
    let optimized_in = qwen_ssm_qkv_weight_names(prefix)
        .iter()
        .any(|name| has_tensor(name))
        && qwen_ssm_gate_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name));
    (legacy_in || optimized_in)
        && qwen_ssm_conv1d_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_dt_bias_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_a_names(prefix).iter().any(|name| has_tensor(name))
        && qwen_ssm_ba_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_norm_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_out_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
}

fn qwen_ssm_layer_tensors_present_in(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    qwen_ssm_layer_tensors_present_with(|name| tensors.contains_key(name), prefix)
}

#[derive(Clone, Copy)]
struct QwenSsmDims {
    conv_kernel: u64,
    time_step_rank: u64,
    value_dim: u64,
    conv_dim: u64,
    qkvz_dim: u64,
    ba_dim: u64,
    head_v_dim: u64,
}

fn qwen_ssm_dims(config: &QwenGgufConfig, prefix: &str) -> Result<Option<QwenSsmDims>> {
    if !config.recurrent_ssm_tensor_layout {
        return Ok(None);
    }
    let metadata_prefix = &config.architecture;
    let Some(conv_kernel) = config
        .ssm_conv_kernel
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.conv_kernel");
    };
    let Some(inner_size) = config
        .ssm_inner_size
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.inner_size");
    };
    let Some(state_size) = config
        .ssm_state_size
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.state_size");
    };
    let Some(time_step_rank) = config
        .ssm_time_step_rank
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.time_step_rank");
    };
    let Some(group_count) = config
        .ssm_group_count
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.group_count");
    };
    if time_step_rank % group_count != 0 {
        bail!(
            "SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.time_step_rank={time_step_rank} to be divisible by {metadata_prefix}.ssm.group_count={group_count}"
        );
    }
    if inner_size % time_step_rank != 0 {
        bail!(
            "SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.inner_size={inner_size} to be divisible by {metadata_prefix}.ssm.time_step_rank={time_step_rank}"
        );
    }
    let head_v_dim = inner_size / time_step_rank;
    let key_dim = state_size
        .checked_mul(group_count)
        .context("SSM key dimension overflows u64")?;
    let value_dim = head_v_dim
        .checked_mul(time_step_rank)
        .context("SSM value dimension overflows u64")?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_dim))
        .context("SSM convolution dimension overflows u64")?;
    let qkvz_dim = key_dim
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_dim.checked_mul(2)?))
        .context("SSM qkvz dimension overflows u64")?;
    let ba_dim = time_step_rank
        .checked_mul(2)
        .context("SSM beta/alpha dimension overflows u64")?;
    Ok(Some(QwenSsmDims {
        conv_kernel,
        time_step_rank,
        value_dim,
        conv_dim,
        qkvz_dim,
        ba_dim,
        head_v_dim,
    }))
}

#[derive(Clone, Debug, Serialize)]
pub struct QwenTensorValidation {
    pub valid: bool,
    pub required_tensors: usize,
    pub optional_tensors_present: usize,
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub errors: Vec<String>,
}

fn validate_qwen_tensors(gguf: &GgufFile, config: &QwenGgufConfig) -> QwenTensorValidation {
    let tensors = gguf
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut validator = QwenTensorValidator {
        tensors,
        errors: Vec::new(),
        required_tensors: 0,
        optional_tensors_present: 0,
    };

    let embed = u64::from(config.embedding_length);
    let vocab = config.vocab_size.map(u64::from);
    let block_count = config.block_count;
    let head_count = u64::from(config.attention_head_count);
    let kv_head_count = u64::from(config.attention_head_count_kv);
    if head_count == 0 {
        validator
            .errors
            .push("qwen attention.head_count must be greater than zero".to_string());
    }
    if kv_head_count == 0 {
        validator
            .errors
            .push("qwen attention.head_count_kv must be greater than zero".to_string());
    }
    if block_count == 0 {
        validator
            .errors
            .push("qwen block_count must be greater than zero".to_string());
    }
    let qk_head_dim = config.attention_key_head_dim().map(u64::from).unwrap_or(0);
    let v_head_dim = config
        .attention_value_head_dim()
        .map(u64::from)
        .unwrap_or(0);
    if qk_head_dim == 0 || v_head_dim == 0 {
        validator.errors.push(format!(
            "qwen attention key/value head dimensions could not be determined from embedding_length {embed}, attention.head_count {head_count}, attention.key_length {:?}, attention.value_length {:?}",
            config.attention_key_length, config.attention_value_length
        ));
    }
    let q_dim = qk_head_dim.saturating_mul(head_count);
    let k_dim = qk_head_dim.saturating_mul(kv_head_count);
    let v_dim = v_head_dim.saturating_mul(kv_head_count);
    let attention_output_dim = v_head_dim.saturating_mul(head_count);
    let ff = config.feed_forward_length.map(u64::from);
    let expert_count = config.expert_count.map(u64::from);
    let expert_ff = config.expert_feed_forward_length.map(u64::from);
    if config.expert_count.is_some() && config.expert_used_count.is_none() {
        validator
            .errors
            .push("qwen MoE metadata missing expert_used_count".to_string());
    }
    if let (Some(used), Some(total)) = (config.expert_used_count, config.expert_count)
        && (used == 0 || used > total)
    {
        validator.errors.push(format!(
            "qwen MoE expert_used_count {used} must be in 1..={total}"
        ));
    }

    validator.require_one_of(
        &qwen_dense_token_embd_weight_names(),
        embedding_matrix_rules(embed, vocab),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_dense_output_norm_weight_names(),
        vec![ShapeRule::exact([embed])],
        DTypePolicy::Any,
    );
    validator.optional_one_of(
        &qwen_dense_output_weight_names(),
        embedding_matrix_rules(embed, vocab),
        DTypePolicy::Matrix,
    );
    if let Some(vocab) = vocab {
        validator.optional_one_of(
            &qwen_dense_output_bias_names(),
            vec![ShapeRule::exact([vocab])],
            DTypePolicy::Any,
        );
    }

    for layer in 0..block_count {
        let prefix = format!("blk.{layer}");
        let uses_mla_attention = qwen_mla_attention_tensors_present_in(&validator.tensors, &prefix);
        let uses_recurrent_ssm = qwen_ssm_layer_tensors_present_in(&validator.tensors, &prefix);
        validator.require_one_of(
            &qwen_dense_attention_norm_weight_names(&prefix),
            vec![ShapeRule::exact([embed])],
            DTypePolicy::Any,
        );
        validator.require_one_of(
            &qwen_dense_ffn_norm_weight_names(&prefix),
            vec![ShapeRule::exact([embed])],
            DTypePolicy::Any,
        );
        if uses_recurrent_ssm {
            require_ssm_layer_tensors(&mut validator, config, &prefix, embed);
        } else if uses_mla_attention {
            require_mla_attention_tensors(&mut validator, config, &prefix, embed, head_count);
        } else if !qkv_split_tensors_present(&validator.tensors, &prefix)
            && let Some(packed_name) =
                dense_packed_qkv_name(&validator.tensors, &prefix, embed, q_dim, k_dim, v_dim)
        {
            validator.require(
                &packed_name,
                matrix_rules(embed, q_dim.saturating_add(k_dim).saturating_add(v_dim)),
                DTypePolicy::Matrix,
            );
        } else {
            let gated_q_name =
                dense_gated_attention_q_name(&validator.tensors, &prefix, embed, q_dim);
            if let Some(gated_q_name) = gated_q_name {
                validator.require(
                    &gated_q_name,
                    matrix_rules(embed, q_dim.saturating_mul(2)),
                    DTypePolicy::Matrix,
                );
            } else {
                validator.require_one_of(
                    &qwen_dense_attention_weight_names(&prefix, "q"),
                    matrix_rules(embed, q_dim),
                    DTypePolicy::Matrix,
                );
            }
            validator.require_one_of(
                &qwen_dense_attention_weight_names(&prefix, "k"),
                matrix_rules(embed, k_dim),
                DTypePolicy::Matrix,
            );
            validator.require_one_of(
                &qwen_dense_attention_weight_names(&prefix, "v"),
                matrix_rules(embed, v_dim),
                DTypePolicy::Matrix,
            );
        }
        if !uses_recurrent_ssm {
            validator.require_one_of(
                &qwen_dense_attention_weight_names(&prefix, "output"),
                matrix_rules(attention_output_dim, embed),
                DTypePolicy::Matrix,
            );
        }
        if config.expert_count.is_some() && moe_router_tensor_present(&validator.tensors, &prefix) {
            let packed_expert_moe_complete =
                moe_packed_expert_tensors_complete(&validator.tensors, &prefix);
            let per_expert_packed_gate_up_moe_present =
                moe_any_per_expert_packed_gate_up_tensor_present(
                    &validator.tensors,
                    &prefix,
                    expert_count,
                );
            let per_expert_packed_gate_up_moe_complete = expert_count.is_some_and(|experts| {
                moe_per_expert_packed_gate_up_tensors_complete(&validator.tensors, &prefix, experts)
            });
            let use_per_expert_packed_gate_up_moe = !packed_expert_moe_complete
                && (per_expert_packed_gate_up_moe_present
                    || per_expert_packed_gate_up_moe_complete);
            let use_per_expert_moe = !packed_expert_moe_complete
                && !use_per_expert_packed_gate_up_moe
                && moe_any_per_expert_tensor_present(&validator.tensors, &prefix, expert_count);
            validator.require_one_of(
                &qwen_moe_router_weight_names(&prefix),
                expert_router_rules(embed, expert_count),
                DTypePolicy::Matrix,
            );
            if let Some(experts) = expert_count {
                validator.optional_one_of(
                    &qwen_moe_router_bias_names(&prefix),
                    vec![ShapeRule::exact([experts])],
                    DTypePolicy::Any,
                );
            }
            if use_per_expert_packed_gate_up_moe {
                require_per_expert_packed_gate_up_moe_tensors(
                    &mut validator,
                    &prefix,
                    embed,
                    expert_ff,
                    expert_count,
                );
            } else if use_per_expert_moe {
                require_per_expert_moe_tensors(
                    &mut validator,
                    &prefix,
                    embed,
                    expert_ff,
                    expert_count,
                );
            } else if let Some(packed_gate_up_name) = moe_packed_expert_gate_up_name(
                &validator.tensors,
                &prefix,
                embed,
                expert_ff,
                expert_count,
            ) {
                let Some(expert_ff) = expert_ff else {
                    validator.errors.push(
                        "model metadata missing expert_feed_forward_length for packed MoE gate/up layout"
                            .to_string(),
                    );
                    continue;
                };
                validator.require(
                    &packed_gate_up_name,
                    expert_matrix_rules(embed, Some(expert_ff.saturating_mul(2)), expert_count),
                    DTypePolicy::Matrix,
                );
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "down"),
                    expert_matrix_rules(embed, Some(expert_ff), expert_count),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_packed_expert_gate_up_bias_names(&prefix)
                        .into_iter()
                        .chain(qwen_moe_packed_expert_up_gate_bias_names(&prefix))
                        .collect::<Vec<_>>(),
                    expert_bias_rules(expert_ff.saturating_mul(2), expert_count),
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &qwen_moe_packed_expert_bias_names(&prefix, "down"),
                    expert_bias_rules(embed, expert_count),
                    DTypePolicy::Any,
                );
            } else {
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "gate"),
                    expert_matrix_rules(embed, expert_ff, expert_count),
                    DTypePolicy::Matrix,
                );
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "up"),
                    expert_matrix_rules(embed, expert_ff, expert_count),
                    DTypePolicy::Matrix,
                );
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "down"),
                    expert_matrix_rules(embed, expert_ff, expert_count),
                    DTypePolicy::Matrix,
                );
                if let Some(expert_ff) = expert_ff {
                    validator.optional_one_of(
                        &qwen_moe_packed_expert_bias_names(&prefix, "gate"),
                        expert_bias_rules(expert_ff, expert_count),
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_packed_expert_bias_names(&prefix, "up"),
                        expert_bias_rules(expert_ff, expert_count),
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_packed_expert_bias_names(&prefix, "down"),
                        expert_bias_rules(embed, expert_count),
                        DTypePolicy::Any,
                    );
                }
            }
            let shared = [
                qwen_moe_shared_expert_weight_names(&prefix, "gate"),
                qwen_moe_shared_expert_weight_names(&prefix, "up"),
                qwen_moe_shared_expert_weight_names(&prefix, "down"),
            ];
            let shared_expert_gate = qwen_moe_shared_expert_gate_weight_names(&prefix);
            let shared_expert_gate_present = shared_expert_gate
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()));
            let shared_split_present = shared
                .iter()
                .filter(|names| {
                    names
                        .iter()
                        .any(|name| validator.tensors.contains_key(name.as_str()))
                })
                .count();
            let shared_packed_gate_up = moe_shared_expert_packed_gate_up_weight_names(&prefix);
            let shared_packed_gate_up_present = shared_packed_gate_up
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()));
            let shared_down_present = shared[2]
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()));
            if shared_packed_gate_up_present {
                let Some(shared_ff) = ff.or(expert_ff) else {
                    validator.errors.push(
                        "model metadata missing feed_forward_length or expert_feed_forward_length for packed shared expert layout"
                            .to_string(),
                    );
                    continue;
                };
                if !shared_down_present {
                    let expected = [
                        shared_packed_gate_up.first().cloned().unwrap_or_default(),
                        shared[2].first().cloned().unwrap_or_default(),
                    ];
                    validator.errors.push(format!(
                        "layer {layer} has incomplete packed shared expert tensors; expected all of {}",
                        expected.join(", ")
                    ));
                }
                validator.optional_one_of(
                    &shared_packed_gate_up,
                    matrix_rules(embed, shared_ff.saturating_mul(2)),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &moe_shared_expert_packed_gate_up_bias_names(&prefix),
                    vec![ShapeRule::exact([shared_ff.saturating_mul(2)])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &shared[2],
                    feed_forward_matrix_rules(embed, Some(shared_ff)),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_shared_expert_bias_names(&prefix, "down"),
                    vec![ShapeRule::exact([embed])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &shared_expert_gate,
                    matrix_rules(embed, 1),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_shared_expert_gate_bias_names(&prefix),
                    vec![ShapeRule::exact([1])],
                    DTypePolicy::Any,
                );
            } else if shared_split_present != 0 && shared_split_present != shared.len() {
                let expected = shared
                    .iter()
                    .filter_map(|names| names.first())
                    .cloned()
                    .collect::<Vec<_>>();
                validator.errors.push(format!(
                    "layer {layer} has incomplete shared expert tensors; expected all of {}",
                    expected.join(", ")
                ));
            } else if shared_split_present == shared.len() {
                let shared_ff = ff.or(expert_ff);
                validator.optional_one_of(
                    &shared[0],
                    feed_forward_matrix_rules(embed, shared_ff),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &shared[1],
                    feed_forward_matrix_rules(embed, shared_ff),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &shared[2],
                    feed_forward_matrix_rules(embed, shared_ff),
                    DTypePolicy::Matrix,
                );
                if let Some(shared_ff) = shared_ff {
                    validator.optional_one_of(
                        &qwen_moe_shared_expert_bias_names(&prefix, "gate"),
                        vec![ShapeRule::exact([shared_ff])],
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_shared_expert_bias_names(&prefix, "up"),
                        vec![ShapeRule::exact([shared_ff])],
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_shared_expert_bias_names(&prefix, "down"),
                        vec![ShapeRule::exact([embed])],
                        DTypePolicy::Any,
                    );
                }
                validator.optional_one_of(
                    &shared_expert_gate,
                    matrix_rules(embed, 1),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_shared_expert_gate_bias_names(&prefix),
                    vec![ShapeRule::exact([1])],
                    DTypePolicy::Any,
                );
            } else if shared_expert_gate_present {
                validator.errors.push(format!(
                    "layer {layer} has shared expert gate tensor without shared expert tensors; expected shared expert gate/up/down tensors or packed shared expert gate_up plus down"
                ));
            }
        } else if let Some(packed_name) =
            dense_packed_ffn_name(&validator.tensors, &prefix, embed, ff)
        {
            let Some(ff) = ff else {
                validator.errors.push(
                    "model metadata missing feed_forward_length for packed FFN layout".to_string(),
                );
                continue;
            };
            validator.require(
                &packed_name,
                matrix_rules(embed, ff.saturating_mul(2)),
                DTypePolicy::Matrix,
            );
            validator.optional_one_of(
                &qwen_dense_packed_ffn_gate_up_bias_names(&prefix)
                    .into_iter()
                    .chain(qwen_dense_packed_ffn_up_gate_bias_names(&prefix))
                    .collect::<Vec<_>>(),
                vec![ShapeRule::exact([ff.saturating_mul(2)])],
                DTypePolicy::Any,
            );
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "down"),
                feed_forward_matrix_rules(embed, Some(ff)),
                DTypePolicy::Matrix,
            );
            validator.optional_one_of(
                &qwen_dense_ffn_bias_names(&prefix, "down"),
                vec![ShapeRule::exact([embed])],
                DTypePolicy::Any,
            );
        } else {
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "gate"),
                feed_forward_matrix_rules(embed, ff),
                DTypePolicy::Matrix,
            );
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "up"),
                feed_forward_matrix_rules(embed, ff),
                DTypePolicy::Matrix,
            );
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "down"),
                feed_forward_matrix_rules(embed, ff),
                DTypePolicy::Matrix,
            );
            if let Some(ff) = ff {
                validator.optional_one_of(
                    &qwen_dense_ffn_bias_names(&prefix, "gate"),
                    vec![ShapeRule::exact([ff])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &qwen_dense_ffn_bias_names(&prefix, "up"),
                    vec![ShapeRule::exact([ff])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &qwen_dense_ffn_bias_names(&prefix, "down"),
                    vec![ShapeRule::exact([embed])],
                    DTypePolicy::Any,
                );
            }
        }

        if !uses_recurrent_ssm {
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "output"),
                vec![ShapeRule::exact([embed])],
                DTypePolicy::Any,
            );
        }

        if !uses_mla_attention && !uses_recurrent_ssm {
            let q_bias_rules = if dense_gated_attention_q_name(
                &validator.tensors,
                &prefix,
                embed,
                q_dim,
            )
            .is_some()
            {
                vec![ShapeRule::exact([q_dim.saturating_mul(2)])]
            } else {
                vec![ShapeRule::exact([q_dim])]
            };
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "q"),
                q_bias_rules,
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "k"),
                vec![ShapeRule::exact([k_dim])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "v"),
                vec![ShapeRule::exact([v_dim])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_head_norm_weight_names(&prefix, "q"),
                vec![ShapeRule::exact([qk_head_dim])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_head_norm_weight_names(&prefix, "k"),
                vec![ShapeRule::exact([qk_head_dim])],
                DTypePolicy::Any,
            );
            if qwen_dense_packed_qkv_bias_names(&prefix)
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()))
            {
                validator.optional_one_of(
                    &qwen_dense_packed_qkv_bias_names(&prefix),
                    vec![ShapeRule::exact([q_dim
                        .saturating_add(k_dim)
                        .saturating_add(v_dim)])],
                    DTypePolicy::Any,
                );
            }
        }
    }

    let total_tensor_bytes = gguf
        .tensors
        .iter()
        .filter_map(|tensor| tensor.byte_len().ok())
        .sum();
    QwenTensorValidation {
        valid: validator.errors.is_empty(),
        required_tensors: validator.required_tensors,
        optional_tensors_present: validator.optional_tensors_present,
        tensor_count: gguf.tensors.len(),
        total_tensor_bytes,
        errors: validator.errors,
    }
}

struct QwenTensorValidator<'a> {
    tensors: BTreeMap<&'a str, &'a TensorInfo>,
    errors: Vec<String>,
    required_tensors: usize,
    optional_tensors_present: usize,
}

impl QwenTensorValidator<'_> {
    fn require(&mut self, name: &str, rules: Vec<ShapeRule>, dtype_policy: DTypePolicy) {
        self.required_tensors += 1;
        self.check(name, &rules, dtype_policy, true);
    }

    fn require_one_of(
        &mut self,
        names: &[String],
        rules: Vec<ShapeRule>,
        dtype_policy: DTypePolicy,
    ) {
        self.required_tensors += 1;
        self.check_one_of(names, &rules, dtype_policy, true);
    }

    fn optional_one_of(
        &mut self,
        names: &[String],
        rules: Vec<ShapeRule>,
        dtype_policy: DTypePolicy,
    ) {
        if names
            .iter()
            .any(|name| self.tensors.contains_key(name.as_str()))
        {
            self.optional_tensors_present += 1;
            self.check_one_of(names, &rules, dtype_policy, false);
        }
    }

    fn check_one_of(
        &mut self,
        names: &[String],
        rules: &[ShapeRule],
        dtype_policy: DTypePolicy,
        required: bool,
    ) {
        let Some(name) = names
            .iter()
            .find(|name| self.tensors.contains_key(name.as_str()))
        else {
            if required {
                if let Some(primary) = names.first() {
                    if names.len() > 1 {
                        self.errors.push(format!(
                            "missing required tensor {primary}; accepted aliases: {}",
                            names[1..].join(", ")
                        ));
                    } else {
                        self.errors
                            .push(format!("missing required tensor {primary}"));
                    }
                }
            }
            return;
        };
        self.check(name, rules, dtype_policy, false);
    }

    fn check(
        &mut self,
        name: &str,
        rules: &[ShapeRule],
        dtype_policy: DTypePolicy,
        required: bool,
    ) {
        let Some(tensor) = self.tensors.get(name) else {
            if required {
                self.errors.push(format!("missing required tensor {name}"));
            }
            return;
        };

        if !rules.iter().any(|rule| rule.matches(&tensor.dimensions)) {
            self.errors.push(format!(
                "tensor {name} has shape {:?}; expected {}",
                tensor.dimensions,
                describe_shape_rules(rules)
            ));
        }

        if dtype_policy == DTypePolicy::Matrix
            && !tensor.dtype.is_quantized()
            && !matches!(
                tensor.dtype,
                GgufTensorType::F16 | GgufTensorType::BF16 | GgufTensorType::F32
            )
        {
            self.errors.push(format!(
                "matrix tensor {name} has dtype {}; CUDA accepts FP16/BF16/F32 or quantized matrix weights",
                tensor.dtype.label()
            ));
        }
    }
}

fn qkv_split_tensors_present(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    ["q", "k", "v"].iter().all(|suffix| {
        qwen_dense_attention_weight_names(prefix, suffix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn qwen_mla_attention_tensors_present_in(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
) -> bool {
    qwen_mla_q_a_weight_names(prefix)
        .iter()
        .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_q_a_norm_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_q_b_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_kv_a_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_kv_a_norm_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_kv_b_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
}

fn moe_router_tensor_present(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    qwen_moe_router_weight_names(prefix)
        .iter()
        .any(|name| tensors.contains_key(name.as_str()))
}

fn moe_packed_expert_tensors_complete(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    ["gate", "up", "down"].iter().all(|kind| {
        qwen_moe_packed_expert_weight_names(prefix, kind)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn moe_any_per_expert_tensor_present(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    experts: Option<u64>,
) -> bool {
    let Some(experts) = experts else {
        return false;
    };
    (0..experts).any(|expert| {
        ["gate", "up", "down"].iter().any(|kind| {
            qwen_moe_per_expert_weight_names(prefix, kind, expert)
                .iter()
                .any(|name| tensors.contains_key(name.as_str()))
        })
    })
}

fn moe_per_expert_packed_gate_up_weight_names(prefix: &str, expert: u64) -> Vec<String> {
    let mut names = qwen_moe_per_expert_gate_up_weight_names(prefix, expert);
    for name in qwen_moe_per_expert_up_gate_weight_names(prefix, expert) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_per_expert_packed_gate_up_bias_names(prefix: &str, expert: u64) -> Vec<String> {
    let mut names = qwen_moe_per_expert_gate_up_bias_names(prefix, expert);
    for name in qwen_moe_per_expert_up_gate_bias_names(prefix, expert) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_shared_expert_packed_gate_up_weight_names(prefix: &str) -> Vec<String> {
    let mut names = qwen_moe_shared_expert_gate_up_weight_names(prefix);
    for name in qwen_moe_shared_expert_up_gate_weight_names(prefix) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_shared_expert_packed_gate_up_bias_names(prefix: &str) -> Vec<String> {
    let mut names = qwen_moe_shared_expert_gate_up_bias_names(prefix);
    for name in qwen_moe_shared_expert_up_gate_bias_names(prefix) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_any_per_expert_packed_gate_up_tensor_present(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    experts: Option<u64>,
) -> bool {
    let Some(experts) = experts else {
        return false;
    };
    (0..experts).any(|expert| {
        moe_per_expert_packed_gate_up_weight_names(prefix, expert)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn moe_per_expert_packed_gate_up_tensors_complete(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    experts: u64,
) -> bool {
    (0..experts).all(|expert| {
        moe_per_expert_packed_gate_up_weight_names(prefix, expert)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
            && qwen_moe_per_expert_weight_names(prefix, "down", expert)
                .iter()
                .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn require_per_expert_packed_gate_up_moe_tensors(
    validator: &mut QwenTensorValidator<'_>,
    prefix: &str,
    embed: u64,
    expert_ff: Option<u64>,
    experts: Option<u64>,
) {
    let Some(experts) = experts else {
        validator
            .errors
            .push("qwen MoE metadata missing expert_count".to_string());
        return;
    };
    let Some(expert_ff) = expert_ff else {
        validator.errors.push(
            "model metadata missing expert_feed_forward_length for per-expert packed MoE gate/up layout"
                .to_string(),
        );
        return;
    };
    for expert in 0..experts {
        validator.require_one_of(
            &moe_per_expert_packed_gate_up_weight_names(prefix, expert),
            matrix_rules(embed, expert_ff.saturating_mul(2)),
            DTypePolicy::Matrix,
        );
        validator.optional_one_of(
            &moe_per_expert_packed_gate_up_bias_names(prefix, expert),
            vec![ShapeRule::exact([expert_ff.saturating_mul(2)])],
            DTypePolicy::Any,
        );
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "down", expert),
            feed_forward_matrix_rules(embed, Some(expert_ff)),
            DTypePolicy::Matrix,
        );
        validator.optional_one_of(
            &qwen_moe_per_expert_bias_names(prefix, "down", expert),
            vec![ShapeRule::exact([embed])],
            DTypePolicy::Any,
        );
    }
}

fn require_per_expert_moe_tensors(
    validator: &mut QwenTensorValidator<'_>,
    prefix: &str,
    embed: u64,
    expert_ff: Option<u64>,
    experts: Option<u64>,
) {
    let Some(experts) = experts else {
        validator
            .errors
            .push("qwen MoE metadata missing expert_count".to_string());
        return;
    };
    for expert in 0..experts {
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "gate", expert),
            feed_forward_matrix_rules(embed, expert_ff),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "up", expert),
            feed_forward_matrix_rules(embed, expert_ff),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "down", expert),
            feed_forward_matrix_rules(embed, expert_ff),
            DTypePolicy::Matrix,
        );
        if let Some(expert_ff) = expert_ff {
            validator.optional_one_of(
                &qwen_moe_per_expert_bias_names(prefix, "gate", expert),
                vec![ShapeRule::exact([expert_ff])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_moe_per_expert_bias_names(prefix, "up", expert),
                vec![ShapeRule::exact([expert_ff])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_moe_per_expert_bias_names(prefix, "down", expert),
                vec![ShapeRule::exact([embed])],
                DTypePolicy::Any,
            );
        }
    }
}

fn require_mla_attention_tensors(
    validator: &mut QwenTensorValidator<'_>,
    config: &QwenGgufConfig,
    prefix: &str,
    embed: u64,
    head_count: u64,
) {
    let metadata_prefix = &config.architecture;
    let Some(q_lora) = config
        .attention_q_lora_rank
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.q_lora_rank"
        ));
        return;
    };
    let Some(kv_lora) = config
        .attention_kv_lora_rank
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.kv_lora_rank"
        ));
        return;
    };
    let Some(qk_rope) = config
        .attention_qk_rope_head_dim
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.qk_rope_head_dim"
        ));
        return;
    };
    let Some(qk_nope) = config.attention_qk_nope_head_dim.map(u64::from) else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.qk_nope_head_dim"
        ));
        return;
    };
    let Some(v_head_dim) = config
        .attention_v_head_dim
        .or(config.attention_value_length)
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.v_head_dim or {metadata_prefix}.attention.value_length"
        ));
        return;
    };

    let qk_head_dim = qk_nope.saturating_add(qk_rope);
    if qk_head_dim == 0 || head_count == 0 {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} has invalid head dimensions qk_nope={qk_nope}, qk_rope={qk_rope}, heads={head_count}"
        ));
        return;
    }
    let q_dim = qk_head_dim.saturating_mul(head_count);
    let kv_a_dim = kv_lora.saturating_add(qk_rope);
    let kv_b_dim = head_count.saturating_mul(qk_nope.saturating_add(v_head_dim));

    validator.require_one_of(
        &qwen_mla_q_a_weight_names(prefix),
        matrix_rules(embed, q_lora),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_mla_q_a_norm_weight_names(prefix),
        vec![ShapeRule::exact([q_lora])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_mla_q_b_weight_names(prefix),
        matrix_rules(q_lora, q_dim),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_mla_kv_a_weight_names(prefix),
        matrix_rules(embed, kv_a_dim),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_mla_kv_a_norm_weight_names(prefix),
        vec![ShapeRule::exact([kv_lora])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_mla_kv_b_weight_names(prefix),
        matrix_rules(kv_lora, kv_b_dim),
        DTypePolicy::Matrix,
    );
}

fn require_ssm_layer_tensors(
    validator: &mut QwenTensorValidator<'_>,
    config: &QwenGgufConfig,
    prefix: &str,
    embed: u64,
) {
    let dims = match qwen_ssm_dims(config, prefix) {
        Ok(Some(dims)) => dims,
        Ok(None) => {
            validator.errors.push(format!(
                "SSM tensor layout in {prefix} requires recurrent SSM config metadata"
            ));
            return;
        }
        Err(err) => {
            validator.errors.push(err.to_string());
            return;
        }
    };
    let legacy_in_present = qwen_ssm_in_weight_names(prefix)
        .iter()
        .any(|name| validator.tensors.contains_key(name.as_str()));
    if legacy_in_present {
        validator.require_one_of(
            &qwen_ssm_in_weight_names(prefix),
            matrix_rules(embed, dims.qkvz_dim),
            DTypePolicy::Matrix,
        );
    } else {
        validator.require_one_of(
            &qwen_ssm_qkv_weight_names(prefix),
            matrix_rules(embed, dims.conv_dim),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_ssm_gate_weight_names(prefix),
            matrix_rules(embed, dims.value_dim),
            DTypePolicy::Matrix,
        );
    }
    validator.require_one_of(
        &qwen_ssm_conv1d_weight_names(prefix),
        matrix_rules(dims.conv_kernel, dims.conv_dim),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_ssm_dt_bias_names(prefix),
        vec![ShapeRule::exact([dims.time_step_rank])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_ssm_a_names(prefix),
        vec![ShapeRule::exact([dims.time_step_rank])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_ssm_ba_weight_names(prefix),
        matrix_rules(embed, dims.ba_dim),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_ssm_norm_weight_names(prefix),
        vec![ShapeRule::exact([dims.head_v_dim])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_ssm_out_weight_names(prefix),
        matrix_rules(dims.value_dim, embed),
        DTypePolicy::Matrix,
    );
}

fn dense_packed_qkv_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    q_dim: u64,
    k_dim: u64,
    v_dim: u64,
) -> Option<String> {
    let qkv_dim = q_dim.checked_add(k_dim)?.checked_add(v_dim)?;
    qwen_dense_packed_qkv_weight_names(prefix)
        .into_iter()
        .find(|name| {
            tensor_matches_matrix_rules(tensors.get(name.as_str()).copied(), embed, qkv_dim)
        })
}

fn dense_gated_attention_q_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    q_dim: u64,
) -> Option<String> {
    let gated_q_dim = q_dim.checked_mul(2)?;
    qwen_dense_attention_weight_names(prefix, "q")
        .into_iter()
        .find(|name| {
            tensor_matches_matrix_rules(tensors.get(name.as_str()).copied(), embed, gated_q_dim)
        })
}

fn dense_packed_ffn_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    ff: Option<u64>,
) -> Option<String> {
    let ff = ff?;
    let packed_rows = ff.checked_mul(2)?;
    for name in qwen_dense_packed_ffn_gate_up_weight_names(prefix)
        .into_iter()
        .chain(qwen_dense_packed_ffn_up_gate_weight_names(prefix))
    {
        if tensor_matches_matrix_rules(tensors.get(name.as_str()).copied(), embed, packed_rows) {
            return Some(name);
        }
    }
    for prefix in layer_prefix_variants(prefix) {
        let gate_name = format!("{prefix}.ffn_gate.weight");
        let up_name = format!("{prefix}.ffn_up.weight");
        if !tensors.contains_key(up_name.as_str())
            && tensor_matches_matrix_rules(
                tensors.get(gate_name.as_str()).copied(),
                embed,
                packed_rows,
            )
        {
            return Some(gate_name);
        }
    }
    None
}

fn moe_packed_expert_gate_up_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    expert_ff: Option<u64>,
    expert_count: Option<u64>,
) -> Option<String> {
    let expert_ff = expert_ff?;
    let packed_rows = expert_ff.checked_mul(2)?;
    qwen_moe_packed_expert_gate_up_weight_names(prefix)
        .into_iter()
        .chain(qwen_moe_packed_expert_up_gate_weight_names(prefix))
        .find(|name| {
            expert_matrix_rules(embed, Some(packed_rows), expert_count)
                .iter()
                .any(|rule| {
                    tensors
                        .get(name.as_str())
                        .is_some_and(|tensor| rule.matches(&tensor.dimensions))
                })
        })
}

fn tensor_matches_matrix_rules(tensor: Option<&TensorInfo>, left: u64, right: u64) -> bool {
    let Some(tensor) = tensor else {
        return false;
    };
    matrix_rules(left, right)
        .iter()
        .any(|rule| rule.matches(&tensor.dimensions))
}

fn tensor_dimensions_match_matrix(dims: &[u64], left: u64, right: u64) -> bool {
    matrix_rules(left, right)
        .iter()
        .any(|rule| rule.matches(dims))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DTypePolicy {
    Any,
    Matrix,
}

#[derive(Clone, Debug)]
enum ShapeRule {
    Exact(Vec<u64>),
    RankWithDim { rank: usize, dim: u64 },
    RankWithDims { rank: usize, dims: Vec<u64> },
}

impl ShapeRule {
    fn exact<const N: usize>(dims: [u64; N]) -> Self {
        Self::Exact(dims.to_vec())
    }

    fn matches(&self, dims: &[u64]) -> bool {
        match self {
            Self::Exact(expected) => dims == expected,
            Self::RankWithDim { rank, dim } => dims.len() == *rank && dims.contains(dim),
            Self::RankWithDims {
                rank,
                dims: required,
            } => dims.len() == *rank && required.iter().all(|dim| dims.contains(dim)),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Exact(dims) => format!("{dims:?}"),
            Self::RankWithDim { rank, dim } => {
                format!("rank {rank} with one dimension equal to {dim}")
            }
            Self::RankWithDims { rank, dims } => {
                format!("rank {rank} with dimensions containing {dims:?}")
            }
        }
    }
}

fn describe_shape_rules(rules: &[ShapeRule]) -> String {
    rules
        .iter()
        .map(ShapeRule::describe)
        .collect::<Vec<_>>()
        .join(" or ")
}

fn matrix_rules(left: u64, right: u64) -> Vec<ShapeRule> {
    if left == 0 || right == 0 {
        vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: left.max(right),
        }]
    } else if left == right {
        vec![ShapeRule::exact([left, right])]
    } else {
        vec![
            ShapeRule::exact([left, right]),
            ShapeRule::exact([right, left]),
        ]
    }
}

fn embedding_matrix_rules(embed: u64, vocab: Option<u64>) -> Vec<ShapeRule> {
    match vocab {
        Some(vocab) => matrix_rules(embed, vocab),
        None => vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: embed,
        }],
    }
}

fn feed_forward_matrix_rules(embed: u64, ff: Option<u64>) -> Vec<ShapeRule> {
    match ff {
        Some(ff) => matrix_rules(embed, ff),
        None => vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: embed,
        }],
    }
}

fn expert_router_rules(embed: u64, experts: Option<u64>) -> Vec<ShapeRule> {
    match experts {
        Some(experts) => matrix_rules(embed, experts),
        None => vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: embed,
        }],
    }
}

fn expert_matrix_rules(embed: u64, ff: Option<u64>, experts: Option<u64>) -> Vec<ShapeRule> {
    match (ff, experts) {
        (Some(ff), Some(experts)) => vec![
            ShapeRule::exact([embed, ff, experts]),
            ShapeRule::exact([ff, embed, experts]),
        ],
        (Some(ff), None) => vec![ShapeRule::RankWithDims {
            rank: 3,
            dims: vec![embed, ff],
        }],
        (None, Some(experts)) => vec![ShapeRule::RankWithDims {
            rank: 3,
            dims: vec![embed, experts],
        }],
        (None, None) => vec![ShapeRule::RankWithDim {
            rank: 3,
            dim: embed,
        }],
    }
}

fn expert_bias_rules(len: u64, experts: Option<u64>) -> Vec<ShapeRule> {
    match experts {
        Some(experts) if len != experts => vec![
            ShapeRule::exact([len, experts]),
            ShapeRule::exact([experts, len]),
        ],
        Some(experts) => vec![ShapeRule::exact([len, experts])],
        None => vec![ShapeRule::RankWithDim { rank: 2, dim: len }],
    }
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

    Ok(GgufFile {
        path,
        mmap,
        version,
        alignment,
        data_start,
        metadata,
        tensors,
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

fn checked_add(left: u64, right: u64, context: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{context} overflows u64"))
}

fn array_len_as_u32(value: &MetadataValue) -> Option<u32> {
    match value {
        MetadataValue::Array(values) => u32::try_from(values.len()).ok(),
        _ => None,
    }
}

fn ensure_token_id_in_range(id: u32, len: usize, label: &str) -> Result<()> {
    if usize::try_from(id).ok().is_some_and(|idx| idx < len) {
        Ok(())
    } else {
        bail!("{label} {id} is outside tokenizer vocab of size {len}");
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
                let q1 = (((ql[ql_offset + l] & 0x0f) | (((qh_byte >> 0) & 0x03) << 4)) as i8) - 32;
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
    if element_count % block_elements != 0 {
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

fn encode_byte_level_text(bytes: &[u8]) -> String {
    let encoder = byte_encoder();
    bytes.iter().map(|byte| encoder[*byte as usize]).collect()
}

fn decode_byte_level_text(text: &str) -> Result<String> {
    let decoder = byte_decoder();
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if let Some(byte) = decoder.get(&ch) {
            bytes.push(*byte);
        } else {
            let mut buffer = [0; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
        }
    }
    String::from_utf8(bytes).context("byte-level GGUF tokenizer produced invalid UTF-8")
}

fn sentencepiece_normalize(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len() + 3);
    normalized.push('\u{2581}');
    for ch in text.chars() {
        if ch == ' ' {
            normalized.push('\u{2581}');
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

fn decode_sentencepiece_text(text: &str) -> Result<String> {
    let mut out = String::new();
    let mut bytes = Vec::new();
    let mut offset = 0usize;
    while offset < text.len() {
        let remaining = &text[offset..];
        if let Some(byte) = remaining.get(..6).and_then(byte_fallback_value) {
            bytes.push(byte);
            offset += 6;
            continue;
        }
        if !bytes.is_empty() {
            let decoded = String::from_utf8(std::mem::take(&mut bytes))
                .context("sentencepiece byte fallback produced invalid UTF-8")?;
            out.push_str(&decoded);
        }
        let ch = remaining
            .chars()
            .next()
            .expect("remaining string is non-empty");
        if ch == '\u{2581}' {
            out.push(' ');
        } else {
            out.push(ch);
        }
        offset += ch.len_utf8();
    }
    if !bytes.is_empty() {
        let decoded = String::from_utf8(bytes)
            .context("sentencepiece byte fallback produced invalid UTF-8")?;
        out.push_str(&decoded);
    }
    Ok(out)
}

fn byte_fallback_value(token: &str) -> Option<u8> {
    let hex = token
        .strip_prefix("<0x")
        .and_then(|value| value.strip_suffix('>'))?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn byte_encoder() -> &'static [char; 256] {
    static ENCODER: OnceLock<[char; 256]> = OnceLock::new();
    ENCODER.get_or_init(|| {
        let mut bytes = Vec::new();
        bytes.extend(b'!'..=b'~');
        bytes.extend(0xA1..=0xAC);
        bytes.extend(0xAE..=0xFF);

        let mut codepoints = bytes
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let mut extra = 0u32;
        for byte in 0u8..=u8::MAX {
            if !bytes.contains(&byte) {
                bytes.push(byte);
                codepoints.push(256 + extra);
                extra += 1;
            }
        }

        let mut encoder = ['\0'; 256];
        for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
            encoder[byte as usize] = char::from_u32(codepoint).expect("valid GPT-2 byte mapping");
        }
        encoder
    })
}

fn byte_decoder() -> &'static HashMap<char, u8> {
    static DECODER: OnceLock<HashMap<char, u8>> = OnceLock::new();
    DECODER.get_or_init(|| {
        byte_encoder()
            .iter()
            .enumerate()
            .map(|(byte, ch)| (*ch, byte as u8))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const LLAMA3_CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

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
