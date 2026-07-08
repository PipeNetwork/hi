use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 2048;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    pub family: ModelFamily,
    pub model_type: String,
    pub architecture: String,
    pub context_length: Option<u32>,
    pub max_output_tokens: u32,
    pub tokenizer: TokenizerInfo,
    pub chat_template: bool,
    pub weight_shards: Vec<WeightShard>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    Qwen2,
    Qwen3,
    Llama,
    Mistral,
    Mixtral,
    Gemma,
    Phi,
    DeepSeek,
    GlmFlash,
    Hy3,
    NemotronH,
}

impl ModelFamily {
    pub fn label(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::Gemma => "gemma",
            Self::Phi => "phi",
            Self::DeepSeek => "deepseek",
            Self::GlmFlash => "glm-flash",
            Self::Hy3 => "hy3",
            Self::NemotronH => "nemotron-h",
        }
    }

    pub fn from_qwen_architecture(architecture: &str) -> Option<Self> {
        Self::from_gguf_architecture(architecture)
    }

    pub fn from_gguf_architecture(architecture: &str) -> Option<Self> {
        let arch = architecture.to_ascii_lowercase();
        if arch.contains("hy_v3") || arch.contains("hyv3") || arch.contains("hunyuan") {
            Some(Self::Hy3)
        } else if arch.contains("qwen3") {
            Some(Self::Qwen3)
        } else if arch.contains("qwen") {
            Some(Self::Qwen2)
        } else if arch.contains("deepseek") {
            Some(Self::DeepSeek)
        } else if arch.contains("glm") {
            Some(Self::GlmFlash)
        } else if arch.contains("mixtral") {
            Some(Self::Mixtral)
        } else if arch.contains("mistral") {
            Some(Self::Mistral)
        } else if arch.contains("gemma") {
            Some(Self::Gemma)
        } else if arch.contains("phi") {
            Some(Self::Phi)
        } else if arch.contains("llama") {
            Some(Self::Llama)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub tokenizer_json: bool,
    pub tokenizer_config: bool,
    pub special_tokens_map: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightShard {
    pub path: String,
    pub bytes: u64,
    pub tensor_count: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListedModel {
    pub path: PathBuf,
    pub id: Option<String>,
    pub supported: bool,
    pub summary: String,
}
