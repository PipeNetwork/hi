use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::model::ModelInfo;

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
    pub media_inputs: Vec<MultimodalInput>,
    /// Original chat messages (content parts intact), needed to interleave media placeholders for
    /// multimodal models. Empty for text-only requests.
    pub messages: Vec<crate::server::ChatMessage>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MultimodalInput {
    Image(ImageInput),
    Video(VideoInput),
    Audio(AudioInput),
}

/// Decoded audio: mono PCM samples in [-1, 1], already resampled to `sampling_rate`.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioInput {
    pub samples: Vec<f32>,
    pub sampling_rate: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageInput {
    pub source: ImageSource,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource {
    Data { media_type: String, bytes: Vec<u8> },
    Url { kind: ImageUrlKind, url: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct VideoInput {
    pub source: VideoSource,
    pub detail: Option<String>,
    pub fps: Option<f32>,
    pub nframes: Option<usize>,
    pub min_frames: Option<usize>,
    pub max_frames: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoSource {
    Data { media_type: String, bytes: Vec<u8> },
    Url { kind: ImageUrlKind, url: String },
    Frames(Vec<ImageInput>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageUrlKind {
    Http,
    Local,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultimodalSupport {
    pub image_inputs: bool,
    pub video_inputs: bool,
    pub audio_inputs: bool,
    pub generation: bool,
    pub status: String,
}

impl MultimodalSupport {
    pub fn text_only() -> Self {
        Self {
            image_inputs: false,
            video_inputs: false,
            audio_inputs: false,
            generation: false,
            status: "text-only".to_string(),
        }
    }

    pub fn image_generation(status: impl Into<String>) -> Self {
        Self {
            image_inputs: true,
            video_inputs: false,
            audio_inputs: false,
            generation: true,
            status: status.into(),
        }
    }

    pub fn image_video_generation(status: impl Into<String>) -> Self {
        Self {
            image_inputs: true,
            video_inputs: true,
            audio_inputs: false,
            generation: true,
            status: status.into(),
        }
    }

    pub fn image_audio_generation(status: impl Into<String>) -> Self {
        Self {
            image_inputs: true,
            video_inputs: false,
            audio_inputs: true,
            generation: true,
            status: status.into(),
        }
    }

    pub fn image_inputs_without_generation(status: impl Into<String>) -> Self {
        Self {
            image_inputs: true,
            video_inputs: false,
            audio_inputs: false,
            generation: false,
            status: status.into(),
        }
    }

    pub fn image_video_inputs_without_generation(status: impl Into<String>) -> Self {
        Self {
            image_inputs: true,
            video_inputs: true,
            audio_inputs: false,
            generation: false,
            status: status.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Clone, Debug)]
pub enum GenerationEvent {
    TokenDelta { token_id: u32, text: String },
    Finished { output: GenerationOutput },
}

pub type GenerationStream = Pin<Box<dyn Stream<Item = Result<GenerationEvent>> + Send>>;

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn model(&self) -> &ModelInfo;

    fn health(&self) -> BackendHealth;

    fn sampling_defaults(&self) -> SamplingDefaults {
        SamplingDefaults::default()
    }

    fn multimodal_support(&self) -> MultimodalSupport {
        MultimodalSupport::text_only()
    }

    fn chat_template(&self) -> Option<&str> {
        None
    }

    async fn stream_generate(&self, request: GenerationRequest) -> Result<GenerationStream>;

    async fn generate(&self, request: GenerationRequest) -> Result<GenerationOutput> {
        let mut stream = self.stream_generate(request).await?;
        let mut fallback_text = String::new();
        let mut final_output = None;
        while let Some(event) = stream.next().await {
            match event? {
                GenerationEvent::TokenDelta { text, .. } => fallback_text.push_str(&text),
                GenerationEvent::Finished { output } => final_output = Some(output),
            }
        }
        final_output.ok_or_else(|| {
            anyhow!(
                "generation stream ended before completion metadata; collected {} bytes",
                fallback_text.len()
            )
        })
    }
}

pub type SharedBackend = Arc<dyn InferenceBackend>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendHealth {
    pub backend: String,
    pub ready: bool,
    pub family: String,
    pub quantization: String,
    pub context_length: Option<u32>,
    pub memory_estimate_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use crate::model::{ModelFamily, ModelInfo, TokenizerInfo};

    use super::*;

    #[tokio::test]
    async fn generate_collects_stream_output() {
        let backend = crate::test_support::MockBackend::new(test_model(), "collected");
        let output = backend
            .generate(GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
                messages: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(output.text, "collected");
    }

    #[tokio::test]
    async fn stream_emits_delta_before_finish() {
        let backend = crate::test_support::MockBackend::new(test_model(), "streamed");
        let mut stream = backend
            .stream_generate(GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: Vec::new(),
                media_inputs: Vec::new(),
                messages: Vec::new(),
            })
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert!(matches!(first, GenerationEvent::TokenDelta { .. }));
        let second = stream.next().await.unwrap().unwrap();
        assert!(matches!(second, GenerationEvent::Finished { .. }));
    }

    fn test_model() -> ModelInfo {
        ModelInfo {
            id: "local-test-model".to_string(),
            path: std::path::PathBuf::from("/tmp/local-test-model"),
            family: ModelFamily::Qwen2,
            model_type: "qwen2".to_string(),
            architecture: "Qwen2ForCausalLM".to_string(),
            context_length: Some(32),
            max_output_tokens: 8,
            tokenizer: TokenizerInfo {
                tokenizer_json: true,
                tokenizer_config: false,
                special_tokens_map: false,
            },
            chat_template: false,
            weight_shards: Vec::new(),
        }
    }
}
