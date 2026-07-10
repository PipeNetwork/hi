// Axum request handlers use `Result<T, Response>` so a failed check can early-return
// a fully-formed HTTP response. `Response` is large, but boxing it in every handler
// signature would be non-idiomatic and churn the whole server surface.
#![allow(clippy::result_large_err)]

pub mod backend;
pub mod model;
pub mod prompt;
pub mod server;
pub mod tool_parser;

pub use backend::{
    BackendHealth, GenerationEvent, GenerationOutput, GenerationRequest, GenerationStream,
    ImageInput, ImageSource, ImageUrlKind, InferenceBackend, MultimodalInput, MultimodalSupport,
    SamplingDefaults, SharedBackend, VideoInput, VideoSource,
};
pub use model::{
    DEFAULT_MAX_OUTPUT_TOKENS, ListedModel, ModelFamily, ModelInfo, TokenizerInfo, WeightShard,
};

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use anyhow::Result;
    use async_trait::async_trait;
    use futures_util::stream;
    use tokio::sync::Mutex;

    use crate::backend::{
        BackendHealth, GenerationEvent, GenerationOutput, GenerationRequest, GenerationStream,
        InferenceBackend, MultimodalSupport, SamplingDefaults,
    };
    use crate::model::ModelInfo;

    pub struct MockBackend {
        model: ModelInfo,
        output: Mutex<String>,
        stream_error: Option<String>,
        last_prompt: Mutex<Option<String>>,
        last_request: Mutex<Option<GenerationRequest>>,
        backend: String,
        quantization: String,
        sampling_defaults: SamplingDefaults,
        multimodal_support: MultimodalSupport,
        chat_template: Option<String>,
    }

    impl MockBackend {
        pub fn new(model: ModelInfo, output: impl Into<String>) -> Self {
            Self {
                model,
                output: Mutex::new(output.into()),
                stream_error: None,
                last_prompt: Mutex::new(None),
                last_request: Mutex::new(None),
                backend: "mock".to_string(),
                quantization: "mock".to_string(),
                sampling_defaults: SamplingDefaults::default(),
                multimodal_support: MultimodalSupport::text_only(),
                chat_template: None,
            }
        }

        pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
            self.backend = backend.into();
            self
        }

        pub fn with_quantization(mut self, quantization: impl Into<String>) -> Self {
            self.quantization = quantization.into();
            self
        }

        pub fn with_sampling_defaults(mut self, sampling_defaults: SamplingDefaults) -> Self {
            self.sampling_defaults = sampling_defaults;
            self
        }

        pub fn with_multimodal_support(mut self, multimodal_support: MultimodalSupport) -> Self {
            self.multimodal_support = multimodal_support;
            self
        }

        pub fn with_chat_template(mut self, chat_template: impl Into<String>) -> Self {
            self.chat_template = Some(chat_template.into());
            self
        }

        pub fn with_stream_error(mut self, error: impl Into<String>) -> Self {
            self.stream_error = Some(error.into());
            self
        }

        pub async fn last_prompt(&self) -> Option<String> {
            self.last_prompt.lock().await.clone()
        }

        pub async fn last_request(&self) -> Option<GenerationRequest> {
            self.last_request.lock().await.clone()
        }
    }

    #[async_trait]
    impl InferenceBackend for MockBackend {
        fn model(&self) -> &ModelInfo {
            &self.model
        }

        fn health(&self) -> BackendHealth {
            BackendHealth {
                backend: self.backend.clone(),
                ready: true,
                family: self.model.family.label().to_string(),
                quantization: self.quantization.clone(),
                context_length: self.model.context_length,
                memory_estimate_bytes: Some(self.model.weight_shards.iter().map(|s| s.bytes).sum()),
            }
        }

        fn sampling_defaults(&self) -> SamplingDefaults {
            self.sampling_defaults
        }

        fn multimodal_support(&self) -> MultimodalSupport {
            self.multimodal_support.clone()
        }

        fn chat_template(&self) -> Option<&str> {
            self.chat_template.as_deref()
        }

        async fn stream_generate(&self, request: GenerationRequest) -> Result<GenerationStream> {
            *self.last_prompt.lock().await = Some(request.prompt.clone());
            *self.last_request.lock().await = Some(request.clone());
            if let Some(error) = &self.stream_error {
                anyhow::bail!("{error}");
            }
            let text = self.output.lock().await.clone();
            let prompt_tokens = (request.prompt.len() / 4).max(1) as u64;
            let completion_tokens = (text.len() / 4).max(1) as u64;
            let mut events = split_stream_text(&text)
                .into_iter()
                .map(|piece| {
                    Ok(GenerationEvent::TokenDelta {
                        token_id: 0,
                        text: piece,
                    })
                })
                .collect::<Vec<_>>();
            events.push(Ok(GenerationEvent::Finished {
                output: GenerationOutput {
                    prompt_tokens,
                    completion_tokens,
                    text,
                },
            }));
            Ok(Box::pin(stream::iter(events)))
        }
    }

    fn split_stream_text(text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if current.len() >= 512 {
                out.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
        out
    }
}
