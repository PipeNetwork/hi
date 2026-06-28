//! `hi-ai` — provider-neutral LLM types, the [`Provider`] trait, and adapters
//! for OpenAI-compatible and Anthropic backends.

pub mod anthropic;
pub mod fallback;
mod http;
pub mod openai;
pub mod provider;
pub mod registry;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use fallback::{Backend, FallbackProvider};
// Re-export the on-disk /models cache helpers so the TUI can load cached model
// metadata at startup (instant) and save fresh results from the background fetch.
pub use http::{cache_key, load_cache, save_cache};
pub use openai::OpenAiProvider;
pub use provider::{
    Provider, ProviderError, ProviderErrorKind, ServedModel, provider_error_kind,
    provider_error_usage,
};
pub use registry::{ModelInfo, Registry};
pub use types::{
    BillableBreakdown, ChatRequest, CompatMode, Completion, Content, Message, RequestProfile, Role,
    StreamEvent, ToolCall, ToolMode, ToolSpec, Usage, estimate_completion_output_tokens,
    estimate_content_tokens, estimate_messages_tokens, estimate_text_tokens,
};
