//! `hi-ai` — provider-neutral LLM types, the [`Provider`] trait, and adapters
//! for OpenAI-compatible and Anthropic backends.

pub mod anthropic;
pub mod fallback;
mod http;
pub mod huggingface;
pub mod mcp;
pub mod moa;
pub mod openai;
pub mod provider;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use fallback::{Backend, FallbackProvider};
pub use huggingface::{
    HfFileInfo, HfModelInfo, HfRepoRef, HuggingFaceHubClient, ModelCandidate, ModelDiscovery,
    ModelDiscoveryQuery, ModelSource,
};
// Re-export the on-disk /models cache helpers so the TUI can load cached model
// metadata at startup (instant) and save fresh results from the background fetch.
pub use http::{agent_http_client, cache_key, load_cache, save_cache};
pub use mcp::{
    McpDiscoveryProvider, McpTool, PIPE_MCP_DEFAULT_URL, PipeMcpClient, PipeMcpModelHealth,
    PipeMcpModelMetadata,
};
pub use moa::{
    MOA_AGGREGATOR_CONSERVATIVE, MOA_MODEL_CONSERVATIVE, MOA_PRESET_CONSERVATIVE,
    MOA_REFERENCE_CONSERVATIVE, MoaConfig, MoaPreset, MoaProvider,
};
pub use openai::OpenAiProvider;
pub use provider::{
    CODING_AGENT_MIN_OUTPUT_TOKENS, OutputCapError, Provider, ProviderError, ProviderErrorKind,
    ServedModel, effective_coding_agent_max_tokens, is_pipenetwork_coding_route,
    provider_error_is_temporary_overload, provider_error_kind, provider_error_usage,
    provider_output_cap_error, provider_retry_after_seconds, provider_route_error_is_retryable,
};
pub use types::{
    ChatRequest, CompatMode, Completion, Content, Message, RateLimitBucket, RateLimitState,
    ReasoningEffort, RequestProfile, Role, StreamEvent, ToolCall, ToolMode, ToolSpec, Usage,
    estimate_completion_output_tokens, estimate_content_tokens, estimate_messages_tokens,
    estimate_text_tokens,
};
