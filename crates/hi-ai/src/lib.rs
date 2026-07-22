//! `hi-ai` — provider-neutral LLM types, the [`Provider`] trait, and adapters
//! for OpenAI-compatible and Anthropic backends.

/// Serializes tests that mutate `HOME`/`XDG_CONFIG_HOME`, which are
/// process-wide. Both the models-cache tests and the credential-store tests
/// redirect the config dir, and cargo runs them on parallel threads, so without
/// a shared lock one test's `set_var` lands under another's feet.
/// A tokio mutex rather than `std`: the models-cache test is async and holds
/// this across `.await`, which a `std` guard must not be.
#[cfg(test)]
pub(crate) static ENV_HOME_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub mod anthropic;
pub mod auth_store;
pub mod circuit_breaker;
pub mod fallback;
mod http;
pub mod huggingface;
pub mod mcp;
pub mod moa;
pub mod openai;
pub mod pipenetwork_auth;
pub mod provider;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod token;
pub mod types;
pub mod xai_auth;

pub use anthropic::AnthropicProvider;
pub use fallback::{Backend, FallbackProvider};
pub use huggingface::{
    HfFileInfo, HfModelInfo, HfRepoRef, HuggingFaceHubClient, ModelCandidate, ModelDiscovery,
    ModelDiscoveryQuery, ModelSource,
};
// Re-export the on-disk /models cache helpers so the TUI can load cached model
// metadata at startup (instant) and save fresh results from the background fetch.
pub use auth_store::StoredToken;
pub use http::{
    agent_http_client, agent_http_client_quick, cache_key, load_cache, save_cache,
    timed_http_client_fallback,
};
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
pub use token::{StaticToken, TokenSource};
pub use types::{
    ChatRequest, CompatMode, Completion, Content, Message, RateLimitBucket, RateLimitState,
    ReasoningEffort, RequestProfile, Role, StreamEvent, ToolCall, ToolMode, ToolSpec, Usage,
    estimate_completion_output_tokens, estimate_content_tokens, estimate_messages_tokens,
    estimate_text_tokens,
};
