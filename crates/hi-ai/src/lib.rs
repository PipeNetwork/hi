//! `hi-ai` — provider-neutral LLM types, the [`Provider`] trait, and adapters
//! for OpenAI-compatible, Anthropic, and xAI Responses backends.

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
pub mod concurrency;
pub mod fallback;
mod http;
pub mod huggingface;
pub mod mcp;
pub mod moa;
pub mod openai;
pub mod pipenetwork_auth;
pub mod provider;
pub mod provider_capabilities;
mod request_envelope;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod token;
mod tool_validation;
pub mod types;
pub mod x402;
pub mod x402_auth;
pub mod xai;
pub mod xai_auth;

pub use anthropic::AnthropicProvider;
pub use circuit_breaker::{BreakerConfig, BreakerEvent, BreakerObserver, BreakerState};
pub use concurrency::{
    ConcurrencyLimitedProvider, DEFAULT_PROVIDER_REQUEST_CONCURRENCY, ProviderConcurrencyConfig,
};
pub use fallback::{Backend, FallbackProvider};
pub use huggingface::{
    HfCollection, HfCollectionItem, HfCollectionNote, HfFileInfo, HfModelInfo, HfRepoRef,
    HuggingFaceHubClient, ModelCandidate, ModelDiscovery, ModelDiscoveryQuery, ModelSource,
};
// Re-export the on-disk /models cache helpers so the TUI can load cached model
// metadata at startup (instant) and save fresh results from the background fetch.
pub use auth_store::StoredToken;
pub use http::{
    HttpRetryEvent, HttpRetryObserver, agent_http_client, agent_http_client_quick, cache_key,
    credential_redirect_policy, load_cache, save_cache, set_http_retry_observer,
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
    CODING_AGENT_MIN_OUTPUT_TOKENS, KeyCheck, OutputCapError, Provider, ProviderCapabilities,
    ProviderError, ProviderErrorKind, ServedModel, effective_coding_agent_max_tokens,
    is_billing_or_quota_text, is_http_auth_rejection, is_pipenetwork_coding_route,
    provider_error_affects_health, provider_error_is_fallback_eligible,
    provider_error_is_temporary_overload, provider_error_kind, provider_error_retryable,
    provider_error_usage, provider_output_cap_error, provider_retry_after_seconds,
    provider_route_error_is_retryable,
};
pub use provider_capabilities::{
    CAPABILITY_RECORD_SCHEMA_VERSION, CAPABILITY_REGISTRY_VERSION, CancellationSupport,
    CapabilityMemberRecord, CapabilityProbe, CapabilityProbeAuditRecord,
    CapabilityProbeDisposition, CapabilityProbeObservation, CapabilityRegistryConfig,
    CapabilityRoute, DEFAULT_CAPABILITY_CACHE_TTL, DEFAULT_CAPABILITY_PROBE_MEMBERS,
    DEFAULT_CAPABILITY_PROBE_TIMEOUT, EffectiveProviderCapabilities, MAX_CAPABILITY_PROBE_MEMBERS,
    MAX_CAPABILITY_PROBE_TIMEOUT, ProviderCapabilityCandidate, ProviderCapabilityRegistry,
    ProviderModalities, ProviderRequestLimits, ReasoningReplayCapabilities, StrictSchemaDialect,
    ToolChoiceCapabilities, UsageReporting,
};
pub use request_envelope::RequestToolEnvelope;
pub use token::{PersistableToken, StaticToken, TokenSource};
pub use tool_validation::{
    MAX_TOOL_ARGUMENT_BYTES, validate_client_tool_batch_limits,
    validate_client_tool_batch_limits_with, validate_client_tool_call,
    validate_client_tool_call_with_limit, validate_client_tool_calls,
};
pub use types::{
    ChatRequest, CompatMode, Completion, Content, CostEstimate, DeepSeekCompat, Message,
    NormalizedUsage, OutputTokenParameter, PromptInput, PromptPart, RateLimitBucket,
    RateLimitState, ReasoningEffort, RequestProfile, Role, StreamEvent, ToolCall, ToolCallChannel,
    ToolMode, ToolSpec, Usage, WireAudit, estimate_completion_output_tokens,
    estimate_content_tokens, estimate_messages_tokens, estimate_request_input_tokens,
    estimate_text_tokens, estimate_tool_schema_tokens,
};
pub use x402::{
    AutoX402Confirmer, X402_CREDIT_TOKEN_PREFIX, X402_DEFAULT_MAX_USD, X402_MIN_TOPUP_MINOR,
    X402_PAYMENT_REQUIRED_HEADER, X402_PAYMENT_RESPONSE_HEADER, X402_PAYMENT_SIGNATURE_HEADER,
    X402_SCHEME_EXACT, X402_SOLANA_MAINNET, X402_USDC_MINT_MAINNET, X402ConfirmBroker,
    X402ConfirmRequest, X402Confirmer, X402PaymentPayload, X402PaymentRequired,
    X402PaymentRequirements, X402PaymentResponse, X402QuoteSummary, X402SettlePayload, X402Settler,
    X402UserPrompt, decode_payment_payload_header, decode_payment_required_header,
    decode_payment_response_header, encode_payment_payload_header, encode_payment_required_header,
    encode_payment_response_header, quote_summary, validate_quote,
};
pub use x402_auth::{
    X402_PROVIDER_ID, credit_token_source, has_credit_token, load_credit_token,
    logout as x402_logout, logout_quiet as x402_logout_quiet, validate_keypair_file,
};
pub use xai::XaiProvider;
