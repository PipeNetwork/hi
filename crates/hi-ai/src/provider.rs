//! The `Provider` trait: the single seam every model backend implements.

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{ChatRequest, Completion, StreamEvent, Usage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Auth,
    RateLimit,
    CapacityUnavailable,
    ModelUnavailable,
    Outage,
    UnsupportedRequestShape,
    UnsupportedTools,
    RequestTooLarge,
    QualityRejected,
    ToolProtocol,
    StreamTimeout,
    MalformedStream,
    EmptyCompletion,
    Other,
}

impl ProviderErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::RateLimit => "rate_limit",
            Self::CapacityUnavailable => "capacity_unavailable",
            Self::ModelUnavailable => "model_unavailable",
            Self::Outage => "outage",
            Self::UnsupportedRequestShape => "unsupported_request_shape",
            Self::UnsupportedTools => "unsupported_tools",
            Self::RequestTooLarge => "request_too_large",
            Self::QualityRejected => "quality_rejected",
            Self::ToolProtocol => "tool_protocol",
            Self::StreamTimeout => "stream_timeout",
            Self::MalformedStream => "malformed_stream",
            Self::EmptyCompletion => "empty_completion",
            Self::Other => "other",
        }
    }
}

#[derive(Debug)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    pub usage: Usage,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            usage: Usage::default(),
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

pub fn provider_error_kind(err: &anyhow::Error) -> Option<ProviderErrorKind> {
    err.downcast_ref::<ProviderError>().map(|e| e.kind)
}

pub fn provider_error_usage(err: &anyhow::Error) -> Usage {
    err.downcast_ref::<ProviderError>()
        .map(|e| e.usage)
        .unwrap_or_default()
}

/// A model the endpoint serves, with whatever live metadata it reports via its
/// `/models` route. Everything past `id` is best-effort — most endpoints report
/// only the id (then these stay `None`), but some (e.g. pipenetwork.ai) also report
/// the context window, pricing, and a health status.
///
/// `Serialize`/`Deserialize` back the on-disk startup cache (see `http::cache`):
/// a successful `/models` fetch is written keyed by provider+base_url and loaded
/// synchronously next startup so model metadata applies instantly, without
/// blocking on the network.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ServedModel {
    pub id: String,
    /// Context window in tokens.
    pub context_window: Option<u32>,
    /// Maximum output tokens the endpoint says this model/route supports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Pricing `(input, output)` in USD per 1M tokens.
    pub price: Option<(f64, f64)>,
    /// Human-readable provider/source label when the endpoint exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_label: Option<String>,
    /// Health label as reported, e.g. "available" or "degraded".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Whether the endpoint currently flags the model as usable.
    #[serde(default = "default_available")]
    pub available: bool,
    /// User-visible reason for unavailability/degradation, if one is public.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability_reason: Option<String>,
    /// Capability tags reported by the endpoint, e.g. "tools" or "reasoning".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
}

pub const CODING_AGENT_MIN_OUTPUT_TOKENS: u32 = 8192;

pub fn effective_coding_agent_max_tokens(
    model: &str,
    configured_max_tokens: u32,
    max_tokens_explicit: bool,
    advertised_max_output_tokens: Option<u32>,
) -> u32 {
    let advertised = advertised_max_output_tokens.filter(|limit| *limit > 0);
    let configured = if !max_tokens_explicit && is_pipenetwork_coding_route(model) {
        configured_max_tokens.max(CODING_AGENT_MIN_OUTPUT_TOKENS)
    } else {
        configured_max_tokens
    };

    match advertised {
        Some(limit) if max_tokens_explicit => configured.min(limit),
        Some(limit) if is_pipenetwork_coding_route(model) => limit,
        Some(limit) => configured.min(limit),
        None => configured,
    }
}

pub fn is_pipenetwork_coding_route(model: &str) -> bool {
    matches!(model, "ipop/coder-balanced" | "pipe/auto-code")
}

impl ServedModel {
    /// A short health label worth flagging, or `None` when the model is healthy
    /// (or the endpoint reported nothing). Used to warn before you rely on a
    /// degraded/limited model.
    pub fn health(&self) -> Option<&str> {
        match self.status.as_deref() {
            Some(s) if !s.eq_ignore_ascii_case("available") => Some(s),
            None if !self.available => Some("unavailable"),
            _ => None,
        }
    }
}

fn default_available() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipenetwork_coding_routes_use_advertised_output_limit_when_implicit() {
        assert_eq!(
            effective_coding_agent_max_tokens(
                "ipop/coder-balanced",
                CODING_AGENT_MIN_OUTPUT_TOKENS,
                false,
                Some(131_072),
            ),
            131_072
        );
        assert_eq!(
            effective_coding_agent_max_tokens(
                "pipe/auto-code",
                CODING_AGENT_MIN_OUTPUT_TOKENS,
                false,
                Some(16_384),
            ),
            16_384
        );
    }

    #[test]
    fn explicit_output_limit_is_honored_and_clamped() {
        assert_eq!(
            effective_coding_agent_max_tokens("ipop/coder-balanced", 4096, true, Some(131_072)),
            4096
        );
        assert_eq!(
            effective_coding_agent_max_tokens("pipe/auto-code", 65_536, true, Some(16_384)),
            16_384
        );
    }

    #[test]
    fn coding_routes_never_drop_below_the_default_without_an_advertised_cap() {
        assert_eq!(
            effective_coding_agent_max_tokens("ipop/coder-balanced", 2048, false, None),
            CODING_AGENT_MIN_OUTPUT_TOKENS
        );
    }
}

/// A model backend. Implementations own the wire-format translation and SSE
/// reassembly so the agent loop stays provider-agnostic.
///
/// `sink` is invoked for each incremental [`StreamEvent`] as it arrives; the
/// returned [`Completion`] is the fully-assembled assistant turn (text,
/// reasoning, and tool calls).
#[async_trait]
pub trait Provider: Send + Sync {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion>;

    /// The models this endpoint actually serves (via its `/models` route), with
    /// any live metadata reported. Default: empty, so callers fall back to the
    /// static models.dev catalog.
    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        Ok(Vec::new())
    }
}
