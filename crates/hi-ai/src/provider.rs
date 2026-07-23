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
    MalformedStream,
    EmptyCompletion,
    Other,
}

impl ProviderErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::RateLimit => "rate_limit",
            Self::CapacityUnavailable => "capacity",
            Self::ModelUnavailable => "request",
            Self::Outage => "request",
            Self::UnsupportedRequestShape => "unsupported_request_shape",
            Self::UnsupportedTools => "unsupported_tools",
            Self::RequestTooLarge => "request_too_large",
            Self::QualityRejected => "quality_rejected",
            Self::ToolProtocol => "tool_protocol",
            Self::MalformedStream => "malformed_stream",
            Self::EmptyCompletion => "empty_completion",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    pub usage: Usage,
    /// Stable public error code supplied by a structured API response.
    pub code: Option<String>,
    /// Explicit retry contract. `None` means a legacy/direct provider did not
    /// supply one and callers may apply the bounded kind-based fallback.
    pub retryable: Option<bool>,
    pub retry_after_seconds: Option<u64>,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            usage: Usage::default(),
            code: None,
            retryable: None,
            retry_after_seconds: None,
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    pub fn with_api_contract(
        mut self,
        code: Option<String>,
        retryable: Option<bool>,
        retry_after_seconds: Option<u64>,
    ) -> Self {
        self.code = code;
        self.retryable = retryable;
        self.retry_after_seconds = retry_after_seconds;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputCapError {
    pub available_output_tokens: Option<u32>,
}

pub fn provider_output_cap_error(err: &anyhow::Error) -> Option<OutputCapError> {
    output_cap_error_from_text(&provider_error_text(err))
}

pub fn provider_retry_after_seconds(err: &anyhow::Error) -> Option<u64> {
    if let Some(error) = err.downcast_ref::<ProviderError>()
        && error.retry_after_seconds.is_some()
    {
        return error.retry_after_seconds;
    }
    retry_after_seconds_from_text(&provider_error_text(err))
}

pub fn provider_error_retryable(err: &anyhow::Error) -> Option<bool> {
    err.downcast_ref::<ProviderError>()
        .and_then(|error| error.retryable)
        .or_else(|| json_bool_field(&provider_error_text(err), "retryable"))
}

pub fn provider_route_error_is_retryable(err: &anyhow::Error) -> bool {
    let Some(kind) = provider_error_kind(err) else {
        return false;
    };
    if !matches!(
        kind,
        ProviderErrorKind::CapacityUnavailable
            | ProviderErrorKind::ModelUnavailable
            | ProviderErrorKind::Outage
    ) {
        return false;
    }
    if let Some(retryable) = provider_error_retryable(err) {
        return retryable;
    }
    let text = provider_error_text(err);
    let lower = text.to_ascii_lowercase();
    match kind {
        ProviderErrorKind::CapacityUnavailable => true,
        ProviderErrorKind::ModelUnavailable => {
            if mentions_any(
                &lower,
                &["unknown model", "model not supported", "model not enabled"],
            ) {
                return false;
            }
            provider_retry_after_seconds(err).is_some()
                || json_bool_field(&text, "retryable") == Some(true)
                || mentions_any(
                    &lower,
                    &[
                        "temporarily unavailable",
                        "temporary unavailable",
                        "try again",
                    ],
                )
        }
        ProviderErrorKind::Outage => false,
        _ => false,
    }
}

/// Only transport/provider availability failures participate in the local
/// backend circuit breaker. Client input, authentication, policy/quality and
/// model tool-output failures must never make a healthy backend look down.
pub fn provider_error_affects_health(err: &anyhow::Error) -> bool {
    if provider_error_retryable(err) == Some(false) {
        return false;
    }
    matches!(
        provider_error_kind(err),
        Some(
            ProviderErrorKind::RateLimit
                | ProviderErrorKind::CapacityUnavailable
                | ProviderErrorKind::Outage
                | ProviderErrorKind::MalformedStream
        )
    )
}

/// Whether switching to a different configured backend is legitimate. An
/// explicit API retry contract wins; otherwise retain a narrow compatibility
/// fallback for direct providers that predate the structured envelope.
pub fn provider_error_is_fallback_eligible(err: &anyhow::Error) -> bool {
    if let Some(retryable) = provider_error_retryable(err) {
        return retryable;
    }
    matches!(
        provider_error_kind(err),
        Some(
            ProviderErrorKind::RateLimit
                | ProviderErrorKind::CapacityUnavailable
                | ProviderErrorKind::ModelUnavailable
                | ProviderErrorKind::Outage
                | ProviderErrorKind::MalformedStream
                | ProviderErrorKind::EmptyCompletion
        )
    )
}

pub fn provider_error_is_temporary_overload(err: &anyhow::Error) -> bool {
    if provider_error_retryable(err) == Some(false) {
        return false;
    }
    let Some(kind) = provider_error_kind(err) else {
        return false;
    };
    if !matches!(
        kind,
        ProviderErrorKind::RateLimit
            | ProviderErrorKind::CapacityUnavailable
            | ProviderErrorKind::ModelUnavailable
            | ProviderErrorKind::Outage
    ) {
        return false;
    }

    let text = provider_error_text(err);
    let lower = text.to_ascii_lowercase();
    if json_code_field(&text).is_some_and(|code| code == "1305") {
        return true;
    }
    mentions_any(
        &lower,
        &[
            "temporarily overloaded",
            "temporary overload",
            "provider overloaded",
            "server overloaded",
            "model overloaded",
            "glm overloaded",
            "glm-5.2 overloaded",
            "temporarily at capacity",
        ],
    ) || (mentions_any(&lower, &["overloaded", "over capacity"])
        && mentions_any(&lower, &["try again", "retry", "temporarily", "temporary"]))
}

fn provider_error_text(err: &anyhow::Error) -> String {
    err.downcast_ref::<ProviderError>()
        .map(|e| e.message.clone())
        .unwrap_or_else(|| err.to_string())
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
    matches!(model, "ipop/coder-balanced" | "pipe/auto-coder")
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

fn output_cap_error_from_text(text: &str) -> Option<OutputCapError> {
    let lower = text.to_ascii_lowercase();
    if let Some(available) = context_overflow_available_output_tokens(&lower) {
        return Some(OutputCapError {
            available_output_tokens: Some(available),
        });
    }

    let names_output_limit = mentions_any(
        &lower,
        &[
            "max_tokens",
            "max output",
            "max_output_tokens",
            "maximum output",
            "output tokens",
            "in the completion",
        ],
    );
    let names_limit_shape = mentions_any(
        &lower,
        &[
            "less than or equal to",
            "at most",
            "max allowed",
            "maximum allowed",
            "must be between",
            "should be [",
            "range of input length",
            "greater than",
        ],
    );
    if !(names_output_limit && names_limit_shape) {
        return None;
    }

    let limit = number_after_any(
        &lower,
        &[
            "less than or equal to",
            "at most",
            "max allowed",
            "maximum allowed",
            "maximum of",
            "max output tokens is",
            "max_output_tokens is",
        ],
    )
    .or_else(|| upper_bound_in_range(&lower))
    .or_else(|| largest_number(&lower));

    Some(OutputCapError {
        available_output_tokens: limit,
    })
}

fn context_overflow_available_output_tokens(lower: &str) -> Option<u32> {
    if !mentions_any(lower, &["maximum context length", "context length"])
        || !mentions_any(lower, &["in the completion", "output tokens"])
    {
        return None;
    }
    let max_context = number_after_any(
        lower,
        &[
            "maximum context length is",
            "maximum context length of",
            "context length is",
            "context window is",
        ],
    )?;
    let prompt_tokens = number_before_any(
        lower,
        &[
            " in the messages",
            " input tokens",
            " prompt tokens",
            " tokens in the prompt",
        ],
    )?;
    if prompt_tokens >= max_context {
        return Some(0);
    }
    Some(max_context - prompt_tokens)
}

fn retry_after_seconds_from_text(text: &str) -> Option<u64> {
    let value = json_value_from_text(text)?;
    u64_field_recursive(&value, "retry_after_seconds")
}

fn json_bool_field(text: &str, key: &str) -> Option<bool> {
    let value = json_value_from_text(text)?;
    bool_field_recursive(&value, key)
}

fn json_code_field(text: &str) -> Option<String> {
    let value = json_value_from_text(text)?;
    code_field_recursive(&value)
}

fn json_value_from_text(text: &str) -> Option<serde_json::Value> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        return Some(value);
    }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(&text[start..=end]).ok()
}

fn u64_field_recursive(value: &serde_json::Value, key: &str) -> Option<u64> {
    match value {
        serde_json::Value::Object(object) => object
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .or_else(|| object.values().find_map(|v| u64_field_recursive(v, key))),
        serde_json::Value::Array(values) => values.iter().find_map(|v| u64_field_recursive(v, key)),
        _ => None,
    }
}

fn bool_field_recursive(value: &serde_json::Value, key: &str) -> Option<bool> {
    match value {
        serde_json::Value::Object(object) => object
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .or_else(|| object.values().find_map(|v| bool_field_recursive(v, key))),
        serde_json::Value::Array(values) => {
            values.iter().find_map(|v| bool_field_recursive(v, key))
        }
        _ => None,
    }
}

fn code_field_recursive(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Object(object) => object
            .get("code")
            .and_then(|code| match code {
                serde_json::Value::Number(number) => Some(number.to_string()),
                serde_json::Value::String(string) => Some(string.clone()),
                _ => None,
            })
            .or_else(|| object.values().find_map(code_field_recursive)),
        serde_json::Value::Array(values) => values.iter().find_map(code_field_recursive),
        _ => None,
    }
}

fn mentions_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn number_after_any(text: &str, markers: &[&str]) -> Option<u32> {
    markers
        .iter()
        .filter_map(|marker| number_after(text, marker))
        .next()
}

fn number_before_any(text: &str, markers: &[&str]) -> Option<u32> {
    markers
        .iter()
        .filter_map(|marker| number_before(text, marker))
        .next()
}

fn number_after(text: &str, marker: &str) -> Option<u32> {
    let start = text.find(marker)? + marker.len();
    let mut digits = String::new();
    let mut seen_digit = false;
    for ch in text[start..].chars() {
        if ch.is_ascii_digit() {
            seen_digit = true;
            digits.push(ch);
        } else if seen_digit && matches!(ch, ',' | '_') {
            continue;
        } else if seen_digit {
            break;
        }
    }
    digits.parse().ok()
}

fn number_before(text: &str, marker: &str) -> Option<u32> {
    let end = text.find(marker)?;
    let mut digits_rev = String::new();
    let mut seen_digit = false;
    for ch in text[..end].chars().rev() {
        if ch.is_ascii_digit() {
            seen_digit = true;
            digits_rev.push(ch);
        } else if seen_digit && matches!(ch, ',' | '_') {
            continue;
        } else if seen_digit {
            break;
        }
    }
    if digits_rev.is_empty() {
        return None;
    }
    digits_rev.chars().rev().collect::<String>().parse().ok()
}

fn upper_bound_in_range(text: &str) -> Option<u32> {
    let open = text.find('[')?;
    let close = text[open..].find(']')? + open;
    largest_number(&text[open..=close])
}

fn largest_number(text: &str) -> Option<u32> {
    let mut best = None;
    let mut current = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if matches!(ch, ',' | '_') && !current.is_empty() {
            continue;
        } else if !current.is_empty() {
            if let Ok(value) = current.parse::<u32>() {
                best = Some(best.map_or(value, |prev: u32| prev.max(value)));
            }
            current.clear();
        }
    }
    best
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
    /// any live metadata reported. Default: empty.
    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        Ok(Vec::new())
    }
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
                "pipe/auto-coder",
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
            effective_coding_agent_max_tokens("pipe/auto-coder", 65_536, true, Some(16_384)),
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

    #[test]
    fn parses_output_cap_limit_from_provider_error_text() {
        let err = ProviderError::new(
            ProviderErrorKind::RequestTooLarge,
            "API error 400 Bad Request: max_tokens must be less than or equal to 8192",
        );

        assert_eq!(
            provider_output_cap_error(&anyhow::Error::new(err)),
            Some(OutputCapError {
                available_output_tokens: Some(8192)
            })
        );
    }

    #[test]
    fn parses_output_cap_from_context_error_when_prompt_fits() {
        let err = ProviderError::new(
            ProviderErrorKind::RequestTooLarge,
            "This model's maximum context length is 131072 tokens. However, you requested 140000 tokens (120000 in the messages, 20000 in the completion).",
        );

        assert_eq!(
            provider_output_cap_error(&anyhow::Error::new(err)),
            Some(OutputCapError {
                available_output_tokens: Some(11072)
            })
        );
    }

    #[test]
    fn does_not_parse_prompt_only_context_overflow_as_output_cap() {
        let err = ProviderError::new(
            ProviderErrorKind::RequestTooLarge,
            "input exceeds the model context length; reduce prompt tokens",
        );

        assert_eq!(provider_output_cap_error(&anyhow::Error::new(err)), None);
    }

    #[test]
    fn parses_retry_after_from_nested_api_error_json() {
        let err = ProviderError::new(
            ProviderErrorKind::ModelUnavailable,
            r#"API error 503 Service Unavailable: {"error":{"message":"model temporarily unavailable","retry_after_seconds":10},"retryable":true}"#,
        );

        let err = anyhow::Error::new(err);
        assert_eq!(provider_retry_after_seconds(&err), Some(10));
        assert!(provider_route_error_is_retryable(&err));
    }

    #[test]
    fn permanent_route_rejection_is_not_retryable_route_error() {
        let err = ProviderError::new(ProviderErrorKind::ModelUnavailable, "unknown model");

        assert!(!provider_route_error_is_retryable(&anyhow::Error::new(err)));
    }

    #[test]
    fn parses_external_glm_overload_as_temporary() {
        let err = ProviderError::new(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"glm-5.2 is temporarily overloaded","code":1305},"retry_after_seconds":0}"#,
        );

        let err = anyhow::Error::new(err);
        assert!(provider_error_is_temporary_overload(&err));
        assert_eq!(provider_retry_after_seconds(&err), Some(0));
    }

    #[test]
    fn normal_rate_limit_is_not_provider_overload() {
        let err = ProviderError::new(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"quota exceeded","code":"rate_limit"}}"#,
        );

        assert!(!provider_error_is_temporary_overload(&anyhow::Error::new(
            err
        )));
    }

    #[test]
    fn retryable_tool_errors_can_fallback_without_poisoning_route_health() {
        let err = anyhow::Error::new(
            ProviderError::new(ProviderErrorKind::ToolProtocol, "bad model tool output")
                .with_api_contract(Some("tool_protocol_error".to_string()), Some(true), None),
        );
        assert!(provider_error_is_fallback_eligible(&err));
        assert!(!provider_route_error_is_retryable(&err));
        assert!(!provider_error_affects_health(&err));
    }

    #[test]
    fn non_retryable_integration_errors_neither_fallback_nor_affect_health() {
        let err =
            anyhow::Error::new(
                ProviderError::new(ProviderErrorKind::Outage, "payload rejected")
                    .with_api_contract(Some("service_unavailable".to_string()), Some(false), None),
            );
        assert!(!provider_error_is_fallback_eligible(&err));
        assert!(!provider_route_error_is_retryable(&err));
        assert!(!provider_error_affects_health(&err));
    }
}
