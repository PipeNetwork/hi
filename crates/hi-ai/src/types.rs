//! Provider-neutral conversation model.
//!
//! Every adapter (OpenAI-compatible, Anthropic) translates these types to and
//! from its own wire format, so the agent core never sees provider specifics.
//! The shape is a superset modeled on content blocks (Anthropic-style) because
//! that round-trips both APIs cleanly — including reasoning/thinking, which the
//! flat OpenAI message shape can't represent on its own.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolMode {
    #[default]
    Auto,
    Required,
    ChatOnly,
    ReadOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompatMode {
    #[default]
    Auto,
    Strict,
}

/// Controls DeepSeek-specific behavior on OpenAI-compatible endpoints.
///
/// `Auto` detects the official DeepSeek endpoint and DeepSeek model aliases;
/// `On` is useful for gateways that proxy DeepSeek under another URL/model;
/// `Off` preserves the generic OpenAI-compatible wire shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepSeekCompat {
    #[default]
    Auto,
    On,
    Off,
}

impl DeepSeekCompat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

impl ToolMode {
    pub fn label(self) -> &'static str {
        match self {
            ToolMode::Auto => "auto",
            ToolMode::Required => "required",
            ToolMode::ChatOnly => "chat-only",
            ToolMode::ReadOnly => "read-only",
        }
    }
}

impl CompatMode {
    pub fn label(self) -> &'static str {
        match self {
            CompatMode::Auto => "auto",
            CompatMode::Strict => "strict",
        }
    }
}

/// How much internal reasoning to ask a reasoning-capable model to spend, for
/// OpenAI-compatible endpoints that accept a `reasoning_effort` parameter
/// (GPT-5 / o-series style, and several routed models such as pipenetwork's).
///
/// Unlike Anthropic's explicit [`ChatRequest::thinking_budget`] token count,
/// this is an abstract level the provider maps to its own internal token
/// target. `None` on a request omits the field entirely, leaving the endpoint's
/// default (which, for some models, is no reasoning at all). Ignored by the
/// Anthropic adapter, which uses `thinking_budget`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl ReasoningEffort {
    /// Levels from least to most effort — for menus, completion, and `/config`.
    pub const ALL: [ReasoningEffort; 5] = [
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
    ];

    /// One step more effort, saturating at the top level.
    pub fn next_higher(self) -> Self {
        match self {
            Self::Minimal => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High | Self::Xhigh => Self::Xhigh,
        }
    }

    /// The wire value sent as the `reasoning_effort` request field.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }

    /// Parse a user-supplied level (case-insensitive, with a few aliases).
    /// Returns `None` for anything unrecognized so callers can report an error.
    pub fn from_arg(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "minimal" | "min" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "x-high" | "extra-high" => Some(Self::Xhigh),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestProfile {
    pub compat: CompatMode,
    pub tool_mode: ToolMode,
    pub stream_usage: Option<bool>,
    #[serde(default)]
    pub deepseek_compat: DeepSeekCompat,
    /// Per-request override used by the agent's single DeepSeek strict-schema
    /// fallback. `None` follows the capability profile; `Some(false)` keeps
    /// DeepSeek thinking/tool replay enabled but omits JSON strict fields.
    #[serde(default)]
    pub deepseek_strict: Option<bool>,
    /// Optional per-request override for DeepSeek's thinking mode. `None`
    /// preserves the normal thinking-enabled DeepSeek profile; `Some(false)`
    /// is useful for short, tool-free synthesis after the tool loop ends.
    #[serde(default)]
    pub deepseek_thinking: Option<bool>,
}

impl Default for RequestProfile {
    fn default() -> Self {
        Self {
            compat: CompatMode::Auto,
            tool_mode: ToolMode::Auto,
            stream_usage: None,
            deepseek_compat: DeepSeekCompat::Auto,
            deepseek_strict: None,
            deepseek_thinking: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    /// Carries tool results back to the model.
    Tool,
}

/// Typed user input accepted by frontends before it is converted into a
/// provider-neutral [`Message`]. Existing string turns remain valid through
/// `From<&str>`/`From<String>`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PromptInput {
    pub parts: Vec<PromptPart>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptPart {
    Text { text: String },
    Image { data: String, media_type: String },
}

impl PromptInput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![PromptPart::Text { text: text.into() }],
        }
    }

    pub fn image(mut self, data: impl Into<String>, media_type: impl Into<String>) -> Self {
        self.parts.push(PromptPart::Image {
            data: data.into(),
            media_type: media_type.into(),
        });
        self
    }

    pub fn push_text(&mut self, text: impl Into<String>) {
        self.parts.push(PromptPart::Text { text: text.into() });
    }

    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                PromptPart::Text { text } => Some(text.as_str()),
                PromptPart::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn into_message(self) -> Message {
        Message {
            role: Role::User,
            content: self
                .parts
                .into_iter()
                .map(|part| match part {
                    PromptPart::Text { text } => Content::Text(text),
                    PromptPart::Image { data, media_type } => Content::Image { data, media_type },
                })
                .collect(),
        }
    }
}

impl From<&str> for PromptInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<String> for PromptInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

/// One conversation message: a role plus an ordered list of content blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text(text.into())],
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.into())],
        }
    }

    /// Create a user message with text and an image.
    pub fn user_with_image(
        text: impl Into<String>,
        data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::User,
            content: vec![
                Content::Image {
                    data: data.into(),
                    media_type: media_type.into(),
                },
                Content::Text(text.into()),
            ],
        }
    }

    pub fn user_input(input: PromptInput) -> Self {
        input.into_message()
    }

    pub fn assistant(content: Vec<Content>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    /// A single tool result, linked back to its call by `call_id`.
    pub fn tool_result(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![Content::ToolResult {
                call_id: call_id.into(),
                output: output.into(),
            }],
        }
    }

    /// Concatenate the text of all `Text` blocks (ignores other block kinds).
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// A single block within a message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Content {
    Text(String),
    /// Model reasoning. `signature` is Anthropic's cryptographic attestation,
    /// which must be echoed back verbatim when continuing after a tool call.
    Thinking {
        text: String,
        signature: Option<String>,
    },
    /// A tool invocation requested by the assistant. `arguments` is a JSON
    /// string (not parsed) so it can be forwarded to either API unchanged.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// The result of executing a tool call.
    ToolResult {
        call_id: String,
        output: String,
    },
    /// An image block (for vision models). Data is base64-encoded.
    Image {
        /// Base64-encoded image data (no data: prefix).
        data: String,
        /// MIME type: "image/png", "image/jpeg", "image/gif", "image/webp".
        media_type: String,
    },
}

/// A tool advertised to the model. `parameters` is a JSON Schema object.
#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A single inference request, independent of provider.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    /// Correlates one logical model call across exact transport/capacity
    /// retries. Payload-changing repairs must allocate a new identity.
    pub request_id: Option<String>,
    /// Zero-based replay number for the same logical request. Routed APIs use
    /// this to correlate a successful replay with an earlier failed attempt.
    pub retry_attempt: u32,
    /// True only for the primary request that answers the user's current turn.
    /// Provider wrappers may use this to keep auxiliary compaction, memory, and
    /// review requests on their normal route.
    pub user_turn: bool,
    /// Canonical objective for the active user turn, before provider-facing
    /// prompt guards or other local shaping. Auxiliary requests leave this
    /// unset.
    pub canonical_objective: Option<String>,
    /// Shared conversation history — `Arc` so the agent can clone the request
    /// cheaply (ref-count bump) instead of copying every message on every round.
    pub messages: Arc<Vec<Message>>,
    pub tools: Arc<[ToolSpec]>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Nucleus-sampling cutoff. Mainly used by recovery sampling (bumped on a
    /// retry after a content-less round). `None` leaves the provider default.
    pub top_p: Option<f32>,
    /// Penalty on already-seen tokens (OpenAI-compatible providers only;
    /// Anthropic has no equivalent and ignores it). Used by recovery sampling to
    /// break a repetition/garbled loop. `None` leaves the provider default.
    pub frequency_penalty: Option<f32>,
    /// When set, asks the provider to emit reasoning with this token budget
    /// (Anthropic extended thinking). Ignored by providers that don't support it.
    pub thinking_budget: Option<u32>,
    /// Abstract reasoning level for OpenAI-compatible endpoints that accept a
    /// `reasoning_effort` parameter (see [`ReasoningEffort`]). `None` omits the
    /// field, leaving the endpoint default. Ignored by the Anthropic adapter,
    /// which uses `thinking_budget` instead.
    pub reasoning_effort: Option<ReasoningEffort>,
    pub profile: RequestProfile,
}

/// Incremental output streamed to the caller as it arrives.
#[derive(Debug)]
pub enum StreamEvent {
    Text(String),
    Reasoning(String),
    /// An out-of-band note from the provider layer (e.g. a fallback switching
    /// models), surfaced to the user as a status line rather than model output.
    Status(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitBucket {
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub remaining: u64,
    #[serde(default)]
    pub reset_seconds: u64,
}

impl RateLimitBucket {
    pub fn has_data(&self) -> bool {
        self.limit > 0 || self.remaining > 0 || self.reset_seconds > 0
    }

    pub fn used(&self) -> u64 {
        self.limit.saturating_sub(self.remaining)
    }

    pub fn used_percent(&self) -> Option<u64> {
        (self.limit > 0).then(|| (self.used().saturating_mul(100) / self.limit).min(100))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitState {
    #[serde(default)]
    pub requests_min: RateLimitBucket,
    #[serde(default)]
    pub requests_hour: RateLimitBucket,
    #[serde(default)]
    pub tokens_min: RateLimitBucket,
    #[serde(default)]
    pub tokens_hour: RateLimitBucket,
    #[serde(default)]
    pub captured_at_unix_seconds: u64,
}

impl RateLimitState {
    pub fn has_data(&self) -> bool {
        self.requests_min.has_data()
            || self.requests_hour.has_data()
            || self.tokens_min.has_data()
            || self.tokens_hour.has_data()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Tokens served from a provider-side prompt cache (Anthropic cache_read).
    /// Billed at a discount to the normal input price (50% for OpenAI, ~10%
    /// for Anthropic); tracked separately so the token display can show
    /// cache hits distinctly.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Tokens written to the provider-side prompt cache this request (Anthropic
    /// cache_creation). Billed at ~125% of normal input price.
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Whether `input_tokens` already includes `cache_read_tokens` (and any
    /// `cache_creation_tokens`). True for OpenAI-compatible providers, where
    /// `prompt_tokens` is the total and `cached_tokens` is a subset; false for
    /// Anthropic, where `input_tokens` excludes the separately-reported cache
    /// tokens.
    ///
    /// Deprecated: prefer [`Usage::context_occupancy`], which is computed at the
    /// provider adapter where the semantics are known, so the agent layer no
    /// longer needs to branch on provider. Kept for now because call sites
    /// still construct `Usage` literals with it; will be removed once migration
    /// completes.
    #[serde(default)]
    pub input_includes_cache: bool,
    /// The total input tokens occupying the context window for this request, as
    /// the provider defines it. Computed at the provider adapter, where whether
    /// cache tokens are included in `input_tokens` (OpenAI) or reported
    /// separately (Anthropic) is known — so this is already the right number
    /// with no double-counting. The agent reads this directly instead of
    /// re-deriving occupancy from the other fields.
    #[serde(default)]
    pub context_occupancy: u64,
    /// Latest provider rate-limit buckets observed on a response. These are not
    /// token usage and do not affect [`Usage::is_zero`]; they ride along with
    /// usage so frontends can show whether failures are route/provider throttles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limits: Option<RateLimitState>,
    /// True when a token field was backfilled from a UTF-8-byte/4 estimate rather
    /// than provider-reported usage (the provider sent no usage frame, or an
    /// all-zeros one). Sticky across [`Usage::add`], so session totals disclose
    /// that they contain guessed numbers — surfaced as `usage_estimated` in
    /// `--report`.
    #[serde(default)]
    pub estimated: bool,
}

/// Provider/model-independent cost estimate. Values are integer micro-USD so
/// telemetry remains deterministic and serializable even when a provider does
/// not expose pricing metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostEstimate {
    pub input_microusd: u64,
    pub output_microusd: u64,
    pub total_microusd: u64,
}

impl CostEstimate {
    /// Calculate cost from a model's USD-per-million-token input/output rates.
    /// Returns `None` for absent, negative, or non-finite pricing.
    pub fn from_usage(usage: &Usage, pricing: Option<(f64, f64)>) -> Option<Self> {
        let (input_price, output_price) = pricing?;
        if !input_price.is_finite()
            || !output_price.is_finite()
            || input_price < 0.0
            || output_price < 0.0
        {
            return None;
        }
        // USD per million tokens × tokens = micro-USD.
        let input = (usage.effective_input_tokens() as f64 * input_price).round();
        let output = (usage.output_tokens as f64 * output_price).round();
        (input.is_finite() && output.is_finite()).then(|| {
            let input_microusd = input.min(u64::MAX as f64) as u64;
            let output_microusd = output.min(u64::MAX as f64) as u64;
            Self {
                input_microusd,
                output_microusd,
                total_microusd: input_microusd.saturating_add(output_microusd),
            }
        })
    }
}

/// Normalized usage record suitable for event streams, reports, and evals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizedUsage {
    pub provider: Option<String>,
    pub route: Option<String>,
    pub model: String,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostEstimate>,
}

impl NormalizedUsage {
    pub fn new(
        provider: Option<String>,
        route: Option<String>,
        model: impl Into<String>,
        usage: Usage,
        pricing: Option<(f64, f64)>,
    ) -> Self {
        let cost = CostEstimate::from_usage(&usage, pricing);
        Self {
            provider,
            route,
            model: model.into(),
            usage,
            cost,
        }
    }
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }

    pub fn is_zero(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_read_tokens == 0
            && self.cache_creation_tokens == 0
    }

    pub fn add(&mut self, other: Usage) {
        // Saturating: token counts come straight off the wire (`as_u64()`), so a
        // corrupt or hostile endpoint reporting near-`u64::MAX` must not panic an
        // overflow-checked build or wrap session totals to garbage in release.
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(other.cache_read_tokens);
        self.cache_creation_tokens = self
            .cache_creation_tokens
            .saturating_add(other.cache_creation_tokens);
        // "Latest observed": a booking that carries no rate-limit snapshot
        // (side-calls, error usage, estimates) must not wipe the last real one —
        // that made the rate-limit display blank out mid-session.
        if other.rate_limits.is_some() {
            self.rate_limits = other.rate_limits;
        }
        self.estimated |= other.estimated;
    }

    /// Deprecated: prefer [`Usage::context_occupancy`], which is set by the
    /// provider adapter and avoids re-deriving occupancy here. Kept callable so
    /// existing call sites continue to work during the migration.
    pub fn effective_input_tokens(&self) -> u64 {
        if self.input_includes_cache {
            self.input_tokens
        } else {
            self.input_tokens
                .saturating_add(self.cache_read_tokens)
                .saturating_add(self.cache_creation_tokens)
        }
    }
}

/// The fully-assembled assistant turn once a stream completes.
#[derive(Debug, Default, Serialize)]
pub struct Completion {
    pub content: Vec<Content>,
    pub usage: Usage,
    pub stop_reason: Option<String>,
}

impl Completion {
    /// The tool calls the assistant requested this turn, in order.
    pub fn tool_calls(&self) -> Vec<ToolCall<'_>> {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some(ToolCall {
                    id,
                    name,
                    arguments,
                }),
                _ => None,
            })
            .collect()
    }
}

/// Fallback ratio used when a provider omits usage. This is deliberately a
/// byte-based heuristic rather than a model tokenizer; provider-reported
/// counts always win when available.
const BYTES_PER_TOKEN: usize = 4;

/// Estimate tokens from UTF-8 byte length for usage/context fallback paths.
pub fn estimate_text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        0
    } else {
        text.len().div_ceil(BYTES_PER_TOKEN) as u64
    }
}

pub fn estimate_messages_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .flat_map(|m| &m.content)
        .map(estimate_content_tokens)
        .fold(0, u64::saturating_add)
}

pub fn estimate_content_tokens(content: &Content) -> u64 {
    match content {
        Content::Text(t) => estimate_text_tokens(t),
        Content::Thinking { text, signature } => estimate_text_tokens(text)
            .saturating_add(signature.as_deref().map(estimate_text_tokens).unwrap_or(0)),
        Content::ToolCall {
            id,
            name,
            arguments,
        } => estimate_text_tokens(id)
            .saturating_add(estimate_text_tokens(name))
            .saturating_add(estimate_text_tokens(arguments)),
        Content::ToolResult { call_id, output } => {
            estimate_text_tokens(call_id).saturating_add(estimate_text_tokens(output))
        }
        // Base64 image data: a rough token estimate from the encoded length;
        // include the MIME type because it is present in the wire payload.
        Content::Image { data, media_type } => {
            estimate_text_tokens(data).saturating_add(estimate_text_tokens(media_type))
        }
    }
}

pub fn estimate_completion_output_tokens(content: &[Content]) -> u64 {
    content
        .iter()
        .map(estimate_generated_content_tokens)
        .fold(0, u64::saturating_add)
}

/// Estimate only model-generated content in a completion. Provider-generated
/// wire metadata (tool-call ids and Anthropic thinking signatures) is included
/// in [`estimate_content_tokens`] for replay/context accounting, but must not
/// inflate a completion-token fallback.
fn estimate_generated_content_tokens(content: &Content) -> u64 {
    match content {
        Content::Text(text) => estimate_text_tokens(text),
        Content::Thinking { text, .. } => estimate_text_tokens(text),
        Content::ToolCall {
            name, arguments, ..
        } => estimate_text_tokens(name).saturating_add(estimate_text_tokens(arguments)),
        Content::ToolResult { output, .. } => estimate_text_tokens(output),
        Content::Image { data, .. } => estimate_text_tokens(data),
    }
}

/// Estimate the input tokens for one complete provider request.
///
/// Providers remain authoritative when they return usage. This bounded
/// UTF-8-byte heuristic is only used for context preflight and for adapters
/// that omit usage. Keeping messages and advertised tools in one function is
/// important: omitting tool schemas here makes fallback usage and context
/// admission disagree about the same request.
pub fn estimate_request_input_tokens(messages: &[Message], tools: &[ToolSpec]) -> u64 {
    estimate_messages_tokens(messages).saturating_add(estimate_tool_schema_tokens(tools))
}

/// Estimate the serialized cost of advertised tool definitions.
pub fn estimate_tool_schema_tokens(tools: &[ToolSpec]) -> u64 {
    tools.iter().fold(0, |total, tool| {
        let tool_tokens = estimate_text_tokens(&tool.name)
            .saturating_add(estimate_text_tokens(&tool.description))
            .saturating_add(estimate_text_tokens(&tool.parameters.to_string()));
        total.saturating_add(tool_tokens)
    })
}

/// A borrowed view of a requested tool call.
pub struct ToolCall<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a str,
}

#[cfg(test)]
mod tests {
    use super::{
        Content, CostEstimate, Message, PromptInput, RateLimitBucket, RateLimitState,
        ReasoningEffort, ToolSpec, Usage, estimate_completion_output_tokens,
        estimate_request_input_tokens, estimate_text_tokens,
    };

    #[test]
    fn effort_escalation_steps_up_and_saturates() {
        assert_eq!(ReasoningEffort::Minimal.next_higher(), ReasoningEffort::Low);
        assert_eq!(ReasoningEffort::Medium.next_higher(), ReasoningEffort::High);
        assert_eq!(ReasoningEffort::High.next_higher(), ReasoningEffort::Xhigh);
        assert_eq!(ReasoningEffort::Xhigh.next_higher(), ReasoningEffort::Xhigh);
    }

    #[test]
    fn add_preserves_last_observed_rate_limits_and_sticks_estimated() {
        let mut totals = Usage {
            input_tokens: 100,
            output_tokens: 10,
            rate_limits: Some(RateLimitState {
                requests_min: RateLimitBucket {
                    limit: 10,
                    remaining: 8,
                    reset_seconds: 1,
                },
                ..RateLimitState::default()
            }),
            ..Usage::default()
        };

        // A booking with no rate-limit snapshot (side-call, error usage,
        // estimate) must not wipe the last observed one.
        totals.add(Usage {
            input_tokens: 50,
            output_tokens: 5,
            estimated: true,
            ..Usage::default()
        });
        assert_eq!(totals.input_tokens, 150);
        assert!(
            totals.rate_limits.is_some(),
            "zero-snapshot add wiped rate limits"
        );
        // Estimated is sticky: once any component was guessed, totals say so.
        assert!(totals.estimated);
        totals.add(Usage {
            input_tokens: 5,
            ..Usage::default()
        });
        assert!(totals.estimated, "estimated must not reset");

        // A booking that carries a fresh snapshot replaces the old one.
        totals.add(Usage {
            rate_limits: Some(RateLimitState {
                requests_min: RateLimitBucket {
                    limit: 10,
                    remaining: 3,
                    reset_seconds: 2,
                },
                ..RateLimitState::default()
            }),
            ..Usage::default()
        });
        assert_eq!(totals.rate_limits.unwrap().requests_min.remaining, 3);
    }

    #[test]
    fn cost_estimate_is_optional_and_uses_micro_usd() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Usage::default()
        };
        let cost = CostEstimate::from_usage(&usage, Some((2.0, 4.0))).unwrap();
        assert_eq!(cost.input_microusd, 200);
        assert_eq!(cost.output_microusd, 200);
        assert_eq!(cost.total_microusd, 400);
        assert!(CostEstimate::from_usage(&usage, None).is_none());
    }

    #[test]
    fn typed_prompt_preserves_text_and_image_blocks() {
        let input = PromptInput::text("inspect this").image("aGVsbG8=", "image/png");
        let message = input.clone().into_message();
        assert_eq!(input.text_content(), "inspect this");
        assert!(matches!(message.content[0], Content::Text(_)));
        assert!(matches!(message.content[1], Content::Image { .. }));
    }

    #[test]
    fn request_estimate_uses_the_same_content_and_tool_schema_paths() {
        let messages = vec![
            Message::system("rules"),
            Message::assistant(vec![Content::Thinking {
                text: "think".into(),
                signature: Some("sig".into()),
            }]),
            Message::assistant(vec![Content::ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: r#"{"path":"README.md"}"#.into(),
            }]),
            Message::tool_result("call_1", "contents"),
        ];
        let tools = vec![ToolSpec {
            name: "read".into(),
            description: "read a file".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        }];

        let expected_messages = messages
            .iter()
            .flat_map(|message| &message.content)
            .map(super::estimate_content_tokens)
            .fold(0, u64::saturating_add);
        let expected_tools = estimate_text_tokens("read")
            + estimate_text_tokens("read a file")
            + estimate_text_tokens(&tools[0].parameters.to_string());
        assert_eq!(
            estimate_request_input_tokens(&messages, &tools),
            expected_messages.saturating_add(expected_tools)
        );
        assert_eq!(
            estimate_completion_output_tokens(&messages[1].content),
            estimate_text_tokens("think")
        );
    }

    #[test]
    fn estimate_arithmetic_saturates_and_empty_text_is_zero() {
        assert_eq!(estimate_text_tokens(""), 0);
        let usage = Usage {
            input_tokens: u64::MAX,
            cache_read_tokens: u64::MAX,
            cache_creation_tokens: u64::MAX,
            input_includes_cache: false,
            ..Usage::default()
        };
        assert_eq!(usage.effective_input_tokens(), u64::MAX);
    }
}
