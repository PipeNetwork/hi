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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestProfile {
    pub compat: CompatMode,
    pub tool_mode: ToolMode,
    pub stream_usage: Option<bool>,
}

impl Default for RequestProfile {
    fn default() -> Self {
        Self {
            compat: CompatMode::Auto,
            tool_mode: ToolMode::Auto,
            stream_usage: None,
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
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A single inference request, independent of provider.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
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
    /// True when a token field was backfilled from a chars/4 estimate rather
    /// than provider-reported usage (the provider sent no usage frame, or an
    /// all-zeros one). Sticky across [`Usage::add`], so session totals disclose
    /// that they contain guessed numbers — surfaced as `usage_estimated` in
    /// `--report`.
    #[serde(default)]
    pub estimated: bool,
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
            self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens
        }
    }
}

/// The fully-assembled assistant turn once a stream completes.
#[derive(Debug, Default)]
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

const CHARS_PER_TOKEN: usize = 4;

pub fn estimate_text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        0
    } else {
        text.len().div_ceil(CHARS_PER_TOKEN) as u64
    }
}

pub fn estimate_messages_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .flat_map(|m| &m.content)
        .map(estimate_content_tokens)
        .sum()
}

pub fn estimate_content_tokens(content: &Content) -> u64 {
    match content {
        Content::Text(t) => estimate_text_tokens(t),
        Content::Thinking { text, .. } => estimate_text_tokens(text),
        Content::ToolCall {
            name, arguments, ..
        } => estimate_text_tokens(name) + estimate_text_tokens(arguments),
        Content::ToolResult { output, .. } => estimate_text_tokens(output),
        // Base64 image data: a rough token estimate from the encoded length.
        Content::Image { data, .. } => estimate_text_tokens(data),
    }
}

pub fn estimate_completion_output_tokens(content: &[Content]) -> u64 {
    content.iter().map(estimate_content_tokens).sum()
}

/// A borrowed view of a requested tool call.
pub struct ToolCall<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a str,
}

#[cfg(test)]
mod tests {
    use super::{RateLimitBucket, RateLimitState, Usage};

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
}
