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
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Tokens served from a provider-side prompt cache (Anthropic cache_read).
    /// Billed at a discount to the normal input price (50% for OpenAI, ~10%
    /// for Anthropic); tracked separately so cost calculations can apply the
    /// discount.
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
    /// tokens. Determines how `context_used` is computed so it isn't
    /// double-counted.
    #[serde(default)]
    pub input_includes_cache: bool,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub fn is_zero(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_read_tokens == 0
            && self.cache_creation_tokens == 0
    }

    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_creation_tokens += other.cache_creation_tokens;
    }

    /// The effective number of input tokens occupying the context window for
    /// this request. On providers where `input_tokens` already includes the
    /// cached subset (OpenAI), that subset is not added again; on providers
    /// where cache tokens are reported separately (Anthropic), it is.
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
