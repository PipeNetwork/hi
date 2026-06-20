//! Provider-neutral conversation model.
//!
//! Every adapter (OpenAI-compatible, Anthropic) translates these types to and
//! from its own wire format, so the agent core never sees provider specifics.
//! The shape is a superset modeled on content blocks (Anthropic-style) because
//! that round-trips both APIs cleanly — including reasoning/thinking, which the
//! flat OpenAI message shape can't represent on its own.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// When set, asks the provider to emit reasoning with this token budget
    /// (Anthropic extended thinking). Ignored by providers that don't support it.
    pub thinking_budget: Option<u32>,
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

#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
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

/// A borrowed view of a requested tool call.
pub struct ToolCall<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a str,
}
