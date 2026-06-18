//! `pi-ai` — provider-neutral LLM types, the [`Provider`] trait, and adapters
//! for OpenAI-compatible and Anthropic backends.

pub mod anthropic;
mod http;
pub mod openai;
pub mod provider;
pub mod registry;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAiProvider;
pub use provider::Provider;
pub use registry::{ModelInfo, Registry};
pub use types::{
    ChatRequest, Completion, Content, Message, Role, StreamEvent, ToolCall, ToolSpec, Usage,
};
