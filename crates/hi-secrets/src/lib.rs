//! Secret sanitization for outbound data — traces, reports, tool output, telemetry.
//!
//! Regex-based redaction of API keys, tokens, private keys, and credentials in
//! arbitrary text and JSON. Inspired by grok-build's `xai-grok-secrets` crate,
//! adapted for hi's outbound channels (trace events, tool output, delegate logs).
//!
//! # Quick start
//!
//! ```
//! use hi_secrets::redact_secrets;
//!
//! let dirty = "api_key=sk-abcdefghijklmnopqrstuvwxyz123456";
//! let clean = redact_secrets(dirty);
//! assert!(clean.contains("[REDACTED_SECRET]"));
//! assert!(!clean.contains("sk-abcdefghijklmnopqrstuvwxyz123456"));
//! ```

mod sanitizer;

pub use sanitizer::{
    redact_json_string_values, redact_secrets, redact_url, redact_user_paths, walk_json_strings,
};

/// Environment variable names that must never be inherited by untrusted
/// subprocesses or copied into incident bundles.
pub const SECRET_ENV_NAMES: &[&str] = &[
    "HI_API_KEY",
    "HI_WEB_SEARCH_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "PIPENETWORK_API_KEY",
    "OLLAMA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
];

#[cfg(test)]
mod secret_env_tests {
    #[test]
    fn secret_env_names_include_pipe_and_hi_keys() {
        assert!(super::SECRET_ENV_NAMES.contains(&"PIPENETWORK_API_KEY"));
        assert!(super::SECRET_ENV_NAMES.contains(&"HI_API_KEY"));
    }
}
