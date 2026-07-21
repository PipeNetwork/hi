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
