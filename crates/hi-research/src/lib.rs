//! Thin HTTP client for Pipe `POST /v1/research` and page reads.
//!
//! Interactive `hi` must not host SearXNG or Firecrawl. Unknown request fields
//! are rejected server-side, so this crate serializes only the live contract.

mod client;
mod error;
mod types;

pub use client::{ResearchClient, ResearchClientConfig};
pub use error::{ResearchError, ResearchErrorKind};
pub use types::*;

use std::sync::OnceLock;

static PROCESS_DEFAULTS: OnceLock<ResearchClientConfig> = OnceLock::new();

/// Store origin/key once at CLI startup so tools do not grow extra arguments.
pub fn install_process_defaults(origin: impl Into<String>, api_key: impl Into<String>) {
    let _ = PROCESS_DEFAULTS.set(ResearchClientConfig {
        origin: origin.into(),
        api_key: api_key.into(),
    });
}

pub fn process_defaults() -> Option<&'static ResearchClientConfig> {
    PROCESS_DEFAULTS.get()
}

pub fn credentials_configured() -> bool {
    if process_defaults().is_some_and(|cfg| !cfg.api_key.trim().is_empty()) {
        return true;
    }
    std::env::var("PIPENETWORK_API_KEY")
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// Normalize an origin that may be `https://host`, `https://host/`, or `https://host/v1`.
pub fn normalize_origin(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    trimmed
        .strip_suffix("/v1")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
        .to_string()
}

pub fn parse_judge_choice(value: &str) -> Option<JudgeChoice> {
    match value.trim().to_ascii_lowercase().as_str() {
        "model" => Some(JudgeChoice::Model),
        "tests" | "test" | "verify" => Some(JudgeChoice::Tests),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_judge_model_and_tests() {
        assert_eq!(parse_judge_choice("model"), Some(JudgeChoice::Model));
        assert_eq!(parse_judge_choice(" tests "), Some(JudgeChoice::Tests));
        assert_eq!(parse_judge_choice("nope"), None);
    }

    #[test]
    fn normalize_strips_v1_suffix() {
        assert_eq!(
            normalize_origin("https://api.pipenetwork.ai/v1/"),
            "https://api.pipenetwork.ai"
        );
    }
}
