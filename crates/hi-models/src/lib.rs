//! Default model IDs for the `hi` CLI, loaded from an embedded JSON manifest.
//!
//! Provides canonical model identifiers so that first-run users get a working
//! model without manual configuration, and so that model IDs stay current as
//! providers release new versions. The manifest is a compile-time
//! `include_str!` so there is no runtime file dependency.
//!
//! Inspired by grok-build's `xai-grok-models` crate.
//!
//! # Quick start
//!
//! ```
//! let model = hi_models::default_model();
//! assert!(!model.is_empty());
//! ```

use std::sync::OnceLock;

use serde::Deserialize;

/// The embedded default-models manifest (compile-time).
pub const DEFAULT_MODELS_JSON: &str = include_str!("../default_models.json");

/// A single entry in the default-models manifest.
#[derive(Debug, Deserialize)]
struct DefaultModelEntry {
    /// The canonical model ID (e.g. `"grok-4"`).
    id: String,
}

/// The full default-models manifest, keyed by role.
#[derive(Debug, Deserialize)]
struct DefaultModels {
    default: DefaultModelEntry,
    web_search: Option<DefaultModelEntry>,
    image_description: Option<DefaultModelEntry>,
    session_summary: Option<DefaultModelEntry>,
}

/// Parse the embedded manifest. Panics if the embedded JSON is malformed
/// (a compile-time data error, not a runtime condition).
fn parse() -> DefaultModels {
    serde_json::from_str(DEFAULT_MODELS_JSON)
        .expect("hi-models: embedded default_models.json is malformed")
}

/// The cached parsed manifest, initialized once on first access.
static PARSED: OnceLock<DefaultModels> = OnceLock::new();

/// The parsed manifest, cached for the process lifetime.
fn models() -> &'static DefaultModels {
    PARSED.get_or_init(parse)
}

/// The default model ID for general chat / coding.
pub fn default_model() -> &'static str {
    models().default.id.as_str()
}

/// The default model ID for web-search-augmented queries.
/// Falls back to [`default_model`] if the manifest doesn't specify one.
pub fn default_web_search_model() -> &'static str {
    models()
        .web_search
        .as_ref()
        .map(|e| e.id.as_str())
        .unwrap_or_else(default_model)
}

/// The default model ID for image description / vision tasks.
/// Falls back to [`default_model`] if the manifest doesn't specify one.
pub fn default_image_description_model() -> &'static str {
    models()
        .image_description
        .as_ref()
        .map(|e| e.id.as_str())
        .unwrap_or_else(default_model)
}

/// The default model ID for session summarization.
/// Falls back to [`default_model`] if the manifest doesn't specify one.
pub fn default_session_summary_model() -> &'static str {
    models()
        .session_summary
        .as_ref()
        .map(|e| e.id.as_str())
        .unwrap_or_else(default_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_nonempty() {
        assert!(!default_model().is_empty());
    }

    #[test]
    fn web_search_falls_back_to_default() {
        let ws = default_web_search_model();
        assert!(!ws.is_empty());
    }

    #[test]
    fn image_description_falls_back_to_default() {
        let id = default_image_description_model();
        assert!(!id.is_empty());
    }

    #[test]
    fn session_summary_falls_back_to_default() {
        let ss = default_session_summary_model();
        assert!(!ss.is_empty());
    }

    #[test]
    fn manifest_is_valid_json() {
        let v: serde_json::Value = serde_json::from_str(DEFAULT_MODELS_JSON).unwrap();
        assert!(v.is_object());
        assert!(v["default"].is_object());
        assert!(v["default"]["id"].is_string());
    }
}
