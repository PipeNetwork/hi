//! Voice dictation (local speech-to-text) for the `hi` TUI.
//!
//! Ctrl+Space toggles recording; stopping transcribes what was captured and
//! inserts it into the prompt. Everything runs on the machine — audio is
//! captured with `cpal` and transcribed by Whisper through `whisper-rs`
//! (whisper.cpp), Metal-accelerated on Apple Silicon. No audio leaves the host
//! and no API key is involved.
//!
//! The two costs are a one-time ~1.6 GB model download (see [`model`]) and the
//! seconds it takes to load that model, which is why [`stt::Transcriber`] is
//! built once and reused.
//!
//! # Shape
//!
//! ```no_run
//! # fn run() -> Result<(), hi_voice::VoiceError> {
//! use hi_voice::{Recorder, Transcriber, VoiceConfig};
//!
//! let transcriber = Transcriber::load(&VoiceConfig::default())?; // once
//! let recorder = Recorder::start()?;                             // Ctrl+Space
//! // ... user speaks ...
//! let audio = recorder.stop()?;                                  // Ctrl+Space
//! let text = transcriber.transcribe(&audio)?;
//! # let _ = text;
//! # Ok(())
//! # }
//! ```

pub mod audio;
pub mod model;
pub mod stt;

pub use audio::{Recorder, WHISPER_SAMPLE_RATE};
pub use model::{
    DEFAULT_MODEL_BYTES, DEFAULT_MODEL_FILE, MODEL_REPO_URL, download_model, model_dir, model_path,
    resolve_model_path,
};
pub use stt::Transcriber;

use thiserror::Error;

/// Whether microphone capture is supported on this platform.
///
/// `cpal` covers all three desktop platforms; Linux capture goes through ALSA,
/// which is present on any desktop system worth dictating on.
pub const AUDIO_SUPPORTED: bool = cfg!(any(
    target_os = "macos",
    target_os = "windows",
    target_os = "linux"
));

/// Errors from the voice pipeline.
#[derive(Debug, Clone, Error)]
pub enum VoiceError {
    /// Audio capture is not supported on this platform.
    #[error("audio capture is not supported on this platform")]
    AudioNotSupported,
    /// Opening or reading the microphone failed.
    #[error("audio capture failed: {0}")]
    AudioCaptureFailed(String),
    /// The Whisper model file was not found.
    #[error("whisper model not found at {path}\n{hint}")]
    ModelMissing {
        /// Where we looked.
        path: String,
        /// How to obtain it.
        hint: String,
    },
    /// Fetching the model failed.
    #[error("downloading the voice model failed: {0}")]
    Download(String),
    /// Whisper failed to load or decode.
    #[error("transcription failed: {0}")]
    Transcribe(String),
}

/// Configuration for voice dictation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VoiceConfig {
    /// Language code to transcribe as, or [`STT_LANGUAGE_AUTO`] to detect.
    pub language: String,
    /// Explicit model path. When unset, [`model::resolve_model_path`] falls
    /// back to `HI_VOICE_MODEL` and then the default location.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            language: STT_LANGUAGE_DEFAULT.to_string(),
            model_path: None,
        }
    }
}

/// Supported transcription languages.
pub const STT_LANGUAGES: &[&str] = &[
    "auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "nl", "pl",
    "tr", "sv", "he",
];

/// The default language code.
pub const STT_LANGUAGE_DEFAULT: &str = "en";

/// The "auto-detect" language code.
pub const STT_LANGUAGE_AUTO: &str = "auto";

/// A supported language with its display name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttLanguage {
    /// Code as passed to Whisper.
    pub code: String,
    /// Human-readable name.
    pub name: String,
}

/// The supported languages with display names.
pub fn stt_languages() -> Vec<SttLanguage> {
    STT_LANGUAGES
        .iter()
        .map(|&code| SttLanguage {
            name: language_name(code).to_string(),
            code: code.to_string(),
        })
        .collect()
}

/// Human-readable name for a language code.
pub fn language_name(code: &str) -> &'static str {
    match code {
        "auto" => "Auto-detect",
        "en" => "English",
        "es" => "Spanish",
        "fr" => "French",
        "de" => "German",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ru" => "Russian",
        "ja" => "Japanese",
        "ko" => "Korean",
        "zh" => "Chinese",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "nl" => "Dutch",
        "pl" => "Polish",
        "tr" => "Turkish",
        "sv" => "Swedish",
        "he" => "Hebrew",
        _ => "Unknown",
    }
}

/// Canonicalize a language code (trimmed, lowercased).
pub fn canonicalize_stt_language(code: &str) -> String {
    code.trim().to_ascii_lowercase()
}

/// Look up a supported language by code.
pub fn stt_language_by_code(code: &str) -> Option<SttLanguage> {
    let canonical = canonicalize_stt_language(code);
    STT_LANGUAGES
        .iter()
        .find(|&&c| c == canonical)
        .map(|&c| SttLanguage {
            code: c.to_string(),
            name: language_name(c).to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_the_default_language_and_no_explicit_model() {
        let config = VoiceConfig::default();
        assert_eq!(config.language, STT_LANGUAGE_DEFAULT);
        assert!(config.model_path.is_none());
    }

    #[test]
    fn language_name_known() {
        assert_eq!(language_name("en"), "English");
        assert_eq!(language_name("auto"), "Auto-detect");
    }

    #[test]
    fn language_name_unknown() {
        assert_eq!(language_name("xx"), "Unknown");
    }

    #[test]
    fn canonicalize_lowercases_and_trims() {
        assert_eq!(canonicalize_stt_language("  EN "), "en");
    }

    #[test]
    fn stt_language_by_code_is_case_insensitive() {
        assert_eq!(stt_language_by_code("FR").unwrap().name, "French");
        assert!(stt_language_by_code("nope").is_none());
    }

    #[test]
    fn every_supported_language_has_a_name() {
        let languages = stt_languages();
        assert_eq!(languages.len(), STT_LANGUAGES.len());
        assert!(
            languages.iter().all(|l| l.name != "Unknown"),
            "every code in STT_LANGUAGES needs a display name: {languages:?}"
        );
    }

    #[test]
    fn a_missing_model_error_renders_path_and_hint() {
        let err = VoiceError::ModelMissing {
            path: "/tmp/x.bin".into(),
            hint: "download it".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("/tmp/x.bin"), "{rendered}");
        assert!(rendered.contains("download it"), "{rendered}");
    }
}
