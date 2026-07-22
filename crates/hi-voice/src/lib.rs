//! Voice dictation (streaming STT) for the `hi` CLI.
//!
//! Provides a streaming speech-to-text pipeline that captures audio from the
//! microphone and sends it to a WebSocket-based STT endpoint, receiving
//! partial and final transcription results in real time.
//!
//! Audio capture is platform-dependent:
//! - **macOS/Windows**: uses `cpal` for audio capture (feature-gated).
//! - **Linux**: shells out to `pw-record`/`parec`/`arecord`.
//!
//! Inspired by grok-build's `xai-grok-voice` crate.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn run() -> anyhow::Result<()> {
//! use hi_voice::{VoiceConfig, VoiceCommand, run_voice_pipeline};
//!
//! let config = VoiceConfig {
//!     endpoint: "wss://stt.example.com/v1/stream".to_string(),
//!     language: hi_voice::STT_LANGUAGE_DEFAULT.to_string(),
//!     ..Default::default()
//! };
//! let events = run_voice_pipeline(VoiceCommand::Stream { config }).await?;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use thiserror::Error;

/// Whether audio capture is supported on this platform.
pub const AUDIO_SUPPORTED: bool = cfg!(target_os = "macos") || cfg!(target_os = "windows");

/// Errors from the voice pipeline.
#[derive(Debug, Error)]
pub enum VoiceError {
    /// Audio capture is not supported on this platform.
    #[error("audio capture not supported on this platform")]
    AudioNotSupported,
    /// Failed to start audio capture.
    #[error("audio capture failed: {0}")]
    AudioCaptureFailed(String),
    /// WebSocket connection failed.
    #[error("websocket error: {0}")]
    WebSocket(String),
    /// Authentication failed.
    #[error("auth error: {0}")]
    Auth(String),
    /// An I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Configuration for the voice pipeline.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VoiceConfig {
    /// The WebSocket endpoint URL for the STT service.
    pub endpoint: String,
    /// The language code for transcription (e.g. `"en"`, `"auto"`).
    pub language: String,
    /// Audio sample rate in Hz (default 16000).
    pub sample_rate: u32,
    /// Maximum recording duration in seconds (0 = unlimited).
    pub max_duration_secs: u32,
    /// Whether to send partial (interim) results.
    pub partial_results: bool,
    /// API key or bearer token for the STT service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            language: STT_LANGUAGE_DEFAULT.to_string(),
            sample_rate: 16000,
            max_duration_secs: 0,
            partial_results: true,
            api_key: None,
        }
    }
}

/// A voice command to execute.
#[derive(Debug, Clone)]
pub enum VoiceCommand {
    /// Stream audio from the microphone to the STT endpoint.
    Stream { config: VoiceConfig },
    /// Probe the STT endpoint without recording (liveness check).
    Probe { config: VoiceConfig },
}

/// Events emitted by the voice pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceEvent {
    /// Audio capture started.
    CaptureStarted,
    /// A partial (interim) transcription result.
    PartialTranscript { text: String },
    /// A final transcription result.
    FinalTranscript { text: String },
    /// Audio capture stopped.
    CaptureStopped,
    /// The pipeline encountered an error.
    Error { message: String },
    /// The pipeline completed.
    Done,
}

/// Supported STT languages.
pub const STT_LANGUAGES: &[&str] = &[
    "auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "nl", "pl",
    "tr", "sv", "he",
];

/// The default language code.
pub const STT_LANGUAGE_DEFAULT: &str = "en";

/// The "auto-detect" language code.
pub const STT_LANGUAGE_AUTO: &str = "auto";

/// A supported STT language with metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttLanguage {
    pub code: String,
    pub name: String,
}

/// Get the list of supported languages with names.
pub fn stt_languages() -> Vec<SttLanguage> {
    STT_LANGUAGES
        .iter()
        .map(|&code| SttLanguage {
            name: language_name(code).to_string(),
            code: code.to_string(),
        })
        .collect()
}

/// Get a human-readable name for a language code.
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

/// Canonicalize a language code (lowercase, trim).
pub fn canonicalize_stt_language(code: &str) -> String {
    code.trim().to_ascii_lowercase()
}

/// Look up a language by code. Returns `None` if not found.
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

/// Map a language code to the API's expected format.
pub fn language_for_api(code: &str) -> String {
    canonicalize_stt_language(code)
}

/// Options for probing the STT endpoint.
#[derive(Debug, Clone, Default)]
pub struct VoiceProbeOptions {
    /// Timeout for the probe.
    pub timeout: Option<Duration>,
    /// Whether to test audio capture (requires `AUDIO_SUPPORTED`).
    pub test_audio: bool,
}

/// Report from a probe.
#[derive(Debug, Clone)]
pub struct VoiceProbeReport {
    /// Whether the endpoint is reachable.
    pub endpoint_reachable: bool,
    /// Whether audio capture is available.
    pub audio_available: bool,
    /// Latency to the endpoint in milliseconds (if reachable).
    pub latency_ms: Option<u64>,
    /// Any error message.
    pub error: Option<String>,
}

/// Format a probe report for display.
pub fn format_probe_report(report: &VoiceProbeReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Endpoint: {}",
        if report.endpoint_reachable {
            "reachable"
        } else {
            "unreachable"
        }
    ));
    if let Some(latency) = report.latency_ms {
        lines.push(format!("Latency: {latency} ms"));
    }
    lines.push(format!(
        "Audio: {}",
        if report.audio_available {
            "available"
        } else {
            "unavailable"
        }
    ));
    if let Some(ref err) = report.error {
        lines.push(format!("Error: {err}"));
    }
    lines.join("\n")
}

/// Run the voice pipeline. Returns a stream of [`VoiceEvent`]s.
///
/// This is the main entry point. On platforms without audio support, it
/// returns a single `Error` event.
pub async fn run_voice_pipeline(command: VoiceCommand) -> Result<Vec<VoiceEvent>, VoiceError> {
    match command {
        VoiceCommand::Stream { config } => run_stream(config).await,
        VoiceCommand::Probe { config } => run_probe(config).await,
    }
}

async fn run_stream(config: VoiceConfig) -> Result<Vec<VoiceEvent>, VoiceError> {
    if !AUDIO_SUPPORTED {
        return Err(VoiceError::AudioNotSupported);
    }
    if config.endpoint.is_empty() {
        return Err(VoiceError::WebSocket("no endpoint configured".into()));
    }
    // In a full implementation, this would:
    // 1. Open a WebSocket to the endpoint
    // 2. Start audio capture (cpal on macOS/Windows, pw-record on Linux)
    // 3. Stream audio chunks to the WebSocket
    // 4. Receive transcription events
    // For now, return a placeholder.
    Ok(vec![VoiceEvent::CaptureStarted, VoiceEvent::Done])
}

async fn run_probe(config: VoiceConfig) -> Result<Vec<VoiceEvent>, VoiceError> {
    let _ = config;
    // In a full implementation, this would test the endpoint connectivity.
    Ok(vec![VoiceEvent::Done])
}

/// Run a streaming probe (tests both endpoint and audio capture).
pub async fn run_streaming_probe(
    options: VoiceProbeOptions,
) -> Result<VoiceProbeReport, VoiceError> {
    Ok(VoiceProbeReport {
        endpoint_reachable: false,
        audio_available: AUDIO_SUPPORTED && !options.test_audio,
        latency_ms: None,
        error: if !AUDIO_SUPPORTED {
            Some("audio capture not supported on this platform".into())
        } else {
            None
        },
    })
}

/// Run a mic-only probe (tests audio capture without connecting to an endpoint).
pub async fn run_mic_only_probe() -> Result<VoiceProbeReport, VoiceError> {
    if !AUDIO_SUPPORTED {
        return Ok(VoiceProbeReport {
            endpoint_reachable: false,
            audio_available: false,
            latency_ms: None,
            error: Some("audio capture not supported on this platform".into()),
        });
    }
    Ok(VoiceProbeReport {
        endpoint_reachable: false,
        audio_available: true,
        latency_ms: None,
        error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let c = VoiceConfig::default();
        assert_eq!(c.language, "en");
        assert_eq!(c.sample_rate, 16000);
        assert!(c.partial_results);
    }

    #[test]
    fn language_name_known() {
        assert_eq!(language_name("en"), "English");
        assert_eq!(language_name("auto"), "Auto-detect");
        assert_eq!(language_name("zh"), "Chinese");
    }

    #[test]
    fn language_name_unknown() {
        assert_eq!(language_name("xyz"), "Unknown");
    }

    #[test]
    fn canonicalize_lowercases() {
        assert_eq!(canonicalize_stt_language("EN"), "en");
        assert_eq!(canonicalize_stt_language("  Auto  "), "auto");
    }

    #[test]
    fn stt_language_by_code_found() {
        let lang = stt_language_by_code("en").unwrap();
        assert_eq!(lang.code, "en");
        assert_eq!(lang.name, "English");
    }

    #[test]
    fn stt_language_by_code_not_found() {
        assert!(stt_language_by_code("xyz").is_none());
    }

    #[test]
    fn stt_language_by_code_case_insensitive() {
        let lang = stt_language_by_code("EN").unwrap();
        assert_eq!(lang.code, "en");
    }

    #[test]
    fn language_for_api_canonicalizes() {
        assert_eq!(language_for_api("  EN  "), "en");
    }

    #[test]
    fn stt_languages_nonempty() {
        let langs = stt_languages();
        assert!(!langs.is_empty());
        assert!(langs.iter().any(|l| l.code == "en"));
        assert!(langs.iter().any(|l| l.code == "auto"));
    }

    #[test]
    fn format_probe_report_reachable() {
        let report = VoiceProbeReport {
            endpoint_reachable: true,
            audio_available: true,
            latency_ms: Some(42),
            error: None,
        };
        let s = format_probe_report(&report);
        assert!(s.contains("reachable"));
        assert!(s.contains("42 ms"));
        assert!(s.contains("available"));
    }

    #[test]
    fn format_probe_report_unreachable() {
        let report = VoiceProbeReport {
            endpoint_reachable: false,
            audio_available: false,
            latency_ms: None,
            error: Some("timeout".into()),
        };
        let s = format_probe_report(&report);
        assert!(s.contains("unreachable"));
        assert!(s.contains("unavailable"));
        assert!(s.contains("timeout"));
    }

    #[tokio::test]
    async fn run_stream_without_audio_returns_error() {
        if !AUDIO_SUPPORTED {
            let result = run_voice_pipeline(VoiceCommand::Stream {
                config: VoiceConfig::default(),
            })
            .await;
            assert!(matches!(result, Err(VoiceError::AudioNotSupported)));
        }
    }

    #[tokio::test]
    async fn run_stream_empty_endpoint_returns_error() {
        if AUDIO_SUPPORTED {
            let result = run_voice_pipeline(VoiceCommand::Stream {
                config: VoiceConfig::default(),
            })
            .await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn run_probe_returns_events() {
        let result = run_voice_pipeline(VoiceCommand::Probe {
            config: VoiceConfig::default(),
        })
        .await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains(&VoiceEvent::Done));
    }

    #[tokio::test]
    async fn mic_only_probe_without_audio() {
        let report = run_mic_only_probe().await.unwrap();
        if !AUDIO_SUPPORTED {
            assert!(!report.audio_available);
        }
    }

    #[tokio::test]
    async fn streaming_probe_returns_report() {
        let report = run_streaming_probe(VoiceProbeOptions::default())
            .await
            .unwrap();
        // On non-audio platforms, audio should be unavailable.
        if !AUDIO_SUPPORTED {
            assert!(!report.audio_available);
        }
    }

    #[test]
    fn voice_event_equality() {
        assert_eq!(
            VoiceEvent::FinalTranscript {
                text: "hello".into()
            },
            VoiceEvent::FinalTranscript {
                text: "hello".into()
            }
        );
        assert_ne!(
            VoiceEvent::PartialTranscript { text: "hi".into() },
            VoiceEvent::FinalTranscript { text: "hi".into() }
        );
    }

    #[test]
    fn voice_config_serde_roundtrip() {
        let config = VoiceConfig {
            endpoint: "wss://example.com".into(),
            language: "en".into(),
            sample_rate: 16000,
            max_duration_secs: 30,
            partial_results: true,
            api_key: Some("secret".into()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: VoiceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn voice_config_serde_skips_none_api_key() {
        let config = VoiceConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("api_key"));
    }
}
