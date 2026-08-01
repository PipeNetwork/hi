//! Same-API stand-ins for builds without the `audio` feature.
//!
//! Callers stay uniform: [`crate::AUDIO_SUPPORTED`] is `false`, and any path
//! that still reaches capture or transcription gets
//! [`VoiceError::AudioNotSupported`] instead of a compile-time hole.

use crate::{VoiceConfig, VoiceError};

/// Mirrors `audio::WHISPER_SAMPLE_RATE`.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Microphone capture stub; construction always fails.
#[derive(Debug)]
pub struct Recorder {}

impl Recorder {
    pub fn start() -> Result<Self, VoiceError> {
        Err(VoiceError::AudioNotSupported)
    }

    pub fn stop(self) -> Result<Vec<f32>, VoiceError> {
        Err(VoiceError::AudioNotSupported)
    }
}

/// Transcription stub; loading always fails.
#[derive(Debug)]
pub struct Transcriber {}

impl Transcriber {
    pub fn load(_config: &VoiceConfig) -> Result<Self, VoiceError> {
        Err(VoiceError::AudioNotSupported)
    }

    pub fn transcribe(&self, _samples: &[f32]) -> Result<String, VoiceError> {
        Err(VoiceError::AudioNotSupported)
    }
}
