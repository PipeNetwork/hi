//! Whisper transcription.
//!
//! Loading the model costs seconds and ~1.6 GB of RAM, so the context is built
//! once and reused across recordings. Transcription is synchronous and CPU/GPU
//! bound — callers on an async runtime must run it on a blocking thread.

use std::sync::Arc;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{VoiceConfig, VoiceError, model};

/// Whisper needs at least this much audio to produce anything useful; below it
/// the model reliably emits hallucinated filler for silence.
const MIN_SAMPLES: usize = crate::audio::WHISPER_SAMPLE_RATE as usize / 4;

/// A loaded Whisper model, reusable across recordings.
#[derive(Clone)]
pub struct Transcriber {
    context: Arc<WhisperContext>,
    language: String,
}

impl Transcriber {
    /// Load the model named by `config`. Expensive: seconds of I/O and GPU
    /// setup, so hold onto the result.
    pub fn load(config: &VoiceConfig) -> Result<Self, VoiceError> {
        // whisper.cpp and GGML print device/buffer diagnostics straight to
        // stdout, which would scribble over the TUI. Route them into `tracing`
        // instead. Idempotent, but Once keeps it off the hot path.
        static LOGGING: std::sync::Once = std::sync::Once::new();
        LOGGING.call_once(whisper_rs::install_logging_hooks);

        let path = model::resolve_model_path(config.model_path.as_deref())?;
        let path = path
            .to_str()
            .ok_or_else(|| VoiceError::ModelMissing {
                path: path.display().to_string(),
                hint: "model path is not valid UTF-8".into(),
            })?
            .to_string();

        let context = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            .map_err(|err| VoiceError::Transcribe(format!("loading {path}: {err}")))?;
        Ok(Self {
            context: Arc::new(context),
            language: config.language.clone(),
        })
    }

    /// Transcribe 16 kHz mono `f32` audio.
    ///
    /// Returns an empty string when the clip is too short to be speech, rather
    /// than letting Whisper hallucinate text out of a fraction of a second of
    /// room tone.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String, VoiceError> {
        if samples.len() < MIN_SAMPLES {
            return Ok(String::new());
        }

        let mut state = self
            .context
            .create_state()
            .map_err(|err| VoiceError::Transcribe(err.to_string()))?;

        // Beam search costs more CPU than greedy but is materially more
        // accurate, which is the point of running a large model locally.
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: 0.0,
        });
        if self.language != crate::STT_LANGUAGE_AUTO {
            params.set_language(Some(&self.language));
        } else {
            params.set_detect_language(true);
        }
        params.set_n_threads(transcribe_threads());
        params.set_translate(false);
        // This output goes into a TUI prompt, not a terminal: nothing may be
        // printed, and timestamps/special tokens are not wanted in the text.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, samples)
            .map_err(|err| VoiceError::Transcribe(err.to_string()))?;

        let mut text = String::new();
        for segment in state.as_iter() {
            let piece = segment
                .to_str_lossy()
                .map_err(|err| VoiceError::Transcribe(err.to_string()))?;
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(piece);
        }
        Ok(clean_transcript(&text))
    }
}

/// Leave headroom so a long transcription cannot starve the UI thread.
fn transcribe_threads() -> std::os::raw::c_int {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    cores.saturating_sub(2).clamp(1, 8) as std::os::raw::c_int
}

/// Strip the bracketed non-speech annotations Whisper emits for silence and
/// background noise (`[BLANK_AUDIO]`, `(wind blowing)`, and friends). They are
/// never something the user said, so they must not land in the prompt.
fn clean_transcript(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth_square = 0usize;
    let mut depth_round = 0usize;
    for ch in text.chars() {
        match ch {
            '[' => depth_square += 1,
            ']' => depth_square = depth_square.saturating_sub(1),
            '(' => depth_round += 1,
            ')' => depth_round = depth_round.saturating_sub(1),
            _ if depth_square == 0 && depth_round == 0 => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracketed_non_speech_annotations_are_stripped() {
        assert_eq!(clean_transcript("[BLANK_AUDIO]"), "");
        assert_eq!(clean_transcript("hello [noise] world"), "hello world");
        assert_eq!(
            clean_transcript("(wind blowing) commit this"),
            "commit this"
        );
    }

    #[test]
    fn ordinary_speech_survives_cleaning() {
        assert_eq!(
            clean_transcript("  run the   tests and  push  "),
            "run the tests and push"
        );
    }

    #[test]
    fn transcribe_threads_leaves_headroom_and_stays_in_range() {
        let n = transcribe_threads();
        assert!((1..=8).contains(&n), "threads out of range: {n}");
    }
}
