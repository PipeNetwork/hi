//! Voice dictation state machine.
//!
//! Ctrl+Space toggles between recording and transcribing. Capture itself is
//! cheap, but transcription is seconds of CPU/GPU work, so it runs on a
//! blocking thread and its result is collected by [`crate::App::drain_voice`]
//! on the UI tick — the same shape `drain_loops` uses.
//!
//! The model is loaded lazily on first use and cached: it costs seconds and
//! ~1.6 GB, which nobody should pay for a session where they never dictate.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use hi_voice::{Recorder, Transcriber, VoiceConfig, VoiceError};
use ratatui::text::Line;
use tokio::sync::oneshot;

use crate::render::dim;

/// Where dictation currently is.
#[derive(Default)]
pub(crate) enum VoiceState {
    /// Not dictating.
    #[default]
    Idle,
    /// Fetching the model on first use.
    Downloading {
        rx: oneshot::Receiver<Result<(), VoiceError>>,
        /// Bytes fetched so far, written by the download task.
        fetched: Arc<AtomicU64>,
        /// Total size once the server reports one.
        total: Arc<AtomicU64>,
        /// Suppresses repeating the same percentage every tick.
        last_percent: u8,
    },
    /// Capturing from the microphone.
    Recording(Recorder),
    /// Capture finished; Whisper is running on a blocking thread.
    Transcribing {
        rx: oneshot::Receiver<Result<String, VoiceError>>,
    },
}

impl VoiceState {
    /// Whether the microphone is currently open (drives the UI indicator).
    pub(crate) fn is_recording(&self) -> bool {
        matches!(self, Self::Recording(_))
    }

    /// Whether Whisper is still working.
    pub(crate) fn is_transcribing(&self) -> bool {
        matches!(self, Self::Transcribing { .. })
    }

    /// Whether the model is still being fetched.
    pub(crate) fn is_downloading(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }

    /// Download progress as whole percent, for the status indicator.
    ///
    /// Falls back to the known model size until the server reports a content
    /// length, so the bar is never stuck at an unknown value.
    pub(crate) fn download_percent(&self) -> Option<u8> {
        let Self::Downloading { fetched, total, .. } = self else {
            return None;
        };
        let size = match total.load(Ordering::Relaxed) {
            0 => hi_voice::DEFAULT_MODEL_BYTES,
            reported => reported,
        };
        let done = fetched.load(Ordering::Relaxed);
        Some(((done.min(size) * 100) / size.max(1)) as u8)
    }
}

/// Cache for the loaded Whisper model, shared with the blocking task.
pub(crate) type VoiceModelCache = Arc<Mutex<Option<Arc<Transcriber>>>>;

impl crate::App {
    /// Ctrl+Space: start recording, or stop and transcribe.
    pub(crate) fn toggle_voice(&mut self) {
        if !hi_voice::AUDIO_SUPPORTED {
            self.push(Line::styled(
                "voice: audio capture is not supported on this platform".to_string(),
                dim(),
            ));
            return;
        }
        match std::mem::take(&mut self.voice) {
            VoiceState::Idle => {
                // First use on a fresh machine: fetch the model instead of
                // telling the user to go and run curl.
                if hi_voice::model_path(self.voice_config.model_path.as_deref()).is_file() {
                    self.start_voice_recording();
                } else {
                    self.start_voice_model_download();
                }
            }
            VoiceState::Recording(recorder) => self.finish_voice_recording(recorder),
            // Toggling while Whisper is still decoding must not drop the
            // pending result on the floor, so put the state back untouched.
            state @ VoiceState::Transcribing { .. } => {
                self.voice = state;
                self.push(Line::styled(
                    "voice: still transcribing the last recording…".to_string(),
                    dim(),
                ));
            }
            state @ VoiceState::Downloading { .. } => {
                self.voice = state;
                self.push(Line::styled(
                    "voice: still downloading the model…".to_string(),
                    dim(),
                ));
            }
        }
    }

    /// Fetch the Whisper model in the background, reporting progress on the
    /// UI tick. Recording starts on the next Ctrl+Space rather than
    /// automatically — a download runs for minutes, and nobody wants the
    /// microphone to silently open when it happens to finish.
    fn start_voice_model_download(&mut self) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.voice = VoiceState::Idle;
            self.push(Line::styled(
                "voice: no async runtime available to download on".to_string(),
                dim(),
            ));
            return;
        };
        let dest = hi_voice::model_path(self.voice_config.model_path.as_deref());
        let fetched = Arc::new(AtomicU64::new(0));
        let total = Arc::new(AtomicU64::new(0));
        let (tx, rx) = oneshot::channel();
        let task_fetched = Arc::clone(&fetched);
        let task_total = Arc::clone(&total);
        handle.spawn(async move {
            let result = hi_voice::download_model(&dest, |done, size| {
                task_fetched.store(done, Ordering::Relaxed);
                if let Some(size) = size {
                    task_total.store(size, Ordering::Relaxed);
                }
            })
            .await;
            let _ = tx.send(result);
        });
        self.voice = VoiceState::Downloading {
            rx,
            fetched,
            total,
            last_percent: u8::MAX,
        };
        self.push(Line::styled(
            format!(
                "voice: first use — downloading the Whisper model (~{:.1} GB), this happens once…",
                hi_voice::DEFAULT_MODEL_BYTES as f64 / 1e9
            ),
            dim(),
        ));
    }

    fn start_voice_recording(&mut self) {
        match Recorder::start() {
            Ok(recorder) => {
                self.voice = VoiceState::Recording(recorder);
                self.push(Line::styled(
                    "voice: recording — Ctrl+Space to stop".to_string(),
                    dim(),
                ));
            }
            Err(err) => {
                self.voice = VoiceState::Idle;
                self.push(Line::styled(format!("voice: {err}"), dim()));
            }
        }
    }

    fn finish_voice_recording(&mut self, recorder: Recorder) {
        let samples = match recorder.stop() {
            Ok(samples) => samples,
            Err(err) => {
                self.voice = VoiceState::Idle;
                self.push(Line::styled(format!("voice: {err}"), dim()));
                return;
            }
        };
        if samples.is_empty() {
            self.voice = VoiceState::Idle;
            self.push(Line::styled(
                "voice: nothing was recorded".to_string(),
                dim(),
            ));
            return;
        }

        // Whisper is synchronous and slow; it must not run on the UI thread.
        // Outside a Tokio runtime (unit tests) there is nowhere to offload it,
        // so report that rather than blocking or panicking in spawn_blocking.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.voice = VoiceState::Idle;
            self.push(Line::styled(
                "voice: no async runtime available to transcribe on".to_string(),
                dim(),
            ));
            return;
        };

        let seconds = samples.len() as f32 / hi_voice::WHISPER_SAMPLE_RATE as f32;
        let cache = Arc::clone(&self.voice_model);
        let config = self.voice_config.clone();
        let (tx, rx) = oneshot::channel();
        handle.spawn_blocking(move || {
            let _ = tx.send(transcribe_with_cache(&cache, &config, &samples));
        });

        self.voice = VoiceState::Transcribing { rx };
        self.push(Line::styled(
            format!("voice: transcribing {seconds:.1}s…"),
            dim(),
        ));
    }

    /// Collect a finished transcription, if one is ready. Called from the UI
    /// tick; never blocks.
    pub(crate) fn drain_voice(&mut self) {
        if self.voice.is_downloading() {
            self.drain_voice_download();
            return;
        }
        let VoiceState::Transcribing { rx } = &mut self.voice else {
            return;
        };
        let outcome = match rx.try_recv() {
            Ok(outcome) => outcome,
            // Still decoding.
            Err(oneshot::error::TryRecvError::Empty) => return,
            Err(oneshot::error::TryRecvError::Closed) => {
                self.voice = VoiceState::Idle;
                self.push(Line::styled(
                    "voice: transcription task died".to_string(),
                    dim(),
                ));
                return;
            }
        };
        self.voice = VoiceState::Idle;
        match outcome {
            Ok(text) if text.trim().is_empty() => {
                self.push(Line::styled("voice: no speech detected".to_string(), dim()));
            }
            Ok(text) => self.insert_dictation(&text),
            Err(err) => self.push(Line::styled(format!("voice: {err}"), dim())),
        }
    }

    /// Report download progress, and pick up the finished download.
    fn drain_voice_download(&mut self) {
        let VoiceState::Downloading {
            rx,
            fetched,
            total,
            last_percent,
        } = &mut self.voice
        else {
            return;
        };

        // Report in whole percent so a 1.6 GB fetch does not spam the
        // transcript on every 120 ms tick.
        let done = fetched.load(Ordering::Relaxed);
        let size = match total.load(Ordering::Relaxed) {
            0 => hi_voice::DEFAULT_MODEL_BYTES,
            reported => reported,
        };
        let percent = ((done.min(size) * 100) / size.max(1)) as u8;
        let should_report = percent != *last_percent && percent.is_multiple_of(10);

        match rx.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => {
                if should_report {
                    *last_percent = percent;
                    self.push(Line::styled(
                        format!("voice: downloading the model… {percent}%"),
                        dim(),
                    ));
                }
            }
            Err(oneshot::error::TryRecvError::Closed) => {
                self.voice = VoiceState::Idle;
                self.push(Line::styled(
                    "voice: the model download task died".to_string(),
                    dim(),
                ));
            }
            Ok(Ok(())) => {
                self.voice = VoiceState::Idle;
                self.push(Line::styled(
                    "voice: model ready — press Ctrl+Space to dictate".to_string(),
                    dim(),
                ));
            }
            Ok(Err(err)) => {
                self.voice = VoiceState::Idle;
                self.push(Line::styled(format!("voice: {err}"), dim()));
            }
        }
    }

    /// Insert dictated text at the cursor, keeping word spacing sane when the
    /// prompt already has content.
    fn insert_dictation(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let needs_space = self
            .input
            .text()
            .chars()
            .last()
            .is_some_and(|c| !c.is_whitespace());
        if needs_space {
            self.input.insert(' ');
        }
        self.input.insert_str(text);
    }
}

/// Load the model on first use, then transcribe. Runs on a blocking thread.
fn transcribe_with_cache(
    cache: &VoiceModelCache,
    config: &VoiceConfig,
    samples: &[f32],
) -> Result<String, VoiceError> {
    let transcriber = {
        let mut guard = cache
            .lock()
            .map_err(|_| VoiceError::Transcribe("voice model cache poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(Arc::new(Transcriber::load(config)?));
        }
        Arc::clone(guard.as_ref().expect("model just loaded"))
    };
    transcriber.transcribe(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_app;

    #[test]
    fn dictation_appends_with_a_separating_space() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.insert_str("commit");
        app.insert_dictation("and push");
        assert_eq!(app.input.text(), "commit and push");
    }

    #[test]
    fn dictation_into_an_empty_prompt_adds_no_leading_space() {
        let mut app = test_app("openai", "gpt-4o");
        app.insert_dictation("run the tests");
        assert_eq!(app.input.text(), "run the tests");
    }

    #[test]
    fn dictation_does_not_double_space_after_existing_whitespace() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.insert_str("fix ");
        app.insert_dictation("the flaky test");
        assert_eq!(app.input.text(), "fix the flaky test");
    }

    #[test]
    fn blank_dictation_leaves_the_prompt_untouched() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.insert_str("keep");
        app.insert_dictation("   ");
        assert_eq!(app.input.text(), "keep");
    }

    #[test]
    fn draining_while_idle_is_a_no_op() {
        let mut app = test_app("openai", "gpt-4o");
        app.drain_voice();
        assert!(!app.voice.is_recording() && !app.voice.is_transcribing());
    }

    #[test]
    fn a_dropped_transcription_task_reports_instead_of_hanging() {
        let mut app = test_app("openai", "gpt-4o");
        let (tx, rx) = oneshot::channel::<Result<String, VoiceError>>();
        drop(tx);
        app.voice = VoiceState::Transcribing { rx };
        app.drain_voice();
        assert!(
            !app.voice.is_transcribing(),
            "a closed channel must return to Idle rather than wait forever"
        );
    }

    #[test]
    fn a_pending_transcription_stays_pending() {
        let mut app = test_app("openai", "gpt-4o");
        let (tx, rx) = oneshot::channel::<Result<String, VoiceError>>();
        app.voice = VoiceState::Transcribing { rx };
        app.drain_voice();
        assert!(app.voice.is_transcribing(), "still waiting on Whisper");
        drop(tx);
    }

    /// Build a Downloading state with the given byte counters.
    fn downloading(done: u64, size: u64) -> (oneshot::Sender<Result<(), VoiceError>>, VoiceState) {
        let (tx, rx) = oneshot::channel();
        (
            tx,
            VoiceState::Downloading {
                rx,
                fetched: Arc::new(AtomicU64::new(done)),
                total: Arc::new(AtomicU64::new(size)),
                last_percent: u8::MAX,
            },
        )
    }

    #[test]
    fn an_in_flight_download_stays_pending_and_reports_progress_once() {
        let mut app = test_app("openai", "gpt-4o");
        let (tx, state) = downloading(50, 100);
        app.voice = state;

        app.drain_voice();
        assert!(app.voice.is_downloading(), "still downloading");
        let at_50 = app
            .transcript
            .iter()
            .filter(|e| e.text().contains("50%"))
            .count();
        assert_eq!(at_50, 1, "progress is reported once at each decile");

        // A second tick at the same percentage must not repeat the line.
        app.drain_voice();
        let at_50 = app
            .transcript
            .iter()
            .filter(|e| e.text().contains("50%"))
            .count();
        assert_eq!(at_50, 1, "unchanged progress does not spam the transcript");
        drop(tx);
    }

    #[test]
    fn a_finished_download_returns_to_idle_and_says_it_is_ready() {
        let mut app = test_app("openai", "gpt-4o");
        let (tx, state) = downloading(100, 100);
        app.voice = state;
        tx.send(Ok(())).unwrap();

        app.drain_voice();
        assert!(!app.voice.is_downloading());
        assert!(
            app.transcript
                .iter()
                .any(|e| e.text().contains("model ready")),
            "tells the user dictation is now available"
        );
    }

    #[test]
    fn a_failed_download_reports_the_error_and_does_not_wedge() {
        let mut app = test_app("openai", "gpt-4o");
        let (tx, state) = downloading(10, 100);
        app.voice = state;
        tx.send(Err(VoiceError::Download("connection reset".into())))
            .unwrap();

        app.drain_voice();
        assert!(
            !app.voice.is_downloading(),
            "a failed download must return to Idle so Ctrl+Space can retry"
        );
        assert!(
            app.transcript
                .iter()
                .any(|e| e.text().contains("connection reset")),
            "surfaces why it failed"
        );
    }

    #[test]
    fn the_indicator_is_absent_when_idle_and_present_while_working() {
        let mut app = test_app("openai", "gpt-4o");
        assert!(
            app.voice_indicator().is_none(),
            "an idle session shows no voice status"
        );

        let (tx, rx) = oneshot::channel::<Result<String, VoiceError>>();
        app.voice = VoiceState::Transcribing { rx };
        let line = app.voice_indicator().expect("transcribing shows a status");
        assert!(
            line_text(&line).contains("transcribing"),
            "got: {}",
            line_text(&line)
        );
        drop(tx);

        let (tx, state) = downloading(25, 100);
        app.voice = state;
        let line = app.voice_indicator().expect("downloading shows a status");
        assert!(
            line_text(&line).contains("25%"),
            "download percent is shown: {}",
            line_text(&line)
        );
        drop(tx);
    }

    #[test]
    fn the_recording_dot_pulses_across_redraws() {
        // The dot must actually change colour as the spinner advances,
        // otherwise "live" is a claim the UI is not making. Asserted against
        // the pure colour function: building a Recording state needs a real
        // Recorder, and therefore a microphone, which no test should require.
        let cycle: Vec<_> = (0..20)
            .map(crate::App::recording_dot_color_at)
            .map(|c| format!("{c:?}"))
            .collect();
        let distinct: std::collections::HashSet<_> = cycle.iter().collect();
        assert!(
            distinct.len() > 4,
            "the dot should breathe through several shades, saw {}: {cycle:?}",
            distinct.len()
        );
        assert_eq!(
            crate::App::recording_dot_color_at(0),
            crate::App::recording_dot_color_at(20),
            "the pulse repeats on a 20-tick cycle"
        );
        assert_ne!(
            crate::App::recording_dot_color_at(0),
            crate::App::recording_dot_color_at(10),
            "trough and crest differ"
        );
    }

    #[test]
    fn download_percent_is_none_unless_downloading() {
        let mut app = test_app("openai", "gpt-4o");
        assert_eq!(app.voice.download_percent(), None);
        let (tx, state) = downloading(1, 4);
        app.voice = state;
        assert_eq!(app.voice.download_percent(), Some(25));
        drop(tx);
    }

    /// Flatten a rendered line back to text for assertions.
    fn line_text(line: &ratatui::text::Line<'static>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[tokio::test]
    async fn a_finished_transcription_lands_in_the_prompt() {
        let mut app = test_app("openai", "gpt-4o");
        let (tx, rx) = oneshot::channel();
        tx.send(Ok("push the branch".to_string())).unwrap();
        app.voice = VoiceState::Transcribing { rx };
        app.drain_voice();
        assert_eq!(app.input.text(), "push the branch");
        assert!(!app.voice.is_transcribing());
    }
}
