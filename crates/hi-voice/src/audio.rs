//! Microphone capture.
//!
//! Capture runs on a dedicated OS thread that owns the `cpal` stream, because a
//! `cpal::Stream` is `!Send` on CoreAudio — it cannot be moved onto a Tokio
//! worker or held across an await. The thread is told to stop through an
//! `AtomicBool` and hands the finished audio back through its `JoinHandle`.
//!
//! Whisper wants 16 kHz mono `f32`, so this module also downmixes and resamples
//! before returning.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Sample, SampleFormat, SizedSample, StreamConfig};

use crate::VoiceError;

/// Sample rate Whisper expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Refuse to start a stream whose buffer we could never resample sensibly.
const MIN_INPUT_RATE: u32 = 8_000;

/// An in-progress recording. Dropping this stops capture and discards audio;
/// call [`Recorder::stop`] to keep it.
pub struct Recorder {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Result<Vec<f32>, VoiceError>>>,
}

impl Recorder {
    /// Open the default input device and begin capturing.
    ///
    /// Returns once the stream is actually running, so device and permission
    /// failures surface here rather than silently producing no audio. On macOS
    /// the first call triggers the microphone permission prompt.
    pub fn start() -> Result<Self, VoiceError> {
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), VoiceError>>();
        let thread_stop = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("hi-voice-capture".into())
            .spawn(move || capture(thread_stop, ready_tx))
            .map_err(|err| VoiceError::AudioCaptureFailed(err.to_string()))?;

        // The thread reports readiness (or the failure that stopped it) before
        // we hand a Recorder back.
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                stop,
                thread: Some(thread),
            }),
            Ok(Err(err)) => {
                let _ = thread.join();
                Err(err)
            }
            Err(_) => {
                let _ = thread.join();
                Err(VoiceError::AudioCaptureFailed(
                    "capture thread exited before it started".into(),
                ))
            }
        }
    }

    /// Stop capturing and return the recorded audio as 16 kHz mono `f32`.
    pub fn stop(mut self) -> Result<Vec<f32>, VoiceError> {
        self.stop.store(true, Ordering::SeqCst);
        let Some(thread) = self.thread.take() else {
            return Err(VoiceError::AudioCaptureFailed(
                "recorder already stopped".into(),
            ));
        };
        thread
            .join()
            .map_err(|_| VoiceError::AudioCaptureFailed("capture thread panicked".into()))?
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // A dropped recorder must not leave the capture thread holding the
        // microphone open.
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Body of the capture thread: own the stream, poll for the stop flag, then
/// downmix and resample what was collected.
fn capture(
    stop: Arc<AtomicBool>,
    ready: mpsc::Sender<Result<(), VoiceError>>,
) -> Result<Vec<f32>, VoiceError> {
    let collected = Arc::new(Mutex::new(Vec::<f32>::new()));

    let started = (|| -> Result<(Device, StreamConfig, SampleFormat), VoiceError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| VoiceError::AudioCaptureFailed("no input device".into()))?;
        let supported = device
            .default_input_config()
            .map_err(|err| VoiceError::AudioCaptureFailed(err.to_string()))?;
        let format = supported.sample_format();
        let config = supported.config();
        if config.sample_rate < MIN_INPUT_RATE {
            return Err(VoiceError::AudioCaptureFailed(format!(
                "input device sample rate {} Hz is too low",
                config.sample_rate
            )));
        }
        Ok((device, config, format))
    })();

    let (device, config, format) = match started {
        Ok(parts) => parts,
        Err(err) => {
            let _ = ready.send(Err(err.clone()));
            return Err(err);
        }
    };

    let channels = config.channels.max(1) as usize;
    let input_rate = config.sample_rate;

    let stream = match build_stream(&device, &config, format, Arc::clone(&collected), channels) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = ready.send(Err(err.clone()));
            return Err(err);
        }
    };
    if let Err(err) = stream.play() {
        let err = VoiceError::AudioCaptureFailed(err.to_string());
        let _ = ready.send(Err(err.clone()));
        return Err(err);
    }

    let _ = ready.send(Ok(()));

    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Drop before reading the buffer so no callback can still be appending.
    drop(stream);

    let samples = std::mem::take(&mut *collected.lock().expect("capture buffer poisoned"));
    Ok(resample_to_whisper(&samples, input_rate))
}

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    sink: Arc<Mutex<Vec<f32>>>,
    channels: usize,
) -> Result<cpal::Stream, VoiceError> {
    match format {
        SampleFormat::F32 => build_typed::<f32>(device, config, sink, channels),
        SampleFormat::I16 => build_typed::<i16>(device, config, sink, channels),
        SampleFormat::U16 => build_typed::<u16>(device, config, sink, channels),
        SampleFormat::I32 => build_typed::<i32>(device, config, sink, channels),
        other => Err(VoiceError::AudioCaptureFailed(format!(
            "unsupported sample format {other:?}"
        ))),
    }
}

fn build_typed<T>(
    device: &Device,
    config: &StreamConfig,
    sink: Arc<Mutex<Vec<f32>>>,
    channels: usize,
) -> Result<cpal::Stream, VoiceError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream::<T, _, _>(
            config.clone(),
            move |data: &[T], _| {
                let mut sink = match sink.lock() {
                    Ok(sink) => sink,
                    Err(_) => return,
                };
                // Downmix interleaved frames to mono by averaging channels.
                for frame in data.chunks(channels) {
                    let sum: f32 = frame.iter().map(|&s| f32::from_sample(s)).sum();
                    sink.push(sum / frame.len() as f32);
                }
            },
            |err| tracing::warn!("voice: audio input error: {err}"),
            None,
        )
        .map_err(|err| VoiceError::AudioCaptureFailed(err.to_string()))
}

/// Resample mono `f32` to [`WHISPER_SAMPLE_RATE`].
///
/// When the input rate is an exact multiple of the target (the common case —
/// 48 kHz and 32 kHz devices), average each block of samples. Block averaging
/// is a crude low-pass, which matters: plain decimation would alias
/// higher-frequency content down into the speech band and cost accuracy.
/// Otherwise fall back to linear interpolation.
pub fn resample_to_whisper(input: &[f32], input_rate: u32) -> Vec<f32> {
    if input.is_empty() || input_rate == WHISPER_SAMPLE_RATE {
        return input.to_vec();
    }
    if input_rate > WHISPER_SAMPLE_RATE && input_rate % WHISPER_SAMPLE_RATE == 0 {
        let factor = (input_rate / WHISPER_SAMPLE_RATE) as usize;
        return input
            .chunks(factor)
            .map(|block| block.iter().sum::<f32>() / block.len() as f32)
            .collect();
    }
    let ratio = f64::from(input_rate) / f64::from(WHISPER_SAMPLE_RATE);
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = (pos - pos.floor()) as f32;
            let a = input[idx.min(input.len() - 1)];
            let b = *input.get(idx + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_is_identity_at_the_target_rate() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_to_whisper(&input, WHISPER_SAMPLE_RATE), input);
    }

    #[test]
    fn integer_ratio_downsampling_averages_each_block() {
        // 48 kHz -> 16 kHz is 3:1, the overwhelmingly common device case.
        let input = vec![0.0, 3.0, 6.0, 1.0, 1.0, 1.0];
        let out = resample_to_whisper(&input, 48_000);
        assert_eq!(out, vec![3.0, 1.0], "each block of 3 averaged");
    }

    #[test]
    fn non_integer_ratio_interpolates_and_shortens() {
        let input: Vec<f32> = (0..441).map(|i| i as f32).collect();
        let out = resample_to_whisper(&input, 44_100);
        assert_eq!(out.len(), 160, "44.1 kHz -> 16 kHz is a 2.75625:1 ratio");
        assert!(out[0] == 0.0, "first sample is preserved: {:?}", &out[..3]);
        assert!(
            out.windows(2).all(|w| w[1] > w[0]),
            "a monotonic ramp stays monotonic through interpolation"
        );
    }

    #[test]
    fn empty_input_stays_empty() {
        assert!(resample_to_whisper(&[], 48_000).is_empty());
    }
}
