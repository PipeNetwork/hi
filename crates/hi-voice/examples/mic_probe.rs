//! Record briefly from the default input device and report what arrived.
//!
//! Verifies the capture half without a TUI — device open, OS microphone
//! permission, resampling — by printing sample count and RMS level:
//!
//! ```sh
//! cargo run -p hi-voice --example mic_probe -- 2
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seconds: f32 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "2".into())
        .parse()?;

    eprintln!("opening the default input device…");
    let recorder = hi_voice::Recorder::start()?;
    eprintln!("recording for {seconds}s — say something");
    std::thread::sleep(std::time::Duration::from_secs_f32(seconds));
    let samples = recorder.stop()?;

    let rms = if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    };
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    println!(
        "samples={} ({:.2}s at {} Hz) rms={rms:.5} peak={peak:.5}",
        samples.len(),
        samples.len() as f32 / hi_voice::WHISPER_SAMPLE_RATE as f32,
        hi_voice::WHISPER_SAMPLE_RATE,
    );
    if peak == 0.0 {
        eprintln!(
            "warning: every sample is silent — the microphone permission may be denied, \
             or the wrong input device is selected"
        );
    }
    Ok(())
}
