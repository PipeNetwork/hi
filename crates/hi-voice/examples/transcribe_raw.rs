//! Transcribe a raw 16 kHz mono little-endian `f32` PCM file.
//!
//! Exercises the real model end to end without a microphone, which is how the
//! speech path can be checked in CI or by hand:
//!
//! ```sh
//! say -o /tmp/clip.raw --data-format=LEF32@16000 "commit and push the branch"
//! cargo run -p hi-voice --example transcribe_raw -- /tmp/clip.raw
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: transcribe_raw <raw-f32-16k-mono-file>")?;

    let bytes = std::fs::read(&path)?;
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    eprintln!(
        "{} samples ({:.2}s at {} Hz)",
        samples.len(),
        samples.len() as f32 / hi_voice::WHISPER_SAMPLE_RATE as f32,
        hi_voice::WHISPER_SAMPLE_RATE,
    );

    let started = std::time::Instant::now();
    let transcriber = hi_voice::Transcriber::load(&hi_voice::VoiceConfig::default())?;
    eprintln!("model loaded in {:.2}s", started.elapsed().as_secs_f32());

    let started = std::time::Instant::now();
    let text = transcriber.transcribe(&samples)?;
    eprintln!("transcribed in {:.2}s", started.elapsed().as_secs_f32());

    println!("{text}");
    Ok(())
}
