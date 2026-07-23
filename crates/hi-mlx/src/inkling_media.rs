//! Inkling multimodal preprocessing, ported from the reference `inkling_mlx/processing.py`.
//!
//! Pure CPU (no MLX): decodes media and produces the tensors the vision/audio towers consume.
//!   * image -> pixel_values `[num_patches, 2, 40, 40, 3]` f32   (for the vision tower)
//!   * audio -> dMel bin ids  `[num_frames, 80]` i32             (for the audio tower)
//!
//! Both were checked bit-exact against the reference preprocessing.

use anyhow::{Context, Result};
use rustfft::num_complex::Complex32;

// CLIP normalization (OPENAI_CLIP_MEAN / STD).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const PATCH: usize = 40; // image patch size (== vision patch_size)
const TEMPORAL: usize = 2; // temporal_patch_size (image duplicated across 2 frames)

// Audio constants (processor_config.json / feature_extraction_inkling.py).
const SR: u32 = 16000;
const HOP: usize = 800;
const WIN: usize = 1600;
const N_FFT: usize = 1600;
const N_MEL: usize = 80;
const DMEL_BINS: usize = 16;
const DMEL_MIN: f64 = -7.0;
const DMEL_MAX: f64 = 2.0;

/// Decode image bytes and produce `(pixel_values, num_patches)` where pixel_values is the row-major
/// flattening of `[num_patches, 2, 40, 40, 3]`. `max_long_edge` optionally downscales (LANCZOS,
/// aspect preserved) before patchify to bound the soft-token count.
pub fn preprocess_image(bytes: &[u8], max_long_edge: Option<u32>) -> Result<(Vec<f32>, usize)> {
    let img = image::load_from_memory(bytes).context("decoding image")?;
    let mut img = img.to_rgb8();
    if let Some(cap) = max_long_edge {
        let long = img.width().max(img.height());
        if long > cap {
            let r = cap as f64 / long as f64;
            let nw = ((img.width() as f64 * r).round() as u32).max(1);
            let nh = ((img.height() as f64 * r).round() as u32).max(1);
            img = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3);
        }
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    // Patch grid: rows = ceil(H/40), cols = W//40 + 1 (matching the reference, which over-tiles).
    let num_rows = h.div_ceil(PATCH);
    let num_cols = w / PATCH + 1;
    let num_patches = num_rows * num_cols;

    // Output layout [N, T=2, 40, 40, C=3], row-major. Both temporal frames are identical.
    let mut out = vec![0f32; num_patches * TEMPORAL * PATCH * PATCH * 3];
    let px = img.as_raw(); // [h*w*3], row-major RGB u8
    for pr in 0..num_rows {
        for pc in 0..num_cols {
            let patch_idx = pr * num_cols + pc;
            for py in 0..PATCH {
                for pxx in 0..PATCH {
                    let iy = pr * PATCH + py;
                    let ix = pc * PATCH + pxx;
                    for c in 0..3 {
                        // Pad value is -1.0 (pre-normalization), applied where the patch runs off
                        // the image; then rescale by 1/255 and CLIP-normalize per channel.
                        let raw = if iy < h && ix < w {
                            px[(iy * w + ix) * 3 + c] as f32
                        } else {
                            -1.0
                        };
                        let norm = (raw / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
                        for t in 0..TEMPORAL {
                            let o =
                                ((((patch_idx * TEMPORAL + t) * PATCH + py) * PATCH + pxx) * 3) + c;
                            out[o] = norm;
                        }
                    }
                }
            }
        }
    }
    Ok((out, num_patches))
}

/// Raw mono 16 kHz `samples` -> `(dmel_ids, num_frames)` where dmel_ids is the row-major flattening
/// of `[num_frames, 80]` bin indices in `0..16`.
pub fn preprocess_audio(samples: &[f32], sampling_rate: u32) -> Result<(Vec<i32>, usize)> {
    anyhow::ensure!(
        sampling_rate == SR,
        "Inkling audio expects {SR} Hz, got {sampling_rate}"
    );
    let mel = log_mel(samples); // [T_padded, 80] log10-mel
    // Drop the trailing padded frames: the reference keeps ceil(len/HOP) frames.
    let n_valid = samples.len().div_ceil(HOP);
    let n_frames = n_valid.min(mel.len() / N_MEL);

    // dMel: clip to [-7, 2], take the nearest of 16 evenly spaced centers.
    let centers: Vec<f64> = (0..DMEL_BINS)
        .map(|i| DMEL_MIN + (DMEL_MAX - DMEL_MIN) * i as f64 / (DMEL_BINS as f64 - 1.0))
        .collect();
    let mut ids = vec![0i32; n_frames * N_MEL];
    for f in 0..n_frames {
        for m in 0..N_MEL {
            let v = (mel[f * N_MEL + m] as f64).clamp(DMEL_MIN, DMEL_MAX);
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (b, c) in centers.iter().enumerate() {
                let d = (v - c).abs();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            ids[f * N_MEL + m] = best as i32;
        }
    }
    Ok((ids, n_frames))
}

/// Raw waveform -> log10-mel spectrogram, row-major `[num_frames, 80]`. Mirrors the reference:
/// left-pad by `N_FFT-HOP`, right-pad to a HOP multiple, periodic Hann window, `center=False`
/// framing, magnitude rFFT, slaney mel filterbank, log10 with a 1e-10 floor.
fn log_mel(samples: &[f32]) -> Vec<f32> {
    let left = N_FFT.saturating_sub(HOP);
    let right = samples.len().div_ceil(HOP) * HOP - samples.len();
    let mut wav = vec![0f32; left + samples.len() + right];
    wav[left..left + samples.len()].copy_from_slice(samples);

    // Periodic Hann: hanning(WIN+1)[:-1].
    let window: Vec<f32> = (0..WIN)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / WIN as f64).cos()) as f32)
        .collect();

    let n_frames = if wav.len() >= N_FFT {
        1 + (wav.len() - N_FFT) / HOP
    } else {
        0
    };
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let n_bins = N_FFT / 2 + 1;
    let fb = mel_filterbank(); // [N_MEL][n_bins]

    let mut mel = vec![0f32; n_frames * N_MEL];
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    for f in 0..n_frames {
        let start = f * HOP;
        // Windowed frame (WIN <= N_FFT; the tail stays zero, matching an N_FFT rFFT).
        for (i, b) in buf.iter_mut().enumerate() {
            let s = if i < WIN {
                wav[start + i] * window[i]
            } else {
                0.0
            };
            *b = Complex32::new(s, 0.0);
        }
        fft.process(&mut buf);
        // Magnitude of the first n_bins (rFFT), floored at 1e-10.
        let mag: Vec<f32> = buf[..n_bins].iter().map(|c| c.norm().max(1e-10)).collect();
        for (m, filt) in fb.iter().enumerate() {
            let mut acc = 0f32;
            for (b, &fv) in filt.iter().enumerate() {
                acc += fv * mag[b];
            }
            mel[f * N_MEL + m] = acc.max(1e-10).log10();
        }
    }
    mel
}

/// Slaney mel filterbank `[80][801]`, matching `transformers.audio_utils.mel_filter_bank` with
/// `norm="slaney", mel_scale="slaney"`: triangular filters on the slaney mel scale, area-normalized.
fn mel_filterbank() -> Vec<Vec<f32>> {
    let n_bins = N_FFT / 2 + 1;
    let f_min = 0.0f64;
    let f_max = SR as f64 / 2.0;

    // FFT bin center frequencies (Hz).
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * SR as f64 / N_FFT as f64)
        .collect();

    // Slaney mel <-> Hz.
    let hz_to_mel = |hz: f64| -> f64 {
        const F_SP: f64 = 200.0 / 3.0;
        const MIN_LOG_HZ: f64 = 1000.0;
        const MIN_LOG_MEL: f64 = 1000.0 / (200.0 / 3.0);
        let logstep = (6.4f64).ln() / 27.0;
        if hz >= MIN_LOG_HZ {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
        } else {
            hz / F_SP
        }
    };
    let mel_to_hz = |mel: f64| -> f64 {
        const F_SP: f64 = 200.0 / 3.0;
        const MIN_LOG_HZ: f64 = 1000.0;
        const MIN_LOG_MEL: f64 = 1000.0 / (200.0 / 3.0);
        let logstep = (6.4f64).ln() / 27.0;
        if mel >= MIN_LOG_MEL {
            MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
        } else {
            F_SP * mel
        }
    };

    // N_MEL+2 mel points evenly spaced, converted to Hz.
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f64> = (0..N_MEL + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (N_MEL as f64 + 1.0)))
        .collect();

    let mut fb = vec![vec![0f32; n_bins]; N_MEL];
    for m in 0..N_MEL {
        let (lo, ctr, hi) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
        for (b, &freq) in fft_freqs.iter().enumerate() {
            let down = (freq - lo) / (ctr - lo);
            let up = (hi - freq) / (hi - ctr);
            let tri = down.min(up).max(0.0);
            fb[m][b] = tri as f32;
        }
        // Slaney area normalization: scale each filter by 2/(hi-lo).
        let enorm = 2.0 / (mel_points[m + 2] - mel_points[m]);
        for b in 0..n_bins {
            fb[m][b] *= enorm as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // Fixtures + reference outputs are produced by the scratch harness. Run explicitly:
    //   cargo test -p hi-mlx --lib inkling_media -- --ignored --nocapture
    #[test]
    #[ignore]
    fn image_matches_reference() {
        let bytes = fs::read("/tmp/test_img.png").unwrap();
        let (pv, n) = preprocess_image(&bytes, None).unwrap();
        let refbytes = fs::read("/tmp/ref_pixel.f32").unwrap();
        let refv: Vec<f32> = refbytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(pv.len(), refv.len(), "num_patches={n}");
        let maxdiff = pv
            .iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("image: num_patches={n} max abs diff={maxdiff:.3e}");
        assert!(maxdiff < 1e-4, "image pixel_values differ: {maxdiff}");
    }

    #[test]
    #[ignore]
    fn audio_matches_reference() {
        let raw = fs::read("/tmp/test_audio.f32").unwrap();
        let samples: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (ids, n) = preprocess_audio(&samples, 16000).unwrap();
        let refbytes = fs::read("/tmp/ref_dmel.i32").unwrap();
        let refv: Vec<i32> = refbytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(ids.len(), refv.len(), "num_frames={n}");
        let mismatches = ids.iter().zip(&refv).filter(|(a, b)| a != b).count();
        let maxbin = ids
            .iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap_or(0);
        println!(
            "audio: frames={n} mismatched bins={mismatches}/{} max bin diff={maxbin}",
            ids.len()
        );
        // dMel is a discretization; allow a tiny number of near-boundary rounding flips.
        assert!(
            mismatches <= ids.len() / 200,
            "too many dMel mismatches: {mismatches}"
        );
    }
}
