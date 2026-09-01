//! Locating the Whisper model weights on disk.
//!
//! Resolution mirrors `hi_tools::checkpoint::default_state_root`: an explicit
//! override first, then XDG, then `$HOME`.
//!
//! [`download_model`] fetches the weights on first use so a new machine needs
//! no manual setup. It streams to a `.part` file and renames on success, which
//! is what keeps an interrupted download from ever being mistaken for a usable
//! model — [`resolve_model_path`] only accepts the final name.

use std::path::PathBuf;

use crate::VoiceError;

/// Transcription quality tier — which Whisper model dictation uses.
///
/// The two ship from the same whisper.cpp release and take the same code path;
/// they differ only in the decoder they carry, so switching is purely a matter
/// of which file is on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Quality {
    /// `large-v3-turbo` (~1.6 GB): large-v3's encoder with a distilled 4-layer
    /// decoder. Near-large-v3 accuracy at a fraction of the decode cost — the
    /// right default for interactive dictation.
    #[default]
    Fast,
    /// `large-v3` (~3.1 GB): the full 32-layer decoder. Meaningfully better on
    /// hard audio, proper nouns, and non-English, at a few× the decode time —
    /// still well under a second per utterance on Apple Silicon.
    Max,
}

impl Quality {
    /// The ggml model filename for this tier.
    pub fn model_file(self) -> &'static str {
        match self {
            Self::Fast => "ggml-large-v3-turbo.bin",
            Self::Max => "ggml-large-v3.bin",
        }
    }

    /// Where the model is published.
    pub fn model_url(self) -> &'static str {
        match self {
            Self::Fast => {
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
            }
            Self::Max => {
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
            }
        }
    }

    /// Published size, for a progress percentage before the server reports a
    /// content length. Approximate is fine — it only scales the bar.
    pub fn model_bytes(self) -> u64 {
        match self {
            Self::Fast => 1_624_555_275,
            Self::Max => 3_095_033_483,
        }
    }

    /// Short human label for status lines.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "fast (large-v3-turbo)",
            Self::Max => "max (large-v3)",
        }
    }

    /// Parse a tier keyword. Accepts the synonyms a user is likely to reach
    /// for; returns `None` for anything else so the caller can fall through to
    /// language handling.
    pub fn parse(word: &str) -> Option<Self> {
        match word.trim().to_ascii_lowercase().as_str() {
            "fast" | "turbo" | "default" => Some(Self::Fast),
            "max" | "hq" | "high" | "best" | "quality" | "accurate" => Some(Self::Max),
            _ => None,
        }
    }
}

/// Directory holding voice models, honouring `HI_VOICE_MODEL_DIR`, then
/// `XDG_DATA_HOME`, then `$HOME/.local/share`.
pub fn model_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("HI_VOICE_MODEL_DIR") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(path).join("hi").join("voice");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("hi")
            .join("voice");
    }
    std::env::temp_dir().join("hi-voice")
}

/// Resolve the model path for a quality tier, preferring an explicit override.
///
/// The override wins even if it does not exist, so a typo reports the path the
/// user actually asked for rather than silently falling back to the default.
pub fn resolve_model_path(
    configured: Option<&str>,
    quality: Quality,
) -> Result<PathBuf, VoiceError> {
    let path = model_path(configured, quality);
    if path.is_file() {
        return Ok(path);
    }
    Err(VoiceError::ModelMissing {
        path: path.display().to_string(),
        hint: format!(
            "it downloads automatically on first use; to use an existing ggml \
             Whisper model instead, point HI_VOICE_MODEL at it or fetch it with:\n  \
             mkdir -p {dir} && curl -L -o {path} {url}",
            dir = model_dir().display(),
            path = path.display(),
            url = quality.model_url(),
        ),
    })
}

/// Where the model should live, whether or not it exists yet.
///
/// An explicit path (config or `HI_VOICE_MODEL`) pins the file regardless of
/// tier — the user has named exactly what they want. Otherwise the tier picks
/// the filename under [`model_dir`].
pub fn model_path(configured: Option<&str>, quality: Quality) -> PathBuf {
    match configured.map(str::trim).filter(|p| !p.is_empty()) {
        Some(explicit) => PathBuf::from(explicit),
        None => match std::env::var_os("HI_VOICE_MODEL") {
            Some(from_env) => PathBuf::from(from_env),
            None => model_dir().join(quality.model_file()),
        },
    }
}

/// Download a quality tier's model to `dest`, reporting bytes through
/// `progress`.
///
/// Writes to a `.part` sibling and renames on success, so an interrupted
/// download can never be mistaken for a usable model — [`resolve_model_path`]
/// only accepts the final name. Rename within a directory is atomic.
pub async fn download_model(
    quality: Quality,
    dest: &std::path::Path,
    progress: impl Fn(u64, Option<u64>),
) -> Result<(), VoiceError> {
    download_from(quality.model_url(), dest, progress).await
}

/// [`download_model`] against an arbitrary URL, so the streaming and
/// atomic-rename behaviour can be tested without fetching 1.6 GB.
pub async fn download_from(
    url: &str,
    dest: &std::path::Path,
    progress: impl Fn(u64, Option<u64>),
) -> Result<(), VoiceError> {
    use futures_util::StreamExt;

    let parent = dest.parent().ok_or_else(|| {
        VoiceError::Download(format!("model path has no parent: {}", dest.display()))
    })?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|err| VoiceError::Download(format!("creating {}: {err}", parent.display())))?;

    let response = reqwest::get(url)
        .await
        .map_err(|err| VoiceError::Download(err.to_string()))?
        .error_for_status()
        .map_err(|err| VoiceError::Download(err.to_string()))?;
    let total = response.content_length();

    let part = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| VoiceError::Download(format!("creating {}: {err}", part.display())))?;

    let mut fetched = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| VoiceError::Download(err.to_string()))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|err| VoiceError::Download(format!("writing {}: {err}", part.display())))?;
        fetched += chunk.len() as u64;
        progress(fetched, total);
    }
    tokio::io::AsyncWriteExt::flush(&mut file)
        .await
        .map_err(|err| VoiceError::Download(err.to_string()))?;
    drop(file);

    tokio::fs::rename(&part, dest).await.map_err(|err| {
        VoiceError::Download(format!("moving {} into place: {err}", part.display()))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_existing_path_is_used_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("custom.bin");
        std::fs::write(&model, b"weights").unwrap();
        let resolved = resolve_model_path(Some(model.to_str().unwrap()), Quality::Fast).unwrap();
        assert_eq!(resolved, model);
    }

    #[test]
    fn a_missing_model_reports_the_path_and_how_to_get_it() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.bin");
        let err = resolve_model_path(Some(missing.to_str().unwrap()), Quality::Fast).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("absent.bin"),
            "names the path it looked for: {rendered}"
        );
        assert!(
            rendered.contains("curl"),
            "tells the user how to fetch it: {rendered}"
        );
    }

    #[test]
    fn blank_configured_paths_fall_through_to_the_default() {
        // A whitespace-only config value must not be treated as a real path.
        //
        // Asserted against `model_path`, which is pure: `resolve_model_path`
        // additionally checks the filesystem, so testing through it would pass
        // or fail depending on whether this machine happens to have downloaded
        // the model.
        assert_eq!(
            model_path(Some("   "), Quality::Fast),
            model_path(None, Quality::Fast)
        );
        assert_eq!(
            model_path(Some("\t\n"), Quality::Max),
            model_path(None, Quality::Max)
        );
    }

    #[test]
    fn each_quality_tier_maps_to_its_own_file() {
        if std::env::var_os("HI_VOICE_MODEL").is_none() {
            let fast = model_path(None, Quality::Fast);
            let max = model_path(None, Quality::Max);
            assert_ne!(fast, max, "tiers must not share a file");
            assert!(fast.ends_with("ggml-large-v3-turbo.bin"), "{fast:?}");
            assert!(max.ends_with("ggml-large-v3.bin"), "{max:?}");
        }
    }

    #[test]
    fn quality_parses_its_synonyms_and_rejects_language_codes() {
        assert_eq!(Quality::parse("max"), Some(Quality::Max));
        assert_eq!(Quality::parse("BEST"), Some(Quality::Max));
        assert_eq!(Quality::parse("turbo"), Some(Quality::Fast));
        assert_eq!(Quality::parse("fast"), Some(Quality::Fast));
        // Language codes must fall through so the caller can handle them.
        assert_eq!(Quality::parse("en"), None);
        assert_eq!(Quality::parse("auto"), None);
    }

    #[tokio::test]
    async fn a_failed_download_leaves_no_file_in_place() {
        // A 404 must not leave a truncated or empty file where the model goes:
        // `resolve_model_path` would then happily hand it to Whisper.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.bin");
        let err = download_from(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/definitely-not-a-real-file",
            &dest,
            |_, _| {},
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, VoiceError::Download(_)),
            "expected a download error, got {err:?}"
        );
        assert!(!dest.exists(), "no model file may be left behind");
    }

    /// Ignored by default: needs the network. Run with
    /// `cargo test -p hi-voice -- --ignored` to exercise the real transfer.
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn download_streams_to_a_part_file_then_renames_it_into_place() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("nested").join("model.bin");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = std::sync::Arc::clone(&seen);

        // A small file from the same host, so this tests the real code path
        // (redirects, chunked streaming, progress) without 1.6 GB.
        download_from(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/README.md",
            &dest,
            move |done, total| recorder.lock().unwrap().push((done, total)),
        )
        .await
        .unwrap();

        assert!(dest.is_file(), "renamed into place, creating parent dirs");
        assert!(!dest.with_extension("part").exists(), "no .part left over");
        let progress = seen.lock().unwrap();
        assert!(!progress.is_empty(), "progress was reported");
        let final_bytes = progress.last().unwrap().0;
        assert_eq!(
            final_bytes,
            std::fs::metadata(&dest).unwrap().len(),
            "final progress matches the bytes actually written"
        );
    }

    #[test]
    fn the_default_path_lives_under_the_model_dir() {
        // Only meaningful when no override is set; with one, that override is
        // the whole answer.
        if std::env::var_os("HI_VOICE_MODEL").is_none() {
            assert_eq!(
                model_path(None, Quality::Fast),
                model_dir().join(Quality::Fast.model_file())
            );
        }
    }
}
