//! Knobs for `/review`, set by the frontend from `[review]` in the config
//! file (or left at their defaults).

use crate::review::DEFAULT_REVIEW_PASSES;

/// Files one audit turn is asked to read. A recent-work scope or an `all`
/// chunk with more files is split into turns of at most this many; smaller
/// `all` chunks are packed together up to it. A live audit of 62
/// uncommitted files in one turn read 11 of them and reported "none".
pub const DEFAULT_CHUNK_FILES: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewSettings {
    /// See [`DEFAULT_CHUNK_FILES`]. Clamped to at least 1.
    pub chunk_files: usize,
    /// Directories an audit without a spec skips (recent work, chunks, and
    /// named directories alike): a workspace-relative path
    /// (`bench/fixtures`) or a bare name matched against every path
    /// component (`node_modules`). Ignored files are skipped regardless.
    pub exclude: Vec<String>,
    /// Fix passes when `/review` (or `--spec-review`) does not say.
    pub passes: u32,
    /// Command run read-only before each audit and re-audit turn; its
    /// failures are handed to the model as candidate defects. `None` falls
    /// back to the session's verify command, then to `cargo check` when the
    /// workspace root holds a `Cargo.toml`.
    pub seed: Option<String>,
}

impl Default for ReviewSettings {
    fn default() -> Self {
        Self {
            chunk_files: DEFAULT_CHUNK_FILES,
            exclude: Vec::new(),
            passes: DEFAULT_REVIEW_PASSES,
            seed: None,
        }
    }
}

impl ReviewSettings {
    pub fn chunk_files(&self) -> usize {
        self.chunk_files.max(1)
    }

    /// The seed command for a workspace at `root`: the configured one, else
    /// `verify`, else `cargo check` for a Cargo workspace, else nothing.
    pub fn seed_command(&self, root: &std::path::Path, verify: Option<&str>) -> Option<String> {
        if let Some(seed) = self
            .seed
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return Some(seed.to_string());
        }
        if let Some(verify) = verify.map(str::trim).filter(|v| !v.is_empty()) {
            return Some(verify.to_string());
        }
        root.join("Cargo.toml")
            .is_file()
            .then(|| CARGO_SEED.to_string())
    }
}

/// Default seed for a Cargo workspace: compile errors and warnings in every
/// target, no test run.
pub const CARGO_SEED: &str = "cargo check --workspace --all-targets";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_command_prefers_config_then_verify_then_cargo() {
        let dir = tempfile::tempdir().unwrap();
        let settings = ReviewSettings::default();
        assert_eq!(settings.seed_command(dir.path(), None), None);
        assert_eq!(
            settings.seed_command(dir.path(), Some("make test")),
            Some("make test".into())
        );
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        assert_eq!(
            settings.seed_command(dir.path(), None),
            Some(CARGO_SEED.into())
        );
        assert_eq!(
            settings.seed_command(dir.path(), Some("  ")),
            Some(CARGO_SEED.into()),
            "a blank verify command is no verify command"
        );
        let configured = ReviewSettings {
            seed: Some("npm test".into()),
            ..ReviewSettings::default()
        };
        assert_eq!(
            configured.seed_command(dir.path(), Some("make test")),
            Some("npm test".into())
        );
        assert_eq!(
            ReviewSettings {
                chunk_files: 0,
                ..ReviewSettings::default()
            }
            .chunk_files(),
            1
        );
    }
}
