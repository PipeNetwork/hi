//! Crash-durable storage for sealed detached candidates.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use hi_workspace::ArtifactRef;
use serde::{Deserialize, Serialize};

use super::DetachedVerifiedCandidate;

const ARTIFACT_SCHEMA_VERSION: u16 = 2;
const LEGACY_ARTIFACT_SCHEMA_VERSION: u16 = 1;
const ARTIFACT_DIRECTORY: &str = "candidate-artifacts";

pub(crate) fn read_resource_body(state_root: &Path, reference: &str) -> Result<Option<String>> {
    let Some(token) = reference.strip_prefix("candidate/") else {
        return Ok(None);
    };
    ensure!(
        token.len() == blake3::OUT_LEN * 2 && token.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid candidate artifact resource identity"
    );
    let path = state_root
        .join(ARTIFACT_DIRECTORY)
        .join(format!("{token}.json"));
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading candidate artifact {}", path.display()));
        }
    };
    ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "candidate artifact exceeds the resource read limit"
    );
    // Reuse the durable decoder so a forged filename or malformed candidate
    // cannot be exposed merely because a caller guessed its resource URI.
    PersistedDetachedCandidate::from_bytes(path, bytes.clone())?;
    String::from_utf8(bytes)
        .context("candidate artifact is not UTF-8 JSON")
        .map(Some)
}

#[derive(Clone, Debug)]
pub struct PersistedDetachedCandidate {
    pub detached: DetachedVerifiedCandidate,
    pub artifact: ArtifactRef,
    path: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct StoredCandidate {
    schema_version: u16,
    detached: DetachedVerifiedCandidate,
}

impl PersistedDetachedCandidate {
    pub fn persist(detached: DetachedVerifiedCandidate, state_root: &Path) -> Result<Self> {
        detached.candidate.validate()?;
        let directory = state_root.join(ARTIFACT_DIRECTORY);
        let directory_existed = directory.exists();
        fs::create_dir_all(&directory).with_context(|| {
            format!(
                "creating candidate artifact directory {}",
                directory.display()
            )
        })?;
        if !directory_existed {
            sync_directory(state_root)?;
        }
        let token = artifact_token(&detached);
        let path = directory.join(format!("{token}.json"));
        let bytes = serde_json::to_vec(&StoredCandidate {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            detached,
        })
        .context("serializing sealed candidate artifact")?;

        if path.exists() {
            ensure!(
                fs::read(&path)? == bytes,
                "candidate artifact path already contains different bytes"
            );
        } else {
            let temporary = directory.join(format!(".{token}.{}.tmp", uuid::Uuid::new_v4()));
            let mut cleanup = TemporaryArtifact::new(temporary.clone());
            let mut file = create_private_file(&temporary)?;
            file.write_all(&bytes)
                .context("writing sealed candidate artifact")?;
            file.sync_all()
                .context("fsyncing sealed candidate artifact")?;
            fs::rename(&temporary, &path)
                .with_context(|| format!("publishing candidate artifact {}", path.display()))?;
            cleanup.disarm();
            sync_directory(&directory)?;
        }
        Self::from_bytes(path, bytes)
    }

    pub fn discover(state_root: &Path) -> Result<Vec<Self>> {
        let directory = state_root.join(ARTIFACT_DIRECTORY);
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error).context("reading candidate artifact directory"),
        };
        let mut artifacts = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)
                .with_context(|| format!("reading candidate artifact {}", path.display()))?;
            artifacts.push(Self::from_bytes(path, bytes)?);
        }
        artifacts.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(artifacts)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Remove evidence only after the workspace job's terminal transition has
    /// been acknowledged by its authoritative journal/controller.
    pub fn remove_after_terminal(self) -> Result<()> {
        let parent = self
            .path
            .parent()
            .context("candidate artifact has no parent directory")?;
        ensure!(
            parent.file_name().and_then(|value| value.to_str()) == Some(ARTIFACT_DIRECTORY),
            "refusing to remove candidate artifact outside its owned directory"
        );
        match fs::remove_file(&self.path) {
            Ok(()) => sync_directory(parent),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .with_context(|| format!("removing candidate artifact {}", self.path.display())),
        }
    }

    fn from_bytes(path: PathBuf, bytes: Vec<u8>) -> Result<Self> {
        let stored: StoredCandidate = serde_json::from_slice(&bytes)
            .with_context(|| format!("decoding candidate artifact {}", path.display()))?;
        ensure!(
            matches!(
                stored.schema_version,
                LEGACY_ARTIFACT_SCHEMA_VERSION | ARTIFACT_SCHEMA_VERSION
            ),
            "unsupported candidate artifact schema {}",
            stored.schema_version
        );
        stored.detached.candidate.validate()?;
        let expected = format!("{}.json", artifact_token(&stored.detached));
        ensure!(
            path.file_name().and_then(|value| value.to_str()) == Some(expected.as_str()),
            "candidate artifact filename does not match its sealed identity"
        );
        Ok(Self {
            detached: stored.detached,
            artifact: ArtifactRef {
                uri: format!(
                    "artifact://candidate/{}",
                    expected.trim_end_matches(".json")
                ),
                digest: Some(format!("blake3:{}", blake3::hash(&bytes).to_hex())),
                size_bytes: Some(bytes.len() as u64),
            },
            path,
        })
    }
}

impl std::ops::Deref for PersistedDetachedCandidate {
    type Target = DetachedVerifiedCandidate;

    fn deref(&self) -> &Self::Target {
        &self.detached
    }
}

fn artifact_token(detached: &DetachedVerifiedCandidate) -> String {
    let identity = format!(
        "{}\0{}\0{}\0{}",
        detached.candidate.candidate_id,
        detached.candidate.job_id,
        detached.candidate.candidate_digest,
        detached.source_snapshot_id
    );
    blake3::hash(identity.as_bytes()).to_hex().to_string()
}

#[cfg(unix)]
fn create_private_file(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    Ok(OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?)
}

#[cfg(not(unix))]
fn create_private_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

struct TemporaryArtifact {
    path: PathBuf,
    armed: bool,
}

impl TemporaryArtifact {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryArtifact {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}
