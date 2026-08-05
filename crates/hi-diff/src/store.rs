use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use blake3::Hasher;
use serde::Serialize;

use crate::{ArtifactRef, DiffCase, DiffRunSnapshot, DiffRunSpec};

#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("creating diff artifact root {}", root.display()))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn put_bytes(&self, kind: &str, bytes: &[u8]) -> Result<ArtifactRef> {
        let mut hasher = Hasher::new();
        hasher.update(bytes);
        let digest = hasher.finalize().to_hex().to_string();
        let relative_path = format!("blobs/{}/{}-{}", &digest[..2], digest, sanitize(kind));
        let path = self.root.join(&relative_path);
        if !path.exists() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let temporary = path.with_extension("tmp");
            std::fs::write(&temporary, bytes)?;
            std::fs::rename(&temporary, &path)?;
        }
        Ok(ArtifactRef {
            digest,
            relative_path,
            kind: kind.to_string(),
            bytes: bytes.len() as u64,
        })
    }

    pub fn put_json<T: Serialize>(&self, kind: &str, value: &T) -> Result<ArtifactRef> {
        self.put_bytes(kind, &serde_json::to_vec_pretty(value)?)
    }

    pub fn put_named_json<T: Serialize>(&self, name: &str, value: &T) -> Result<PathBuf> {
        let name = sanitize(name);
        let path = self.root.join(name);
        let bytes = serde_json::to_vec_pretty(value)?;
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(&temporary, &path)?;
        Ok(path)
    }

    pub fn read(&self, artifact: &ArtifactRef) -> Result<Vec<u8>> {
        Ok(std::fs::read(self.root.join(&artifact.relative_path))?)
    }
}

#[derive(Clone, Debug)]
pub struct RunStore {
    artifacts: ArtifactStore,
}

impl RunStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            artifacts: ArtifactStore::new(root)?,
        })
    }

    pub fn artifacts(&self) -> &ArtifactStore {
        &self.artifacts
    }

    pub fn write_spec(&self, spec: &DiffRunSpec) -> Result<ArtifactRef> {
        self.artifacts
            .put_json(&format!("run-{}-spec.json", spec.run_id), spec)
    }

    pub fn write_case(&self, run_id: &str, case: &DiffCase) -> Result<ArtifactRef> {
        self.artifacts
            .put_json(&format!("run-{run_id}-case-{}.json", case.id()), case)
    }

    pub fn write_snapshot(&self, snapshot: &DiffRunSnapshot) -> Result<ArtifactRef> {
        self.artifacts
            .put_json(&format!("run-{}-snapshot.json", snapshot.run_id), snapshot)
    }

    pub fn write_named_snapshot(&self, snapshot: &DiffRunSnapshot) -> Result<PathBuf> {
        self.artifacts
            .put_named_json(&format!("run-{}-snapshot.json", snapshot.run_id), snapshot)
    }

    pub fn write_named_spec(&self, spec: &DiffRunSpec) -> Result<PathBuf> {
        self.artifacts
            .put_named_json(&format!("run-{}-spec.json", spec.run_id), spec)
    }

    pub fn append_event(&self, run_id: &str, event: &crate::DiffEvent) -> Result<()> {
        let path = self
            .artifacts
            .root()
            .join(format!("run-{run_id}-events.jsonl"));
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, event)?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }
}

fn sanitize(kind: &str) -> String {
    kind.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn default_root() -> PathBuf {
    if let Some(root) = std::env::var_os("HI_DIFF_DIR") {
        return PathBuf::from(root);
    }
    if let Some(root) = std::env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(root).join("hi/diff");
    }
    if let Some(root) = std::env::var_os("HOME") {
        return PathBuf::from(root).join(".local/state/hi/diff");
    }
    PathBuf::from(".hi/diff")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_addressed_artifacts_are_reusable() {
        let temp = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(temp.path()).unwrap();
        let first = store.put_bytes("summary.json", b"hello").unwrap();
        let second = store.put_bytes("summary.json", b"hello").unwrap();
        assert_eq!(first.digest, second.digest);
        assert_eq!(store.read(&first).unwrap(), b"hello");
    }
}
