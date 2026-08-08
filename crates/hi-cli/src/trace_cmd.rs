//! `hi trace show` — human-readable view of the latest run's trace summary.
//!
//! Reads the most recent local trace's manifest and prints its root hash and
//! attestation label, so a self-hosted run visibly reports `local-unattested:…`
//! (and a managed run its worker attestation) without digging through JSON.
//! This surfaces the trust boundary documented in `docs/architecture.md`: the
//! label is what distinguishes worker-anchored evidence from a locally
//! consistent, unattested chain.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hi_trace::TraceManifest;

/// State home matching `start_rsi_trace`'s resolution: `$XDG_STATE_HOME`, else
/// `$HOME/.local/state`, else the current directory.
fn state_home() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".local/state"))
        })
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The local trace root: `$state_home/hi/rsi` (see `TraceWriter::create_local`).
fn trace_root() -> PathBuf {
    state_home().join("hi").join("rsi")
}

/// The most recently modified trace directory under `root`, or `None`.
fn latest_trace_dir(root: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(root).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if !path.is_dir() || !path.join("manifest.json").exists() {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(t, _)| modified > *t) {
            best = Some((modified, path));
        }
    }
    best.map(|(_, path)| path)
}

/// Render the human-readable summary lines for a trace directory.
fn render(dir: &Path) -> Result<Vec<String>> {
    let manifest_path = dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("reading trace manifest {}", manifest_path.display()))?;
    let manifest: TraceManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing trace manifest {}", manifest_path.display()))?;
    let attestation = manifest.attestation.as_deref().unwrap_or("none (unattested)");
    let mode = match manifest.mode {
        hi_trace::TraceMode::Managed => "managed",
        hi_trace::TraceMode::Local => "local",
    };
    Ok(vec![
        format!("trace:       {}", manifest.trace_id),
        format!("mode:        {mode}"),
        format!("complete:    {}", manifest.complete),
        format!("events:      {}", manifest.event_count),
        format!("root_hash:   {}", manifest.root_hash),
        format!("attestation: {attestation}"),
        format!("path:        {}", dir.display()),
    ])
}

pub(crate) fn run_cli(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str).unwrap_or("show") {
        "show" => {
            let root = trace_root();
            let Some(dir) = latest_trace_dir(&root) else {
                bail!("no traces found under {}", root.display());
            };
            for line in render(&dir)? {
                println!("{line}");
            }
            Ok(())
        }
        other => bail!("usage: hi trace show  (unknown subcommand '{other}')"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(dir: &Path, attestation: Option<&str>) {
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        let manifest = serde_json::json!({
            "trace_schema": hi_trace::TRACE_SCHEMA_VERSION,
            "trace_id": dir.file_name().unwrap().to_str().unwrap(),
            "mode": "local",
            "event_count": 1,
            "root_hash": "b".repeat(64),
            "complete": true,
            "fully_observed": true,
            "total_bytes": 0,
            "blobs": [],
            "attestation": attestation,
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn render_shows_local_unattested_label_and_root_hash() {
        let root = std::env::temp_dir().join(format!("hi-trace-show-{}", std::process::id()));
        let dir = root.join("a".repeat(32));
        write_manifest(&dir, Some(&format!("local-unattested:{}", "b".repeat(64))));
        let lines = render(&dir).unwrap();
        let out = lines.join("\n");
        assert!(
            out.contains(&format!("attestation: local-unattested:{}", "b".repeat(64))),
            "label not surfaced: {out}"
        );
        assert!(
            out.contains(&format!("root_hash:   {}", "b".repeat(64))),
            "root hash not surfaced: {out}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn render_marks_unattested_when_no_label() {
        let root = std::env::temp_dir().join(format!("hi-trace-show-none-{}", std::process::id()));
        let dir = root.join("c".repeat(32));
        write_manifest(&dir, None);
        let out = render(&dir).unwrap().join("\n");
        assert!(
            out.contains("attestation: none (unattested)"),
            "expected explicit unattested marker: {out}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
