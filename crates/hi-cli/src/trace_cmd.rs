//! `hi trace` — human-readable views of recorded run traces.
//!
//! `hi trace show` reads the most recent local trace's manifest and prints its
//! root hash and attestation label; `hi trace list` shows recent runs with
//! their labels and completeness. A self-hosted run visibly reports
//! `local-unattested:…` (and a managed run its worker attestation) without
//! digging through JSON. This surfaces the trust boundary documented in
//! `docs/architecture.md`: the label is what distinguishes worker-anchored
//! evidence from a locally consistent, unattested chain.

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

/// A trace directory paired with its last-modified time, for ordering.
struct TraceDir {
    path: PathBuf,
    modified: std::time::SystemTime,
}

/// All trace directories under `root` that contain a manifest, newest first.
fn trace_dirs(root: &Path) -> Vec<TraceDir> {
    let mut dirs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || !path.join("manifest.json").exists() {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            dirs.push(TraceDir { path, modified });
        }
    }
    dirs.sort_by_key(|d| std::cmp::Reverse(d.modified));
    dirs
}

/// The most recently modified trace directory under `root`, or `None`.
fn latest_trace_dir(root: &Path) -> Option<PathBuf> {
    trace_dirs(root).into_iter().next().map(|d| d.path)
}

/// Load and parse a trace directory's manifest.
fn load_manifest(dir: &Path) -> Result<TraceManifest> {
    let manifest_path = dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("reading trace manifest {}", manifest_path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing trace manifest {}", manifest_path.display()))
}

/// Short attestation label for list output: the scheme prefix (the part before
/// the first `:`), which is what distinguishes worker-anchored evidence from a
/// self-hosted `local-unattested` chain, without the long hash.
fn attestation_label(manifest: &TraceManifest) -> &str {
    manifest
        .attestation
        .as_deref()
        .and_then(|a| a.split(':').next())
        .filter(|s| !s.is_empty())
        .unwrap_or("unattested")
}

/// Render the human-readable summary lines for a trace directory.
fn render(dir: &Path) -> Result<Vec<String>> {
    let manifest = load_manifest(dir)?;
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

/// Render the recent-runs table for `hi trace list`, newest first.
fn render_list(root: &Path, limit: usize) -> Result<Vec<String>> {
    let dirs = trace_dirs(root);
    if dirs.is_empty() {
        bail!("no traces found under {}", root.display());
    }
    let mut lines = vec![
        format!("{:<34} {:<8} {:<9} {:<7} {:<18} ROOT", "TRACE", "MODE", "COMPLETE", "EVENTS", "ATTESTATION"),
    ];
    for entry in dirs.iter().take(limit) {
        let manifest = load_manifest(&entry.path)?;
        let mode = match manifest.mode {
            hi_trace::TraceMode::Managed => "managed",
            hi_trace::TraceMode::Local => "local",
        };
        lines.push(format!(
            "{:<34} {:<8} {:<9} {:<7} {:<18} {}",
            manifest.trace_id,
            mode,
            manifest.complete,
            manifest.event_count,
            attestation_label(&manifest),
            &manifest.root_hash[..12.min(manifest.root_hash.len())],
        ));
    }
    Ok(lines)
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
        "list" => {
            let limit = args
                .get(1)
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(20);
            for line in render_list(&trace_root(), limit)? {
                println!("{line}");
            }
            Ok(())
        }
        other => bail!("usage: hi trace show | hi trace list [n]  (unknown subcommand '{other}')"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(dir: &Path, attestation: Option<&str>) {
        write_manifest_full(dir, attestation, true, 1);
    }

    fn write_manifest_full(dir: &Path, attestation: Option<&str>, complete: bool, events: u64) {
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        let manifest = serde_json::json!({
            "trace_schema": hi_trace::TRACE_SCHEMA_VERSION,
            "trace_id": dir.file_name().unwrap().to_str().unwrap(),
            "mode": "local",
            "event_count": events,
            "root_hash": "b".repeat(64),
            "complete": complete,
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

    #[test]
    fn list_shows_recent_runs_with_labels_and_completeness() {
        let root = std::env::temp_dir().join(format!("hi-trace-list-{}", std::process::id()));
        // Three runs: a complete local-unattested one, an incomplete one, and
        // an unattested one. Distinct trace ids let us assert each appears.
        let complete = root.join("d".repeat(32));
        write_manifest_full(&complete, Some(&format!("local-unattested:{}", "b".repeat(64))), true, 5);
        let incomplete = root.join("e".repeat(32));
        write_manifest_full(&incomplete, Some(&format!("local-unattested:{}", "b".repeat(64))), false, 2);
        let plain = root.join("f".repeat(32));
        write_manifest_full(&plain, None, true, 7);

        let lines = render_list(&root, 20).unwrap();
        let out = lines.join("\n");
        // Header + one row per trace.
        assert_eq!(lines.len(), 4, "expected header + 3 rows: {out}");
        assert!(out.contains("ATTESTATION"), "missing header: {out}");
        // Each trace id appears.
        for id in ["d".repeat(32), "e".repeat(32), "f".repeat(32)] {
            assert!(out.contains(&id), "missing trace {id}: {out}");
        }
        // Labels and completeness are surfaced.
        assert!(out.contains("local-unattested"), "label missing: {out}");
        assert!(out.contains("unattested"), "unattested marker missing: {out}");
        assert!(out.contains("false"), "incomplete run not shown: {out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_respects_limit() {
        let root = std::env::temp_dir().join(format!("hi-trace-list-limit-{}", std::process::id()));
        for ch in ["1", "2", "3"] {
            let dir = root.join(ch.repeat(32));
            write_manifest(&dir, None);
        }
        let lines = render_list(&root, 2).unwrap();
        // Header + 2 rows (limit), not 3.
        assert_eq!(lines.len(), 3, "limit not respected: {}", lines.join("\n"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_errors_when_no_traces() {
        let root = std::env::temp_dir().join(format!("hi-trace-list-empty-{}", std::process::id()));
        let result = render_list(&root, 20);
        assert!(result.is_err(), "expected an error for an empty trace root");
    }
}
