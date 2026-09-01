//! `hi trace` — human-readable views of recorded run traces.
//!
//! `hi trace show` reads a trace's manifest and prints its root hash,
//! attestation label, and inline integrity (tamper) status; `hi trace list`
//! shows recent runs with their labels and completeness; `hi trace verify`
//! runs the integrity check alone and fails on a broken chain. A self-hosted
//! run visibly reports its `local-signed:…` attestation (and a managed run its
//! worker attestation) without digging through JSON. This surfaces the trust
//! boundary documented in `docs/architecture.md`: the label is what
//! distinguishes worker-anchored evidence from a locally signed chain, and
//! integrity is local consistency, not authenticity.

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

/// Resolve a user-supplied trace id (full or an unambiguous prefix, as printed
/// by `hi trace list`) to its directory under `root`.
fn find_trace_dir(root: &Path, id: &str) -> Result<PathBuf> {
    let matches: Vec<PathBuf> = trace_dirs(root)
        .into_iter()
        .filter_map(|d| {
            d.path
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| name.starts_with(id))
                .map(|_| d.path.clone())
        })
        .collect();
    match matches.len() {
        0 => bail!("no trace matching '{id}' under {}", root.display()),
        1 => Ok(matches.into_iter().next().unwrap()),
        _ => bail!(
            "trace id prefix '{id}' is ambiguous ({} matches); use more characters",
            matches.len()
        ),
    }
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
/// self-hosted `local-signed` chain, without the long hash.
fn attestation_label(manifest: &TraceManifest) -> &str {
    manifest
        .attestation
        .as_deref()
        .and_then(|a| a.split(':').next())
        .filter(|s| !s.is_empty())
        .unwrap_or("unattested")
}

/// Run the integrity check and return a human-readable status line. Never
/// fails — a broken chain is reported as a status, not an error, so `show`
/// can surface tamper evidence inline.
fn integrity_status(dir: &Path) -> String {
    match hi_trace::validate_trace(dir, hi_trace::DEFAULT_RUN_MAX_BYTES, 1_000_000) {
        Ok(_) => "ok (hash chain + blobs match manifest)".to_string(),
        Err(e) => format!("TAMPERED ({e:#})"),
    }
}

/// Compact integrity marker for the list table: `ok` or `TAMPERED`.
fn integrity_flag(dir: &Path) -> &'static str {
    if hi_trace::validate_trace(dir, hi_trace::DEFAULT_RUN_MAX_BYTES, 1_000_000).is_ok() {
        "ok"
    } else {
        "TAMPERED"
    }
}

/// Render the human-readable summary lines for a trace directory.
fn render(dir: &Path) -> Result<Vec<String>> {
    let manifest = load_manifest(dir)?;
    let attestation = manifest
        .attestation
        .as_deref()
        .unwrap_or("none (unattested)");
    let mode = match manifest.mode {
        hi_trace::TraceMode::Managed => "managed",
        hi_trace::TraceMode::Local => "local",
    };
    Ok(vec![
        format!("trace:       {}", manifest.trace_id),
        format!("mode:        {mode}"),
        format!("complete:    {}", manifest.complete),
        format!("integrity:   {}", integrity_status(dir)),
        format!("events:      {}", manifest.event_count),
        format!("root_hash:   {}", manifest.root_hash),
        format!("attestation: {attestation}"),
        format!("path:        {}", dir.display()),
    ])
}

/// Verify a trace's integrity: recompute the event hash chain and blob hashes
/// from the on-disk files and check them against the manifest. This is the
/// local tamper-evidence check — it proves the trace is internally consistent
/// (uncorrupted, unspliced), not that it is authentic; authenticity requires
/// the external worker's attestation (see `docs/architecture.md`).
///
/// When the trace carries a `local-signed:` attestation, also validate the
/// ed25519 signature against the local signing key, reporting it as a distinct
/// line so a forged or unverifiable signature is visible separately from the
/// chain check.
fn verify(dir: &Path) -> Result<Vec<String>> {
    let manifest = hi_trace::validate_trace(dir, hi_trace::DEFAULT_RUN_MAX_BYTES, 1_000_000)
        .context("trace failed integrity validation")?;
    let attestation = manifest
        .attestation
        .as_deref()
        .unwrap_or("none (unattested)");
    let signature_line = signature_status(&manifest);
    let mut lines = vec![
        format!("trace:       {}", manifest.trace_id),
        "integrity:   ok (hash chain + blobs match manifest)".to_string(),
        format!("events:      {}", manifest.event_count),
        format!("root_hash:   {}", manifest.root_hash),
        format!("attestation: {attestation}"),
    ];
    if let Some(line) = signature_line {
        lines.push(format!("signature:   {line}"));
    }
    lines.push(
        "note:        integrity is local consistency; authenticity needs worker attestation"
            .to_string(),
    );
    Ok(lines)
}

/// Validate a `local-signed:` attestation's signature against the local key.
/// Returns `None` when the trace is not locally signed (nothing to check), or
/// `Some(status)` describing the signature outcome.
fn signature_status(manifest: &hi_trace::TraceManifest) -> Option<String> {
    signature_status_with_key(manifest, &hi_trace::local_signing_key_path())
}

/// Key-path-injectable core of [`signature_status`] so tests can point at a
/// temp key without touching the shared default location or the environment.
fn signature_status_with_key(
    manifest: &hi_trace::TraceManifest,
    key_path: &Path,
) -> Option<String> {
    let attestation = manifest.attestation.as_deref()?;
    if !attestation.starts_with(hi_trace::LOCAL_SIGNED_PREFIX) {
        return None;
    }
    if !key_path.exists() {
        return Some("unverifiable (local signing key not found)".to_string());
    }
    match hi_trace::verify_local_signature(attestation, &manifest.root_hash, key_path) {
        Ok(true) => Some("ok (ed25519 signature matches local key)".to_string()),
        Ok(false) => Some("MISMATCH (signature does not match local key)".to_string()),
        Err(e) => Some(format!("error ({e:#})")),
    }
}

/// Render the recent-runs table for `hi trace list`, newest first.
fn render_list(root: &Path, limit: usize) -> Result<Vec<String>> {
    let dirs = trace_dirs(root);
    if dirs.is_empty() {
        bail!("no traces found under {}", root.display());
    }
    let mut lines = vec![format!(
        "{:<34} {:<8} {:<9} {:<9} {:<7} {:<18} ROOT",
        "TRACE", "MODE", "COMPLETE", "INTEGRITY", "EVENTS", "ATTESTATION"
    )];
    for entry in dirs.iter().take(limit) {
        let manifest = load_manifest(&entry.path)?;
        let mode = match manifest.mode {
            hi_trace::TraceMode::Managed => "managed",
            hi_trace::TraceMode::Local => "local",
        };
        lines.push(format!(
            "{:<34} {:<8} {:<9} {:<9} {:<7} {:<18} {}",
            manifest.trace_id,
            mode,
            manifest.complete,
            integrity_flag(&entry.path),
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
            let dir = match args.get(1).map(String::as_str) {
                Some(id) => find_trace_dir(&root, id)?,
                None => latest_trace_dir(&root)
                    .ok_or_else(|| anyhow::anyhow!("no traces found under {}", root.display()))?,
            };
            for line in render(&dir)? {
                println!("{line}");
            }
            Ok(())
        }
        "verify" => {
            let root = trace_root();
            let dir = match args.get(1).map(String::as_str) {
                Some(id) => find_trace_dir(&root, id)?,
                None => latest_trace_dir(&root)
                    .ok_or_else(|| anyhow::anyhow!("no traces found under {}", root.display()))?,
            };
            for line in verify(&dir)? {
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
        other => bail!(
            "usage: hi trace show [id] | hi trace list [n] | hi trace verify [id]  (unknown subcommand '{other}')\n\
             local-signed attestations use the key at $XDG_STATE_HOME/hi/trace-signing-key"
        ),
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
    fn render_shows_attestation_label_and_root_hash() {
        // A legacy local-unattested label (or any attestation string present in
        // the manifest) renders verbatim in the show output.
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
    fn render_shows_local_signed_label_and_root_hash() {
        // The scheme LocalAttestor currently emits: local-signed:<hex-sig>.
        // The show path must surface it verbatim like any other label.
        let root =
            std::env::temp_dir().join(format!("hi-trace-show-signed-{}", std::process::id()));
        let dir = root.join("b".repeat(32));
        let sig = format!("local-signed:{}", "9".repeat(128));
        write_manifest(&dir, Some(&sig));
        let out = render(&dir).unwrap().join("\n");
        assert!(
            out.contains(&format!("attestation: {sig}")),
            "local-signed label not surfaced: {out}"
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
        // Three runs: a complete local-signed one, an incomplete one, and
        // an unattested one. Distinct trace ids let us assert each appears.
        let complete = root.join("d".repeat(32));
        write_manifest_full(
            &complete,
            Some(&format!("local-signed:{}", "b".repeat(128))),
            true,
            5,
        );
        let incomplete = root.join("e".repeat(32));
        write_manifest_full(
            &incomplete,
            Some(&format!("local-signed:{}", "b".repeat(128))),
            false,
            2,
        );
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
        assert!(out.contains("local-signed"), "label missing: {out}");
        assert!(
            out.contains("unattested"),
            "unattested marker missing: {out}"
        );
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

    /// Build two real traces — one clean, one tampered — and assert the list
    /// table marks their integrity column accordingly.
    #[test]
    fn list_shows_integrity_ok_and_tampered() {
        let root = std::env::temp_dir().join(format!("hi-trace-list-integ-{}", std::process::id()));
        let clean = root.join("7".repeat(32));
        write_real_trace(&clean);
        let broken = root.join("8".repeat(32));
        write_real_trace(&broken);
        tamper_first_event(&broken);

        let lines = render_list(&root, 20).unwrap();
        let out = lines.join("\n");
        assert!(out.contains("INTEGRITY"), "missing integrity header: {out}");
        // The clean row shows ok on its line; the tampered row shows TAMPERED.
        let clean_row = lines.iter().find(|l| l.contains(&"7".repeat(32))).unwrap();
        let broken_row = lines.iter().find(|l| l.contains(&"8".repeat(32))).unwrap();
        assert!(clean_row.contains("ok"), "clean row not ok: {clean_row}");
        assert!(
            broken_row.contains("TAMPERED"),
            "tampered row not flagged: {broken_row}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_errors_when_no_traces() {
        let root = std::env::temp_dir().join(format!("hi-trace-list-empty-{}", std::process::id()));
        let result = render_list(&root, 20);
        assert!(result.is_err(), "expected an error for an empty trace root");
    }

    #[test]
    fn find_trace_dir_resolves_exact_and_prefix() {
        let root = std::env::temp_dir().join(format!("hi-trace-find-{}", std::process::id()));
        let dir = root.join("a".repeat(32));
        write_manifest(&dir, None);
        // Exact id.
        assert_eq!(find_trace_dir(&root, &"a".repeat(32)).unwrap(), dir);
        // Unambiguous prefix.
        assert_eq!(find_trace_dir(&root, &"a".repeat(8)).unwrap(), dir);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn find_trace_dir_rejects_unknown_and_ambiguous() {
        let root = std::env::temp_dir().join(format!("hi-trace-find-err-{}", std::process::id()));
        // Two traces sharing a prefix: 111… and 11f… — prefix "11" is ambiguous.
        write_manifest(&root.join("1".repeat(32)), None);
        write_manifest(&root.join(format!("11{}", "f".repeat(30))), None);
        // Unknown id.
        assert!(find_trace_dir(&root, &"9".repeat(32)).is_err());
        // Ambiguous prefix.
        let err = find_trace_dir(&root, "11").unwrap_err();
        assert!(
            format!("{err:#}").contains("ambiguous"),
            "expected ambiguity error, got: {err:#}"
        );
        // Longer prefix disambiguates.
        assert!(find_trace_dir(&root, "11f").is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Build a genuine trace (valid hash chain) at `dir` so validate_trace has
    /// real content to recompute. Returns the TraceWriter's root hash.
    fn write_real_trace(dir: &Path) -> String {
        let mut writer =
            hi_trace::TraceWriter::create(dir, hi_trace::TraceMode::Local, 1 << 20).unwrap();
        writer
            .record("step", "step", 1, None, None, serde_json::json!({"n": 1}))
            .unwrap();
        let summary = writer.finalize().unwrap();
        summary.root_hash
    }

    /// Build a manifest carrying a real `local-signed:` attestation over
    /// `root_hash`, signed with the key at `key_path`.
    fn signed_manifest(root_hash: &str, key_path: &Path) -> hi_trace::TraceManifest {
        let key = hi_trace::load_or_create_signing_key(key_path).unwrap();
        let attestation = hi_trace::sign_root_hash(&key, root_hash).unwrap();
        serde_json::from_value(serde_json::json!({
            "trace_schema": hi_trace::TRACE_SCHEMA_VERSION,
            "trace_id": "a".repeat(32),
            "mode": "local",
            "event_count": 1,
            "root_hash": root_hash,
            "complete": true,
            "fully_observed": true,
            "total_bytes": 0,
            "blobs": [],
            "attestation": attestation,
        }))
        .unwrap()
    }

    #[test]
    fn signature_status_ok_mismatch_and_missing_key() {
        let root = std::env::temp_dir().join(format!("hi-sig-status-{}", std::process::id()));
        let key_path = root.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        let root_hash = "a".repeat(64);
        let manifest = signed_manifest(&root_hash, &key_path);

        // Valid signature verifies.
        let status = signature_status_with_key(&manifest, &key_path).unwrap();
        assert!(status.starts_with("ok"), "expected ok, got: {status}");

        // Wrong root hash -> mismatch.
        let mut forged = signed_manifest(&root_hash, &key_path);
        forged.root_hash = "b".repeat(64);
        let status = signature_status_with_key(&forged, &key_path).unwrap();
        assert!(
            status.starts_with("MISMATCH"),
            "expected mismatch, got: {status}"
        );

        // Missing key -> unverifiable.
        let missing = root.join("nope").join("key");
        let status = signature_status_with_key(&manifest, &missing).unwrap();
        assert!(
            status.starts_with("unverifiable"),
            "expected unverifiable, got: {status}"
        );

        // Not a local-signed attestation -> None (nothing to check).
        let mut plain = signed_manifest(&root_hash, &key_path);
        plain.attestation = Some("local-unattested:abc".to_string());
        assert!(signature_status_with_key(&plain, &key_path).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_valid_trace_reports_ok() {
        let root = std::env::temp_dir().join(format!("hi-trace-verify-ok-{}", std::process::id()));
        let dir = root.join("d".repeat(32));
        let root_hash = write_real_trace(&dir);
        let out = verify(&dir).unwrap().join("\n");
        assert!(
            out.contains("integrity:   ok"),
            "expected ok integrity: {out}"
        );
        assert!(
            out.contains(&format!("root_hash:   {root_hash}")),
            "root hash mismatch: {out}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_detects_tampered_event_hash() {
        let root = std::env::temp_dir().join(format!("hi-trace-verify-bad-{}", std::process::id()));
        let dir = root.join("e".repeat(32));
        write_real_trace(&dir);
        tamper_first_event(&dir);

        let result = verify(&dir);
        assert!(result.is_err(), "tampered chain must fail validation");
        let err = result.unwrap_err();
        assert!(
            format!("{err:#}").contains("integrity"),
            "expected an integrity error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Corrupt one event's hash in the journal so the recomputed chain diverges
    /// from the manifest root.
    fn tamper_first_event(dir: &Path) {
        let journal = dir.join("events.jsonl");
        let mut lines: Vec<String> = std::fs::read_to_string(&journal)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        let mut event: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        event["event_hash"] = serde_json::json!("0".repeat(64));
        lines[0] = serde_json::to_string(&event).unwrap();
        std::fs::write(&journal, lines.join("\n") + "\n").unwrap();
    }

    #[test]
    fn show_surfaces_integrity_ok_inline() {
        let root = std::env::temp_dir().join(format!("hi-trace-show-ok-{}", std::process::id()));
        let dir = root.join("f".repeat(32));
        write_real_trace(&dir);
        let out = render(&dir).unwrap().join("\n");
        assert!(
            out.contains("integrity:   ok"),
            "show must surface ok integrity inline: {out}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn show_surfaces_tamper_status_without_failing() {
        let root = std::env::temp_dir().join(format!("hi-trace-show-bad-{}", std::process::id()));
        let dir = root.join("0".repeat(32));
        write_real_trace(&dir);
        tamper_first_event(&dir);
        // Unlike `verify`, `show` must not fail on a broken chain — it reports
        // the tamper status inline so the user sees it without a separate call.
        let out = render(&dir)
            .expect("show must not fail on tampered trace")
            .join("\n");
        assert!(
            out.contains("integrity:   TAMPERED"),
            "show must surface tamper status inline: {out}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
