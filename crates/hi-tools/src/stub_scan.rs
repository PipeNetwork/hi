//! Deterministic stub-marker scan over a set of changed files. Feeds the
//! long-horizon skeptic gate: an "implement X" milestone whose turn leaves
//! `todo!()`s in its own changed files is concrete evidence for an objection.
//! Deliberately conservative and extension-gated — plain Python `pass` and
//! prose TODOs are legitimate, so they are not markers. Not a lint; a signal.

use std::io::Read;
use std::path::Path;

/// One stub marker found in a scanned file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StubFinding {
    /// The path as given (workspace-relative).
    pub path: String,
    /// 1-based line number.
    pub line: usize,
    /// The matched marker text.
    pub marker: String,
}

/// Per-file read ceiling — a stub-bearing source file is small; anything
/// bigger is generated/vendored and not worth scanning.
const MAX_FILE_BYTES: u64 = 256 * 1024;
/// Findings ceiling so the skeptic context stays bounded.
const MAX_FINDINGS: usize = 20;

/// Markers by extension. Rust's `todo!(`/`unimplemented!(` and Python's
/// `raise NotImplementedError` are unambiguous placeholders; `pass  # TODO`
/// catches the deliberate-stub idiom without flagging every bare `pass`.
fn markers_for(path: &Path) -> &'static [&'static str] {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => &["todo!(", "unimplemented!("],
        Some("py") => &["raise NotImplementedError", "pass  # TODO", "pass # TODO"],
        _ => &[],
    }
}

/// Scan `paths` (workspace-relative) under `root` for stub markers. Bounded:
/// at most `max_files` files, each ≤ 256 KB; unreadable, binary
/// (NUL-containing), and unrecognized-extension files are skipped silently.
/// Returns at most 20 findings, in input order.
pub fn scan_paths(root: &Path, paths: &[String], max_files: usize) -> Vec<StubFinding> {
    let mut findings = Vec::new();
    for path in paths.iter().take(max_files) {
        let markers = markers_for(Path::new(path));
        if markers.is_empty() {
            continue;
        }
        let absolute = root.join(path);
        let Ok(file) = std::fs::File::open(&absolute) else {
            continue;
        };
        let mut bytes = Vec::new();
        if file
            .take(MAX_FILE_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .is_err()
            || bytes.len() as u64 > MAX_FILE_BYTES
            || bytes.contains(&0)
        {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            for marker in markers {
                if line.contains(marker) {
                    findings.push(StubFinding {
                        path: path.clone(),
                        line: i + 1,
                        marker: (*marker).to_string(),
                    });
                    if findings.len() >= MAX_FINDINGS {
                        return findings;
                    }
                    break; // one finding per line is enough
                }
            }
        }
    }
    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hi-stub-scan-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn detects_rust_and_python_stub_markers() {
        let root = temp_root("detects");
        std::fs::write(
            root.join("lib.rs"),
            "pub fn real() {}\npub fn fake() { todo!(\"later\") }\nfn nope() { unimplemented!() }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("train.py"),
            "def step():\n    raise NotImplementedError\n",
        )
        .unwrap();
        let findings = scan_paths(&root, &["lib.rs".into(), "train.py".into()], 50);
        assert_eq!(findings.len(), 3);
        assert_eq!(findings[0].path, "lib.rs");
        assert_eq!(findings[0].line, 2, "1-based line numbers");
        assert_eq!(findings[0].marker, "todo!(");
        assert_eq!(findings[1].marker, "unimplemented!(");
        assert_eq!(findings[2].path, "train.py");
        assert_eq!(findings[2].marker, "raise NotImplementedError");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ignores_plain_python_pass_and_unlisted_files() {
        let root = temp_root("ignores");
        std::fs::write(
            root.join("ok.py"),
            "class Marker:\n    pass\n\ndef stub():\n    pass  # TODO fill in\n",
        )
        .unwrap();
        std::fs::write(root.join("unlisted.rs"), "fn f() { todo!() }\n").unwrap();
        std::fs::write(root.join("notes.md"), "todo!( in prose\n").unwrap();
        let findings = scan_paths(&root, &["ok.py".into(), "notes.md".into()], 50);
        assert_eq!(findings.len(), 1, "bare pass and .md are not markers");
        assert_eq!(findings[0].line, 5);
        assert_eq!(findings[0].marker, "pass  # TODO");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_by_file_size_count_and_binary_skip() {
        let root = temp_root("bounds");
        let mut big = String::from("fn f() { todo!() }\n");
        big.push_str(&"// padding\n".repeat(30_000));
        std::fs::write(root.join("big.rs"), &big).unwrap();
        std::fs::write(root.join("bin.rs"), b"fn f() { todo!() }\x00\n").unwrap();
        std::fs::write(root.join("late.rs"), "fn f() { todo!() }\n").unwrap();
        std::fs::write(root.join("missing-ok.rs"), "fn f() { todo!() }\n").unwrap();
        // big.rs exceeds the size cap, bin.rs is binary, late.rs is beyond
        // max_files, gone.rs doesn't exist.
        let findings = scan_paths(
            &root,
            &[
                "big.rs".into(),
                "bin.rs".into(),
                "gone.rs".into(),
                "late.rs".into(),
            ],
            3,
        );
        assert!(findings.is_empty(), "findings: {findings:?}");
        let findings = scan_paths(&root, &["missing-ok.rs".into()], 3);
        assert_eq!(findings.len(), 1, "a normal file still scans");
        std::fs::remove_dir_all(root).unwrap();
    }
}
