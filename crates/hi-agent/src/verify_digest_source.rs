use std::path::Path;

const SPAN_CONTEXT_LINES: usize = 4;
const MAX_SOURCE_REGION_FILE_BYTES: u64 = 256 * 1024;
const MAX_SOURCE_LINE_BYTES: usize = 512;

/// A bounded convenience excerpt; oversized lines need bounded shell reads.
/// Diagnostic paths must resolve to regular files inside the workspace.
pub(super) fn source_region(root: &Path, location: &str) -> Option<String> {
    let mut parts = location.split(':');
    let path = parts.next()?;
    let line: usize = parts.next()?.parse().ok()?;
    let relative = Path::new(path);
    if line == 0
        || relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return None;
    }
    let root = root.canonicalize().ok()?;
    let file_path = root.join(relative).canonicalize().ok()?;
    if !file_path.starts_with(&root) {
        return None;
    }
    let bytes =
        hi_tools::read_regular_file_bytes_bounded(&file_path, MAX_SOURCE_REGION_FILE_BYTES).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    if line > lines.len() {
        return None;
    }
    let start = line.saturating_sub(SPAN_CONTEXT_LINES + 1);
    let end = line.saturating_add(SPAN_CONTEXT_LINES).min(lines.len());
    let mut region = format!("   source ({path}:{}-{end}):\n", start + 1);
    for (offset, content) in lines[start..end].iter().enumerate() {
        let number = start + offset + 1;
        let marker = if number == line { ">" } else { " " };
        let mut end = content.len().min(MAX_SOURCE_LINE_BYTES);
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        region.push_str(&format!("   {marker}{number:>5} | {}", &content[..end]));
        if end < content.len() {
            region.push_str(" … [line truncated; inspect with a bounded shell command]");
        }
        region.push('\n');
    }
    Some(region)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minified_source_excerpt_is_bounded_and_reports_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let source = "界".repeat(80_000);
        std::fs::write(dir.path().join("minified.rs"), &source).unwrap();
        let excerpt = source_region(dir.path(), "minified.rs:1:1").unwrap();
        assert!(excerpt.contains(">    1 |"));
        assert!(excerpt.contains("line truncated; inspect with a bounded shell command"));
        assert!(excerpt.len() < 650, "{}", excerpt.len());
        assert!(excerpt.contains(&"界".repeat(170)));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("minified.rs")).unwrap(),
            source
        );
        eprintln!(
            "source excerpt fixture: {} source bytes -> {} excerpt bytes",
            source.len(),
            excerpt.len()
        );
    }

    #[test]
    fn ordinary_source_is_exact_and_invalid_locations_are_omitted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "first\nsecond\nthird\n").unwrap();
        assert_eq!(
            source_region(dir.path(), "lib.rs:2:4").unwrap(),
            "   source (lib.rs:1-3):\n        1 | first\n   >    2 | second\n        3 | third\n"
        );
        for location in [
            "lib.rs:0:1",
            "lib.rs:4:1",
            "../lib.rs:1:1",
            "lib.rs:18446744073709551615:1",
        ] {
            assert!(source_region(dir.path(), location).is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn source_excerpt_does_not_follow_links_outside_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("private.rs"), "outside\n").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("private.rs"),
            dir.path().join("link.rs"),
        )
        .unwrap();
        assert!(source_region(dir.path(), "link.rs:1:1").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn source_excerpt_rejects_fifo_without_opening_it_for_a_blocking_read() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(dir.path().join("pipe.rs"))
                .status()
                .unwrap()
                .success()
        );
        assert!(source_region(dir.path(), "pipe.rs:1:1").is_none());
    }

    #[test]
    fn large_failure_digest_lists_a_sample_but_signs_every_failure() {
        let raw = (0..1_000)
            .map(|n| format!("test case_{n:04} ... FAILED\n"))
            .collect::<String>();
        let digest = super::super::digest_failure(Path::new("/nonexistent"), &raw).unwrap();
        assert_eq!(digest.failure_count, 1_000);
        assert_eq!(digest.signature.len(), 1_000);
        assert!(digest.text.contains("1000 failing test(s)"));
        assert!(
            digest
                .text
                .contains("992 more failing test(s) in the full output below")
        );
        assert!(digest.text.len() < 300, "{}", digest.text);
        let changed = raw.replace("case_0999", "different_failure");
        let changed = super::super::digest_failure(Path::new("/nonexistent"), &changed).unwrap();
        assert_ne!(digest.signature, changed.signature);
        eprintln!(
            "failure-name fixture: {} raw bytes -> {} digest bytes",
            raw.len(),
            digest.text.len()
        );
    }
}
