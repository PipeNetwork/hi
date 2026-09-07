use super::{
    apply_edit, apply_hunk_patch, apply_hunk_patch_text, apply_multi_patch_at, diff,
    edit_not_found_help, plan_multi_patch,
};

#[test]
fn patch_append_after_unterminated_context_preserves_line_boundary() {
    assert_eq!(
        apply_hunk_patch_text("alpha", &[" alpha", "+beta", "+gamma"]).unwrap(),
        "alpha\nbeta\ngamma"
    );
    assert_eq!(
        apply_hunk_patch_text("header\r\nalpha", &[" alpha", "+beta"]).unwrap(),
        "header\r\nalpha\r\nbeta"
    );
}

#[test]
fn edit_not_found_points_at_similar_lines() {
    let file = "fn a() {}\nfn target() {\n    do_thing();\n}\nfn b() {}\n";
    let help = edit_not_found_help(file, "fn target() {\n    do_OTHER();");
    assert!(help.contains("not found"), "{help}");
    // It surfaces the real nearby line with its number so the model can copy it.
    assert!(
        help.contains("fn target() {"),
        "shows the candidate: {help}"
    );
    assert!(help.contains("2\t"), "with a line number: {help}");
}

#[test]
fn diff_leads_with_a_change_summary() {
    // The diff a write/edit shows the user must say what changed up front,
    // not just trail off into raw +/- lines.
    let out = diff("one\ntwo\n", "one\nTWO\nthree\n");
    let first = out.lines().next().unwrap();
    assert!(first.contains("2 additions"), "summary: {first:?}");
    assert!(first.contains("1 deletion"), "summary: {first:?}");
    // Singular form when exactly one line changes.
    let single = diff("a\n", "a\nb\n");
    assert!(
        single.lines().next().unwrap().contains("1 addition,"),
        "singular: {single:?}"
    );
    assert_eq!(diff("same\n", "same\n"), "(no changes)");
}

#[test]
fn diff_shows_context_and_line_numbers() {
    // A change deep in a file must show its surrounding context with gutter
    // line numbers, so the reader can see *where* it lands — not just the
    // changed line floating context-free.
    let before = "a\nb\nc\nd\ne\nf\ng\n";
    let after = "a\nb\nc\nD\ne\nf\ng\n";
    let plain = strip_ansi(&diff(before, after));
    // Summary still leads.
    assert!(
        plain.lines().next().unwrap().contains("1 addition"),
        "summary: {plain}"
    );
    // Unchanged neighbours appear as context (proves we're not changed-only).
    assert!(
        plain.contains(" c\n") || plain.contains(" c"),
        "context: {plain}"
    );
    // The change is on line 4, numbered, with both old and new sides shown.
    assert!(plain.contains("4 - d"), "removed line w/ number: {plain}");
    assert!(plain.contains("4 + D"), "added line w/ number: {plain}");
    // Distant lines (line 1) are NOT shown — only context around the change.
    assert!(
        !plain.contains("1   a") && !plain.contains("1 + a"),
        "far context elided: {plain}"
    );
}

/// Strip ANSI SGR escapes (`\x1b[…m`) so tests can assert on plain text.
fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c2 in chars.by_ref() {
                if c2 == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn exact_unique_match() {
    assert_eq!(
        apply_edit("let x = 1;\n", "let x = 1;", "let x = 2;", false).unwrap(),
        "let x = 2;\n"
    );
}

#[test]
fn missing_old_string_errors() {
    assert!(apply_edit("foo\n", "bar", "baz", false).is_err());
}

#[test]
fn ambiguous_exact_match_errors() {
    assert!(apply_edit("x = 1\nx = 1\n", "x = 1", "y", false).is_err());
}

#[test]
fn tolerates_trailing_whitespace() {
    // The file has a stray trailing space the model's old_string lacks.
    assert_eq!(
        apply_edit("a\nb \nc\n", "a\nb\nc", "a\nB\nc", false).unwrap(),
        "a\nB\nc\n"
    );
}

#[test]
fn tolerates_crlf() {
    let out = apply_edit("a\r\nb\r\n", "a\nb", "X\nY", false).unwrap();
    assert_eq!(out, "X\r\nY\r\n");
}

#[test]
fn preserves_mixed_line_endings_in_matched_span() {
    let out = apply_edit("a\r\nb\nc\r\n", "a\nb\nc", "A\nB\nC", false).unwrap();
    assert_eq!(out, "A\r\nB\nC\r\n");
}

#[test]
fn tolerates_indentation_and_reindents() {
    // File indents 8 spaces; model used 4 — match anyway and re-indent `new`.
    assert_eq!(
        apply_edit(
            "def f():\n        return 0\n",
            "    return 0",
            "    return 1",
            false
        )
        .unwrap(),
        "def f():\n        return 1\n"
    );
}

#[test]
fn ambiguous_flexible_match_errors() {
    // Two lines match once indentation is ignored — refuse rather than guess.
    assert!(apply_edit("  x\n  x\n", "x ", "y", false).is_err());
}

#[test]
fn preserves_trailing_newline() {
    let out = apply_edit("first\nsecond\n", "second", "SECOND", false).unwrap();
    assert_eq!(out, "first\nSECOND\n");
}

#[test]
fn replace_all_swaps_every_occurrence() {
    let out = apply_edit("a\nb\na\nb\n", "a", "X", true).unwrap();
    assert_eq!(out, "X\nb\nX\nb\n");
}

#[test]
fn replace_all_with_no_match_errors() {
    assert!(apply_edit("a\nb\n", "z", "X", true).is_err());
}

#[test]
fn replace_all_unique_still_works() {
    let out = apply_edit("only\n", "only", "once", true).unwrap();
    assert_eq!(out, "once\n");
}

#[test]
fn replace_all_refuses_fuzzy_fallback() {
    // No EXACT match (CRLF differs) but a fuzzy line-match exists. With
    // replace_all we must NOT silently do a single fuzzy replacement and
    // report success — bail so the model re-reads and uses exact text.
    let err = apply_edit("a\r\nb\r\n", "a\nb", "X\nY", true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("replace_all"), "explains the refusal: {err}");
    // Without replace_all the same fuzzy edit still applies (single, unique).
    let out = apply_edit("a\r\nb\r\n", "a\nb", "X\nY", false).unwrap();
    assert!(out.contains('X') && out.contains('Y'), "{out:?}");
}

#[test]
fn confusable_smart_quotes_match_straight_quotes() {
    // The file has smart quotes (pasted from a doc); the model edits with
    // straight quotes. Exact/whitespace strategies miss; the confusable
    // strategy matches and the edit both lands and cleans the glyphs.
    let file = "let s = \u{201C}hello\u{201D};\n";
    let out = apply_edit(file, "let s = \"hello\";", "let s = \"bye\";", false).unwrap();
    assert_eq!(out, "let s = \"bye\";\n");
    assert!(
        !out.contains('\u{201C}'),
        "stray smart quote cleaned: {out:?}"
    );
}

#[test]
fn confusable_em_dash_and_nbsp_match_ascii() {
    // Em dash → hyphen, non-breaking space → space (1:1 char mapping).
    let file = "x\u{00A0}=\u{00A0}a\u{2014}b\n";
    let out = apply_edit(file, "x = a-b", "x = c-d", false).unwrap();
    assert_eq!(out, "x = c-d\n");
}

#[test]
fn confusable_match_still_requires_uniqueness() {
    // Two smart-quote lines that both normalize to `v = "a"`: a straight-
    // quote old_string matches both → ambiguous → the confusable strategy
    // refuses to guess and bails into the not-found help (no exact hit
    // exists because both file lines carry smart quotes).
    let file = "v = \u{201C}a\u{201D}\nv = \u{201C}a\u{201D}\n";
    let err = apply_edit(file, "v = \"a\"", "v = \"b\"", false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("not found"),
        "ambiguous → no silent edit: {err}"
    );
}

#[test]
fn normalize_confusables_is_length_preserving_per_char() {
    // Each mapped glyph becomes exactly one ASCII char (never expands), so
    // char count is preserved — the property splicing relies on.
    let input = "\u{201C}\u{201D}\u{2018}\u{2019}\u{2014}\u{00A0}";
    let out = super::normalize_confusables(input);
    assert_eq!(out, "\"\"''- ");
    assert_eq!(out.chars().count(), input.chars().count());
}

#[test]
fn edit_not_found_help_finds_similar_lines() {
    // The needle has a typo ("funciton" vs "function") — no exact or
    // substring hit, but the similarity fallback should still point at the
    // right line by shared words.
    let text = "fn funciton_add(a, b) {\n    a + b\n}\n";
    let msg = edit_not_found_help(text, "fn function_add(a, b) {");
    assert!(
        msg.contains("funciton_add"),
        "similarity fallback finds the typo'd line: {msg}"
    );
}

#[test]
fn apply_patch_refuses_an_oversized_add() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hi-patch-oversized-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("huge.txt");
    let huge = "x".repeat(crate::read::MAX_READ_FILE_BYTES as usize + 8);
    let patch = format!(
        "*** Begin Patch\n*** Add File: {}\n{huge}\n*** End Patch",
        dest.display()
    );
    let err = plan_multi_patch(&dir, &crate::checkpoint::default_state_root(), &patch).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("refusing to write"), "{msg}");
    assert!(!dest.exists(), "oversized add must not land");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn apply_multi_patch_adds_updates_and_deletes() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hi-patch-test-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("update.txt"), "line1\nline2\nline3\n").unwrap();
    std::fs::write(dir.join("delete.txt"), "bye\n").unwrap();

    // Use absolute paths in the patch so the test doesn't depend on cwd
    // (which races with other async tests that also chdir).
    let upd = dir.join("update.txt");
    let cre = dir.join("created.txt");
    let del = dir.join("delete.txt");
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n@@ line1 @@\n line1\n-line2\n+line2b\n line3\n*** Add File: {}\nnew content\n*** Delete File: {}\n*** End Patch",
        upd.display(),
        cre.display(),
        del.display(),
    );
    let result = apply_multi_patch_at(&dir, &patch).await.unwrap().summary;

    // Update: the `-line2` removal is validated against the original (it
    // must be present), then replaced by `+line2b`. Context lines are
    // preserved; unmentioned lines are kept.
    let updated = std::fs::read_to_string(dir.join("update.txt")).unwrap();
    assert!(updated.contains("line1"), "context kept");
    assert!(updated.contains("line2b"), "added line present");
    assert!(!updated.contains("line2\n"), "removed line dropped");
    assert!(updated.contains("line3"), "trailing context kept");

    // Add: new file written with the given content.
    let created = std::fs::read_to_string(dir.join("created.txt")).unwrap();
    assert_eq!(created, "new content\n");

    // Delete: file removed.
    assert!(!dir.join("delete.txt").exists(), "deleted file is gone");

    // Result summary mentions all three operations.
    assert!(result.contains("updated"), "{result}");
    assert!(result.contains("added"), "{result}");
    assert!(result.contains("deleted"), "{result}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn apply_multi_patch_preserves_trailing_newline_state() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hi-patch-eof-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // A file with NO trailing newline: patching it must NOT add one.
    let no_nl = dir.join("no_nl.txt");
    std::fs::write(&no_nl, "alpha\nbeta").unwrap();
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n-alpha\n+ALPHA\n beta\n*** End Patch",
        no_nl.display(),
    );
    apply_multi_patch_at(&dir, &patch).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(&no_nl).unwrap(),
        "ALPHA\nbeta",
        "no trailing newline is preserved (not silently added)"
    );

    // A file WITH a trailing newline keeps it.
    let with_nl = dir.join("with_nl.txt");
    std::fs::write(&with_nl, "alpha\nbeta\n").unwrap();
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n-alpha\n+ALPHA\n beta\n*** End Patch",
        with_nl.display(),
    );
    apply_multi_patch_at(&dir, &patch).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(&with_nl).unwrap(),
        "ALPHA\nbeta\n",
        "trailing newline preserved"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn apply_multi_patch_preserves_mixed_line_endings() {
    let dir = std::env::temp_dir().join(format!("hi-patch-mixed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("mixed.txt");
    std::fs::write(&file, b"one\r\ntwo\nthree\r\n").unwrap();
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n-one\n+ONE\n two\n-three\n+THREE\n*** End Patch",
        file.display()
    );
    let root = dir.clone();
    apply_multi_patch_at(&root, &patch).await.unwrap();
    assert_eq!(std::fs::read(&file).unwrap(), b"ONE\r\ntwo\nTHREE\r\n");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn apply_multi_patch_rejects_bad_envelope() {
    let dir = std::env::temp_dir().join(format!("hi-patch-envelope-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    assert!(apply_multi_patch_at(&dir, "not a patch").await.is_err());
    assert!(apply_multi_patch_at(&dir, "").await.is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn apply_multi_patch_reports_unknown_directives() {
    // A patch with only unrecognized directives should name them in the
    // error so the model can see what went wrong (e.g. a typo).
    let dir = std::env::temp_dir().join(format!("hi-patch-unknown-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let patch = "*** Begin Patch\n*** UpdateFile: src/a.rs\n-old\n+new\n*** End Patch";
    let err = apply_multi_patch_at(&dir, patch)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("unknown directive"),
        "should mention unknown directive: {err}"
    );
    assert!(
        err.contains("*** UpdateFile:"),
        "should name the offending directive: {err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn apply_multi_patch_rejects_stale_context() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hi-patch-stale-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();

    // The context line "delta" is not in the file — must be rejected.
    let f = dir.join("f.txt");
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n alpha\n delta\n+new\n*** End Patch",
        f.display(),
    );
    let result = apply_multi_patch_at(&dir, &patch).await;
    assert!(result.is_err(), "stale context should be rejected");
    // The file is untouched.
    assert_eq!(
        std::fs::read_to_string(dir.join("f.txt")).unwrap(),
        "alpha\nbeta\ngamma\n"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn later_conflict_leaves_every_patch_target_unchanged() {
    let dir = std::env::temp_dir().join(format!("hi-patch-atomic-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a"), "alpha\n").unwrap();
    std::fs::write(dir.join("b"), "beta\n").unwrap();
    let patch = "*** Begin Patch\n*** Update File: a\n-alpha\n+ALPHA\n*** Update File: b\n-stale\n+BETA\n*** End Patch";
    assert!(apply_multi_patch_at(&dir, patch).await.is_err());
    assert_eq!(std::fs::read_to_string(dir.join("a")).unwrap(), "alpha\n");
    assert_eq!(std::fs::read_to_string(dir.join("b")).unwrap(), "beta\n");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn apply_hunk_patch_preserves_unmentioned_lines() {
    // A patch that only touches the middle of a file must preserve the
    // lines before and after the hunk — not replace the whole file.
    // Context lines are space-prefixed (unified-diff style).
    let orig = vec!["a", "b", "c", "d", "e", "f"];
    let patch = vec![" b", "-c", "+C", " d"];
    let out = apply_hunk_patch(&orig, &patch).unwrap();
    assert_eq!(out, "a\nb\nC\nd\ne\nf\n");
}
