use super::{
    DEFAULT_READ_LIMIT, MAX_GREP_FILE_BYTES, MAX_READ_FILE_BYTES, format_read, is_binary,
    looks_like_numbered_read, read_output_budget, result_char_budget, ripgrep_binary_unavailable,
    run_grep_fallback_sync, run_grep_with_runner_maybe_timeout, run_list_sync, run_read,
};

#[test]
fn bounded_regular_reader_honors_callers_byte_limit() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source.txt");
    std::fs::write(&path, "éé").unwrap();
    assert_eq!(
        super::read_regular_file_bytes_bounded(&path, 4).unwrap(),
        "éé".as_bytes()
    );
    assert!(
        super::read_regular_file_bytes_bounded(&path, 3)
            .unwrap_err()
            .to_string()
            .contains("limit 3")
    );
    std::fs::write(&path, "").unwrap();
    assert!(
        super::read_regular_file_bytes_bounded(&path, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn numbered_read_pages_use_the_read_budget() {
    let page = "   1\t# Solana P2P Marketplace spec\n   2\t## Phase 1\n";
    assert!(looks_like_numbered_read(page));
    assert_eq!(result_char_budget(page), read_output_budget());
    assert_eq!(
        result_char_budget("explore dump\n".repeat(20).as_str()),
        *crate::condense::MAX_OUTPUT_CHARS
    );
}

#[test]
fn read_numbers_lines_and_pages() {
    let body = "alpha\nbravo\ncharlie\ndelta\n";
    // Whole file: every line numbered from 1.
    let all = format_read(body, None, None);
    assert!(all.contains("   1\talpha"), "{all}");
    assert!(all.contains("   4\tdelta"), "{all}");
    // A window keeps absolute line numbers and notes there's more below.
    let win = format_read(body, Some(2), Some(2));
    assert!(
        win.contains("   2\tbravo") && win.contains("   3\tcharlie"),
        "{win}"
    );
    assert!(
        !win.contains("alpha") && !win.contains("delta"),
        "windowed: {win}"
    );
    assert!(
        win.contains("lines 2-3 of 4") && win.contains("offset 4"),
        "footer: {win}"
    );
    assert!(format_read(body, None, Some(0)).contains("alpha"));
    let large = (1..=DEFAULT_READ_LIMIT + 2)
        .map(|n| format!("line {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    let page = format_read(&large, None, None);
    assert!(page.contains("   1\tline 1"), "{page}");
    assert!(
        page.contains(&format!(
            "{DEFAULT_READ_LIMIT:>4}\tline {DEFAULT_READ_LIMIT}"
        )),
        "{page}"
    );
    assert!(
        !page.contains(&format!("line {}", DEFAULT_READ_LIMIT + 1)),
        "{page}"
    );
    assert!(
        page.contains(&format!(
            "lines 1-{DEFAULT_READ_LIMIT} of {}",
            DEFAULT_READ_LIMIT + 2
        )) && page.contains(&format!("offset {}", DEFAULT_READ_LIMIT + 1)),
        "footer: {page}"
    );
    // Empty + past-end are handled.
    assert_eq!(format_read("", None, None), "(empty file)");
    assert!(format_read(body, Some(99), None).contains("past the end"));
}

#[test]
fn bounded_read_reports_the_range_that_was_actually_returned() {
    let body = (1..=100)
        .map(|n| format!("line {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    let page = super::format_read_with_budget(&body, None, None, Some(120));

    assert!(
        page.chars().count() <= 120,
        "{} chars: {page}",
        page.chars().count()
    );
    assert!(
        page.contains("showing lines 1-") && page.contains("of 100"),
        "{page}"
    );
    assert!(page.contains("read more with offset"), "{page}");
    assert!(
        !page.contains("line 100"),
        "page should not claim to contain the tail: {page}"
    );
}

#[test]
fn is_binary_detects_nul_bytes() {
    assert!(!is_binary(b"plain text\n"), "text is not binary");
    assert!(!is_binary(b""), "empty is not binary");
    assert!(is_binary(b"text\x00more"), "NUL → binary");
    // NUL beyond the 8 KB probe window is not detected (same as ripgrep).
    let mut big = vec![b'x'; 9000];
    big.push(0);
    assert!(!is_binary(&big), "NUL past 8 KB probe is not detected");
}

#[tokio::test]
async fn read_rejects_files_that_would_blow_the_cache() {
    let root = std::env::temp_dir().join(format!(
        "hi-read-large-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("large.txt");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(MAX_READ_FILE_BYTES + 1).unwrap();
    let cache = std::sync::Mutex::new(crate::paths::ReadCache::new());
    let error = run_read(&root, &cache, r#"{"path":"large.txt"}"#)
        .await
        .expect_err("large file should be rejected before loading");
    assert!(error.to_string().contains("too large"));
    let edit_err = super::read_text_file(&path.to_string_lossy())
        .await
        .expect_err("edit path must reject the same oversized file");
    assert!(edit_err.to_string().contains("too large"));
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[tokio::test]
async fn text_tools_reject_fifo_without_blocking() {
    use std::os::unix::ffi::OsStrExt;

    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source.pipe");
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: c_path is a valid, NUL-terminated path owned by this test.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);

    let reader_path = path.clone();
    let mut reader =
        tokio::task::spawn_blocking(move || super::read_regular_file_bytes(&reader_path));
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), &mut reader).await;
    if result.is_err() {
        // Release an incorrectly blocking reader before failing, otherwise
        // the Tokio runtime would wait forever for its blocking worker.
        std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        let _ = reader.await;
        panic!("text file reader blocked opening a FIFO");
    }
    let error = result.unwrap().unwrap().unwrap_err();
    assert!(
        error.to_string().contains("not a regular file"),
        "{error:#}"
    );

    let cache = std::sync::Mutex::new(crate::paths::ReadCache::new());
    assert!(
        run_read(root.path(), &cache, r#"{"path":"source.pipe"}"#)
            .await
            .unwrap_err()
            .to_string()
            .contains("not a regular file")
    );
    assert!(
        super::read_text_file(path.to_str().unwrap())
            .await
            .unwrap_err()
            .to_string()
            .contains("not a regular file")
    );
    let state = tempfile::tempdir().unwrap();
    let error = crate::edit::plan_multi_patch(
        root.path(),
        state.path(),
        "*** Begin Patch\n*** Update File: source.pipe\n-old\n+new\n*** End Patch",
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("not a regular file"));
}

#[tokio::test]
async fn multi_read_preserves_requested_order() {
    let root = std::env::temp_dir().join(format!(
        "hi-read-multi-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.txt"), "alpha\n").unwrap();
    std::fs::write(root.join("b.txt"), "bravo\n").unwrap();
    let cache = std::sync::Mutex::new(crate::paths::ReadCache::new());

    let output = run_read(&root, &cache, r#"{"paths":["b.txt","a.txt"],"limit":1}"#)
        .await
        .unwrap()
        .content;
    assert!(
        output.find("──── b.txt ────").unwrap() < output.find("──── a.txt ────").unwrap(),
        "batched reads must retain model-requested order: {output}"
    );
    assert!(
        output.contains("bravo") && output.contains("alpha"),
        "{output}"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn paged_read_reports_truncated_metadata() {
    let root = std::env::temp_dir().join(format!(
        "hi-read-trunc-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let body = (1..=2_000)
        .map(|n| format!("source line {n} with enough text to blow the char budget"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(root.join("big.rs"), &body).unwrap();
    let cache = std::sync::Mutex::new(crate::paths::ReadCache::new());
    let output = run_read(&root, &cache, r#"{"path":"big.rs"}"#)
        .await
        .unwrap();
    assert!(
        crate::read_output_invites_paging(&output.content),
        "expected a paging footer: {}",
        output.content
    );
    match output.truncation {
        crate::TruncationState::Truncated {
            original_bytes,
            retained_bytes,
        } => {
            assert!(
                original_bytes > retained_bytes,
                "original {original_bytes} should exceed retained {retained_bytes}"
            );
        }
        crate::TruncationState::Complete => {
            panic!("budget-clipped read was reported complete")
        }
    }
    let small = run_read(&root, &cache, r#"{"path":"big.rs","limit":5}"#)
        .await
        .unwrap();
    assert!(
        matches!(small.truncation, crate::TruncationState::Truncated { .. }),
        "an explicit short page of a longer file is truncated: {:?}",
        small.truncation
    );
    std::fs::write(root.join("tiny.rs"), "fn ready() {}\n").unwrap();
    let whole = run_read(&root, &cache, r#"{"path":"tiny.rs"}"#)
        .await
        .unwrap();
    assert_eq!(whole.truncation, crate::TruncationState::Complete);
    let typical = (1..=800)
        .map(|n| format!("    pub fn item_{n}() {{}}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(root.join("typical.rs"), typical).unwrap();
    let typical = run_read(&root, &cache, r#"{"path":"typical.rs"}"#)
        .await
        .unwrap();
    assert_eq!(
        typical.truncation,
        crate::TruncationState::Complete,
        "an 800-line source file must fit in one default read"
    );
    assert!(
        !crate::read_output_invites_paging(&typical.content),
        "typical source should not ask the model to page: {}",
        typical.content
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn multi_read_shares_budget_and_preserves_each_file_footer() {
    let root = std::env::temp_dir().join(format!(
        "hi-read-multi-budget-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let first = (1..=1_500)
        .map(|n| format!("first line {n} padded so two files exceed the read budget"))
        .collect::<Vec<_>>()
        .join("\n");
    let second = (1..=1_500)
        .map(|n| format!("second line {n} padded so two files exceed the read budget"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(root.join("first.txt"), first).unwrap();
    std::fs::write(root.join("second.txt"), second).unwrap();
    let cache = std::sync::Mutex::new(crate::paths::ReadCache::new());

    let output = run_read(&root, &cache, r#"{"paths":["first.txt","second.txt"]}"#)
        .await
        .unwrap();

    assert!(
        output.content.chars().count() <= super::read_output_budget(),
        "multi-read exceeded the read budget: {} chars",
        output.content.chars().count()
    );
    assert!(
        matches!(output.truncation, crate::TruncationState::Truncated { .. }),
        "budget-clipped multi-read must report truncated, got {:?}",
        output.truncation
    );
    assert!(output.content.contains("──── first.txt ────"));
    assert!(output.content.contains("──── second.txt ────"));
    assert_eq!(
        output.content.matches("read more with offset").count(),
        2,
        "each file needs an accurate paging footer: {}",
        output.content
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn grep_fallback_searches_off_executor_with_context() {
    let root = std::env::temp_dir().join(format!(
        "hi-grep-fallback-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("main.rs"), "before\nneedle\nafter\n").unwrap();
    std::fs::write(root.join("notes.txt"), "needle in another file\n").unwrap();
    let target = root.to_string_lossy().into_owned();

    let output = run_grep_fallback_sync(&root, &target, "needle", Some("*.rs"), 1)
        .unwrap()
        .content;
    assert!(output.contains("main.rs-1: before"), "{output}");
    assert!(output.contains("main.rs:2: needle"), "{output}");
    assert!(output.contains("main.rs-3: after"), "{output}");
    assert!(!output.contains("notes.txt"), "{output}");
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn repository_discovery_prunes_project_local_cargo_home() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join(".cargo-home/registry/src/demo")).unwrap();
    std::fs::create_dir_all(root.join("nested/.cargo-home/registry/src/demo")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn source_symbol() {}\n").unwrap();
    std::fs::write(
        root.join(".cargo-home/registry/src/demo/lib.rs"),
        "pub fn cached_dependency_symbol() {}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("nested/.cargo-home/registry/src/demo/lib.rs"),
        "pub fn nested_cached_dependency_symbol() {}\n",
    )
    .unwrap();
    let target = root.to_string_lossy().into_owned();

    let listed = run_list_sync(root, &target).unwrap().content;
    assert!(listed.contains("src/lib.rs"), "{listed}");
    assert!(!listed.contains(".cargo-home"), "{listed}");

    let searched = run_grep_fallback_sync(root, &target, "symbol", None, 0)
        .unwrap()
        .content;
    assert!(searched.contains("source_symbol"), "{searched}");
    assert!(!searched.contains("cached_dependency_symbol"), "{searched}");
    assert!(!searched.contains(".cargo-home"), "{searched}");

    let searched = super::run_grep(root, r#"{"pattern":"symbol"}"#)
        .await
        .unwrap()
        .content;
    assert!(searched.contains("source_symbol"), "{searched}");
    assert!(!searched.contains("cached_dependency_symbol"), "{searched}");
    assert!(
        !searched.contains("nested_cached_dependency_symbol"),
        "{searched}"
    );
    assert!(!searched.contains(".cargo-home"), "{searched}");
}

#[tokio::test]
async fn ripgrep_deadline_is_absent_by_default_and_explicit_when_requested() {
    if std::process::Command::new("rg")
        .arg("--version")
        .output()
        .map(|output| !output.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: rg not on PATH");
        return;
    }
    let root = std::env::temp_dir().join(format!(
        "hi-grep-deadline-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("source.txt"), "find-this-symbol\n").unwrap();

    let output =
        run_grep_with_runner_maybe_timeout(&root, None, r#"{"pattern":"find-this-symbol"}"#, None)
            .await
            .expect("default-unlimited ripgrep should complete");
    assert!(output.content.contains("find-this-symbol"), "{output:?}");

    let error = run_grep_with_runner_maybe_timeout(
        &root,
        None,
        r#"{"pattern":"find-this-symbol"}"#,
        Some(std::time::Duration::ZERO),
    )
    .await
    .expect_err("an explicit zero-duration deadline must fire");
    assert!(error.to_string().contains("timed out"), "{error:#}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn grep_fallback_bounds_a_newline_free_file() {
    let root = std::env::temp_dir().join(format!(
        "hi-grep-fallback-large-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("giant.txt"), vec![b'x'; MAX_GREP_FILE_BYTES + 1]).unwrap();
    let target = root.to_string_lossy().into_owned();

    let output = run_grep_fallback_sync(&root, &target, "needle", None, 0)
        .unwrap()
        .content;
    assert!(output.contains("exceeds"), "{output}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn sandboxed_missing_rg_is_treated_as_unavailable() {
    let execution = crate::ProcessExecution {
        status: crate::ToolStatus::Failed,
        outcome: crate::ProcessOutcome {
            exit_code: Some(71),
            stdout_summary: String::new(),
            stderr_summary: "sandbox-exec: execvp() of 'rg' failed: No such file or directory"
                .into(),
            duration_ms: 1,
        },
        truncation: crate::TruncationState::Complete,
    };
    assert!(ripgrep_binary_unavailable(&execution));
}
