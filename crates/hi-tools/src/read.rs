use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::Deserialize;

use crate::edit::sh_quote;
use crate::paths::{ReadCache, cache_key};
use crate::{ProcessRunner, ToolOutcome, ToolStatus};

mod discovery;
mod formatting;
mod grep_fallback;
mod resource;

pub(crate) use resource::workspace_path_from_read_arguments;
pub use resource::{ResourceReadRoutingError, route_resource_read};

#[cfg(test)]
use discovery::run_list_sync;
pub(crate) use discovery::{run_glob, run_list};
#[cfg(test)]
use formatting::format_read;
use formatting::{format_read_for_output, format_read_with_budget};
use grep_fallback::{ripgrep_binary_unavailable, run_grep_fallback_sync};

const DEFAULT_READ_LIMIT: usize = 2000;

/// Dedicated `read` budget. The shared tool-result cap (~5k) is right for
/// grep/bash noise but turns a documented 2,000-line default page into ~113
/// lines of Rust and forces DeepSeek to page a typical source file 6–8 times.
/// 64k is enough for an ~800-line file with line-number gutters.
const DEFAULT_READ_OUTPUT_CHARS: usize = 64_000;
const MIN_READ_OUTPUT_CHARS: usize = 1_000;
const MAX_READ_OUTPUT_CHARS: usize = 200_000;

pub(crate) fn read_output_budget() -> usize {
    std::env::var("HI_READ_RESULT_CHARS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&n| (MIN_READ_OUTPUT_CHARS..=MAX_READ_OUTPUT_CHARS).contains(&n))
        .unwrap_or(DEFAULT_READ_OUTPUT_CHARS)
}

/// True when a `read` result told the model to request another page.
///
/// The footer is the public paging contract (`read more with offset N`).
/// Callers use this to tell a complete file from a budget- or limit-clipped
/// one — `ToolOutcome::truncation` used to stay `Complete` because the
/// renderer already stays under the budget.
pub fn read_output_invites_paging(output: &str) -> bool {
    output.contains("read more with offset")
}

/// True when `content` looks like a numbered `read` page (`   12\t…`).
///
/// The shared ~5k tool-result cap is right for bash/grep noise, but applying
/// it again to a completed `read` page head-and-tails the middle of a
/// spec-sized file. Callers then skip rereads because the original page had
/// no paging footer — DeepSeek Flash hit this on a 14k SPEC.md.
pub(crate) fn looks_like_numbered_read(content: &str) -> bool {
    content.lines().next().is_some_and(|line| {
        line.split_once('\t')
            .is_some_and(|(number, _)| number.trim().parse::<u32>().is_ok())
    })
}

/// Character budget for a model-facing tool result.
///
/// Numbered `read` pages (and pages that invite further paging) keep the
/// dedicated read budget. Everything else uses the shared ~5k cap.
pub(crate) fn result_char_budget(content: &str) -> usize {
    if read_output_invites_paging(content) || looks_like_numbered_read(content) {
        read_output_budget()
    } else {
        *crate::condense::MAX_OUTPUT_CHARS
    }
}

/// Do not materialize arbitrarily large files just to return a bounded page.
/// Model-facing output is much smaller, and callers can use `bash` for binary
/// or genuinely large artifacts when they need byte-level access.
pub(crate) const MAX_READ_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Refuse an after-image that could never be re-read or edited in place.
pub(crate) fn refuse_oversized_text(display_path: &str, len: usize) -> Result<()> {
    if len as u64 > MAX_READ_FILE_BYTES {
        bail!(
            "refusing to write `{display_path}` ({} bytes) — files larger than {} bytes cannot be re-read or edited in place",
            len,
            MAX_READ_FILE_BYTES
        );
    }
    Ok(())
}
/// A multi-file read is a convenience for a small related set, not a way to
/// turn one tool call into an unbounded workspace dump.
const MAX_MULTI_READ_PATHS: usize = 32;
/// Keep a batched read responsive without creating one filesystem future per
/// model-supplied path. `buffered` preserves the request order in the output.
const MULTI_READ_CONCURRENCY: usize = 4;
/// Keep context expansion bounded in both the ripgrep and hermetic fallback
/// search paths. Without this, a model-supplied `context` of `usize::MAX`
/// could make the fallback materialize almost an entire 50k-line file around a
/// single match before the final output truncation runs.
const MAX_GREP_CONTEXT_LINES: usize = 100;
/// The hermetic grep fallback is only used when `rg` is unavailable, but it
/// still must not materialize an arbitrarily large one-line file.
const MAX_GREP_FILE_BYTES: usize = 16 * 1024 * 1024;

/// Run the `read` tool against `arguments` (already-parsed JSON).
///
/// Accepts either a single `path` or a `paths` array. When `paths` is given,
/// every file is read and returned concatenated, each headed by its path —
/// so a model can pull a whole directory of files in one call instead of
/// one call per file. A per-file separator makes the boundary unambiguous.
pub(crate) async fn run_read(
    root: &std::path::Path,
    cache: &std::sync::Mutex<ReadCache>,
    arguments: &str,
) -> Result<ToolOutcome> {
    run_read_with_mcp(root, cache, None, arguments).await
}

pub(crate) async fn run_read_with_mcp(
    root: &std::path::Path,
    cache: &std::sync::Mutex<ReadCache>,
    mcp: Option<&dyn crate::McpBackend>,
    arguments: &str,
) -> Result<ToolOutcome> {
    match resource::parse_and_route_read(root, cache, mcp, arguments).await? {
        resource::RoutedReadPlan::Single(read) => {
            let resource::RoutedRead {
                source,
                offset,
                limit,
                ..
            } = read;
            let content = source.read(cache).await?;
            Ok(crate::ToolOutcome::plain_read(
                format_read_for_output(&content, offset, limit),
                content.len() as u64,
            ))
        }
        resource::RoutedReadPlan::Multiple(reads) => {
            let reads = futures_util::stream::iter(reads.into_iter().map(|read| async move {
                let resource::RoutedRead {
                    display,
                    source,
                    offset,
                    limit,
                } = read;
                let body = source.read(cache).await;
                (display, offset, limit, body)
            }))
            .buffered(MULTI_READ_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
            let mut out = String::new();
            let mut remaining_budget = read_output_budget();
            let mut remaining_files = reads.len();
            let mut source_bytes = 0u64;
            for (display, offset, limit, body) in reads {
                let body = body?;
                let header = format!("──── {display} ────\n");
                out.push_str(&header);
                let body_budget = remaining_budget
                    .saturating_sub(header.chars().count() + 1)
                    .checked_div(remaining_files.max(1))
                    .unwrap_or(0);
                let formatted = format_read_with_budget(&body, offset, limit, Some(body_budget));
                source_bytes = source_bytes.saturating_add(body.len() as u64);
                let used = header.chars().count() + formatted.chars().count() + 1;
                out.push_str(&formatted);
                out.push('\n');
                remaining_budget = remaining_budget.saturating_sub(used);
                remaining_files = remaining_files.saturating_sub(1);
            }
            Ok(crate::ToolOutcome::plain_read(out, source_bytes))
        }
    }
}

/// Read one file as UTF-8 text, using the per-turn cache and bailing clearly
/// on binary files. Shared by the single- and multi-path read paths.
pub(super) async fn read_one(cache: &std::sync::Mutex<ReadCache>, path: &str) -> Result<String> {
    let cached = match cache.lock() {
        Ok(mut cache) => cache.get(&cache_key(std::path::Path::new(path))).cloned(),
        // Poisoned lock — treat as a cache miss and re-read the file, rather than
        // turning every subsequent `read` into a panic (as `.unwrap()` did).
        Err(_) => None,
    };
    if let Some(cached) = cached {
        return Ok(cached);
    }
    let size = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("reading metadata for {path}"))?
        .len();
    if size > MAX_READ_FILE_BYTES {
        bail!(
            "{path} is too large to load into the read cache ({size} bytes; limit {MAX_READ_FILE_BYTES}). Use `bash` with a bounded command such as `sed` or `head` to inspect it."
        );
    }
    // Read as bytes first so we can detect binary files and
    // give a clear message instead of an opaque UTF-8 error.
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading {path}"))?;
    if is_binary(&bytes) {
        bail!(
            "{path} is a binary file ({} bytes) — the `read` tool is for text. \
             Use `bash` to inspect it (e.g. `file {}`, `xxd {} | head`).",
            bytes.len(),
            sh_quote(path),
            sh_quote(path)
        );
    }
    let content = String::from_utf8_lossy(&bytes).into_owned();
    if let Ok(mut cache) = cache.lock() {
        cache.insert(cache_key(std::path::Path::new(path)), content.clone());
    }
    Ok(content)
}

/// Run the `grep` tool against `arguments` (already-parsed JSON).
#[cfg(test)]
pub(crate) async fn run_grep(root: &std::path::Path, arguments: &str) -> Result<ToolOutcome> {
    run_grep_with_runner_maybe_timeout(root, None, arguments, None).await
}

/// Like [`run_grep`] with a caller-owned process runner for the ripgrep fast
/// path. The fallback remains entirely in-process.
pub(crate) async fn run_grep_with_runner(
    root: &std::path::Path,
    process_runner: Option<&crate::ProcessRunner>,
    arguments: &str,
) -> Result<ToolOutcome> {
    run_grep_with_runner_maybe_timeout(root, process_runner, arguments, None).await
}

async fn run_grep_with_runner_maybe_timeout(
    root: &std::path::Path,
    process_runner: Option<&crate::ProcessRunner>,
    arguments: &str,
    timeout: Option<std::time::Duration>,
) -> Result<ToolOutcome> {
    let args: GrepArgs = crate::tools::parse(arguments)?;
    let pattern = &args.pattern;
    let path = args.path.as_deref().unwrap_or(".");
    let target = resolve(root, path)?;
    let context = args.context.unwrap_or(0).min(MAX_GREP_CONTEXT_LINES);

    // Fast path: try ripgrep directly — 5-20x faster than the inline walker,
    // with built-in .gitignore support and SIMD. A missing executable falls
    // through to the hermetic inline implementation; other launch failures are
    // surfaced instead of being cached in process-global state.
    {
        let mut cmd_args = vec![
            "--no-heading".to_string(),
            "--line-number".to_string(),
            "--color=never".to_string(),
            "--no-config".to_string(),
            "--max-count=200".to_string(),
            // Never search VCS metadata, even if the user's ripgrep
            // config enables --hidden (which would otherwise descend
            // into .git and leak repository internals to the model).
            "--glob=!.git".to_string(),
            "--glob=!.hg".to_string(),
            "--glob=!.svn".to_string(),
            "--glob=!.jj".to_string(),
            // A project-local Cargo home is a downloaded dependency cache,
            // not repository source. Searching it polluted context and made
            // affected-package verification target registry crates.
            "--glob=!**/.cargo-home/**".to_string(),
        ];
        if context > 0 {
            cmd_args.push(format!("--context={context}"));
        }
        if let Some(glob) = &args.glob {
            cmd_args.push("--glob".to_string());
            cmd_args.push(glob.clone());
        }
        cmd_args.push("--".to_string());
        cmd_args.push(pattern.clone());
        cmd_args.push(target.clone());
        // Route the fast path through the same bounded process runner as bash
        // and verification. `rg --max-count` caps matches per file, not the
        // total output, so `Command::output()` could still retain gigabytes
        // before the final model-content truncation.
        let owned_runner = process_runner
            .is_none()
            .then(|| ProcessRunner::new(root))
            .transpose()?;
        let runner = process_runner
            .or(owned_runner.as_ref())
            .expect("grep runner is either borrowed or constructed above");
        let output = runner
            .run_program_maybe_timeout("rg", &cmd_args, timeout)
            .await;
        match output {
            Ok(execution) if execution.status == ToolStatus::Succeeded => {
                let text = execution.model_outcome().stdout_summary;
                let out = if text.trim().is_empty() {
                    format!("no matches for {}", args.pattern)
                } else {
                    text
                };
                return Ok(ToolOutcome::bounded_plain(out));
            }
            Ok(execution) if execution.outcome.exit_code == Some(1) => {
                // rg exit 1 = no matches (not an error)
                return Ok(ToolOutcome::plain(format!(
                    "no matches for {}",
                    args.pattern
                )));
            }
            Ok(execution) if ripgrep_binary_unavailable(&execution) => {}
            Ok(execution) => bail!(
                "ripgrep failed for {}: {}",
                target,
                execution.model_content().trim()
            ),
            // A missing rg binary falls through to the hermetic walker. Other
            // launch errors should remain visible instead of being disguised
            // as a slow fallback search. Sandboxed `rg` that fails execvp
            // (relative name, exit 71) is the same as missing — fall through.
            Err(error)
                if error.chain().any(|cause| {
                    cause
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
                }) => {}
            Err(error) => return Err(error).context("starting ripgrep"),
        }
    }

    // Fallback: run the entire ignore-aware walk and file scan on a blocking
    // worker. The normal `rg` path is already a child process, but when rg is
    // absent this path must not walk a large repository on the async executor.
    let root = root.to_path_buf();
    let target = target.clone();
    let pattern = pattern.clone();
    let glob = args.glob.clone();
    tokio::task::spawn_blocking(move || {
        run_grep_fallback_sync(&root, &target, &pattern, glob.as_deref(), context)
    })
    .await
    .context("grep fallback worker failed")?
}

fn resolve(root: &std::path::Path, requested: &str) -> Result<String> {
    Ok(
        crate::transaction::resolve_workspace_target(root, std::path::Path::new(requested))?
            .to_string_lossy()
            .into_owned(),
    )
}

fn display_path(root: &std::path::Path, path: &std::path::Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Heuristic: does `bytes` look like a binary file? A NUL byte in the first 8 KB
/// is the classic signal (ripgrep uses the same heuristic). Empty files are not
/// binary. This lets `grep` and `read` skip/guard against non-text files instead
/// of failing opaquely on `read_to_string`.
pub(crate) fn is_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let probe = &bytes[..bytes.len().min(8192)];
    probe.contains(&0)
}

/// Read a file as UTF-8 text, bailing with a clear message if it's binary
/// (same heuristic as `read`) or not valid UTF-8. Used by the preserving-edit
/// paths (`edit`/`multi_edit`/`apply_patch`), which write the decoded string
/// back to disk — a lossy decode here would silently replace every invalid
/// byte in the whole file with U+FFFD on the write-back, corrupting e.g.
/// Latin-1 files even on lines the edit never touched.
pub(crate) async fn read_text_file(path: &str) -> Result<String> {
    let size = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("reading metadata for {path}"))?
        .len();
    if size > MAX_READ_FILE_BYTES {
        bail!(
            "{path} is too large to edit in place ({size} bytes; limit {MAX_READ_FILE_BYTES}). Use `bash` with a bounded command such as `sed` to modify it."
        );
    }
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading {path}"))?;
    if is_binary(&bytes) {
        bail!(
            "{path} is a binary file ({} bytes) — the `edit`/`multi_edit` tools are for text. \
             Use `bash` to inspect or modify it.",
            bytes.len()
        );
    }
    String::from_utf8(bytes).map_err(|e| {
        anyhow::anyhow!(
            "{path} is not valid UTF-8 (first invalid byte at offset {}) — editing it in place \
             would corrupt its encoding. Use `bash` (e.g. sed/iconv) to modify it.",
            e.utf8_error().valid_up_to()
        )
    })
}

#[derive(Deserialize)]
pub(crate) struct GrepArgs {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    /// Lines of context to show around each match (default: 0).
    #[serde(default)]
    pub context: Option<usize>,
    /// File name glob to filter (e.g. `*.rs`). Only files whose name matches
    /// are searched.
    #[serde(default)]
    pub glob: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_READ_LIMIT, MAX_GREP_FILE_BYTES, MAX_READ_FILE_BYTES, format_read, is_binary,
        looks_like_numbered_read, read_output_budget, result_char_budget,
        ripgrep_binary_unavailable, run_grep_fallback_sync, run_grep_with_runner_maybe_timeout,
        run_list_sync, run_read,
    };

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

        let output = run_grep_with_runner_maybe_timeout(
            &root,
            None,
            r#"{"pattern":"find-this-symbol"}"#,
            None,
        )
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
}
