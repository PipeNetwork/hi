use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use regex::Regex;
use serde::Deserialize;
use std::io::Read;

use crate::condense::truncate;
use crate::edit::sh_quote;
use crate::paths::{ReadCache, cache_key, is_vcs_metadata_dir};
use crate::{ProcessRunner, ToolOutcome, ToolStatus};

const DEFAULT_READ_LIMIT: usize = 2000;
/// Do not materialize arbitrarily large files just to return a bounded page.
/// Model-facing output is much smaller, and callers can use `bash` for binary
/// or genuinely large artifacts when they need byte-level access.
const MAX_READ_FILE_BYTES: u64 = 16 * 1024 * 1024;
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
    let args: ReadArgs = crate::tools::parse(arguments)?;
    // Multi-file mode: read each path and join with a header per file.
    if let Some(paths) = args.paths.as_deref() {
        if paths.is_empty() {
            bail!("`paths` must list at least one path");
        }
        if paths.len() > MAX_MULTI_READ_PATHS {
            bail!("`paths` may contain at most {MAX_MULTI_READ_PATHS} files per call");
        }
        let targets = paths
            .iter()
            .map(|path| Ok((path.clone(), resolve(root, path)?)))
            .collect::<Result<Vec<_>>>()?;
        let reads =
            futures_util::stream::iter(targets.into_iter().map(|(path, target)| async move {
                let body = read_one(cache, &target).await;
                (path, body)
            }))
            .buffered(MULTI_READ_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
        let mut out = String::new();
        for (path, body) in reads {
            let body = body?;
            out.push_str(&format!("──── {path} ────\n"));
            out.push_str(&truncate(&format_read(&body, args.offset, args.limit)));
            out.push('\n');
        }
        return Ok(ToolOutcome::bounded_plain(out));
    }
    // Single-file mode.
    let path = args
        .path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("`read` requires `path` or `paths`"))?;
    let target = resolve(root, path)?;
    let content = read_one(cache, &target).await?;
    Ok(ToolOutcome::plain(truncate(&format_read(
        &content,
        args.offset,
        args.limit,
    ))))
}

/// Read one file as UTF-8 text, using the per-turn cache and bailing clearly
/// on binary files. Shared by the single- and multi-path read paths.
async fn read_one(cache: &std::sync::Mutex<ReadCache>, path: &str) -> Result<String> {
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

/// Run the `list` tool against `arguments` (already-parsed JSON).
pub(crate) async fn run_list(root: &std::path::Path, arguments: &str) -> Result<ToolOutcome> {
    let args: ListArgs = crate::tools::parse(arguments)?;
    let path = args.path.as_deref().unwrap_or(".");
    let target = resolve(root, path)?;
    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || run_list_sync(&root, &target))
        .await
        .context("list worker task failed")?
}

fn run_list_sync(root: &std::path::Path, target: &str) -> Result<ToolOutcome> {
    // Use the `ignore` crate for gitignore-aware directory walking, same
    // semantics as `git ls-files` but without spawning a process.
    let mut out = String::new();
    let mut count = 0u32;
    let walker = ignore::WalkBuilder::new(target)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false) // fall back to all files outside a repo
        .hidden(false)
        .filter_entry(|e| !is_vcs_metadata_dir(e))
        .build();
    for entry in walker {
        let entry = entry.with_context(|| format!("walking {target}"))?;
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let rel = display_path(root, entry.path());
        out.push_str(&rel);
        out.push('\n');
        count += 1;
        if count >= 1000 {
            out.push_str("… (truncated at 1000 entries)\n");
            break;
        }
    }
    let out = if out.is_empty() {
        "(no files found)".to_string()
    } else {
        out
    };
    Ok(ToolOutcome::plain(truncate(&out)))
}

/// Run the `glob` tool against `arguments` (already-parsed JSON).
pub(crate) async fn run_glob(root: &std::path::Path, arguments: &str) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct GlobArgs {
        pattern: String,
        path: Option<String>,
    }
    let args: GlobArgs = crate::tools::parse(arguments)?;
    let path = args.path.as_deref().unwrap_or(".");
    let target = resolve(root, path)?;
    let pattern = args.pattern;
    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || run_glob_sync(&root, &target, &pattern))
        .await
        .context("glob worker task failed")?
}

fn run_glob_sync(root: &std::path::Path, target: &str, pattern: &str) -> Result<ToolOutcome> {
    let mut out = String::new();
    let mut count = 0u32;
    let mut builder = ignore::WalkBuilder::new(target);
    builder
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .hidden(false)
        .filter_entry(|e| !is_vcs_metadata_dir(e));
    let mut override_builder = ignore::overrides::OverrideBuilder::new(target);
    override_builder
        .add(pattern)
        .with_context(|| format!("invalid glob `{pattern}`"))?;
    match override_builder.build() {
        Ok(ov) => {
            let walker = builder.overrides(ov).build();
            for entry in walker {
                let entry = entry.with_context(|| format!("walking {target}"))?;
                if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                    continue;
                }
                let rel = display_path(root, entry.path());
                out.push_str(&rel);
                out.push('\n');
                count += 1;
                if count >= 500 {
                    out.push_str("… (truncated at 500 entries)\n");
                    break;
                }
            }
        }
        Err(e) => bail!("invalid glob `{pattern}`: {e}"),
    }
    let out = if out.is_empty() {
        format!("no files match `{pattern}`")
    } else {
        out
    };
    Ok(ToolOutcome::plain(truncate(&out)))
}

/// Run the `grep` tool against `arguments` (already-parsed JSON).
pub(crate) async fn run_grep(root: &std::path::Path, arguments: &str) -> Result<ToolOutcome> {
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
        let runner = ProcessRunner::new(root)?;
        let output = runner
            .run_program("rg", &cmd_args, std::time::Duration::from_secs(60))
            .await;
        match output {
            Ok(execution) if execution.status == ToolStatus::Succeeded => {
                let text = execution.outcome.stdout_summary;
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
            Ok(execution) => bail!(
                "ripgrep failed for {}: {}",
                target,
                execution.model_content().trim()
            ),
            // A missing rg binary falls through to the hermetic walker. Other
            // launch errors should remain visible instead of being disguised
            // as a slow fallback search.
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

fn run_grep_fallback_sync(
    root: &std::path::Path,
    target: &str,
    pattern: &str,
    glob: Option<&str>,
    context: usize,
) -> Result<ToolOutcome> {
    let re = Regex::new(pattern).context("invalid regex")?;
    let mut builder = ignore::WalkBuilder::new(target);
    builder
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .hidden(false)
        .filter_entry(|e| !is_vcs_metadata_dir(e));
    if let Some(glob) = glob {
        match ignore::overrides::OverrideBuilder::new(target).add(glob) {
            Ok(ovb) => match ovb.build() {
                Ok(ov) => {
                    builder.overrides(ov);
                }
                Err(e) => bail!("invalid glob `{glob}`: {e}"),
            },
            Err(e) => bail!("invalid glob `{glob}`: {e}"),
        }
    }
    let mut out = String::new();
    let mut count = 0u32;
    // Auto-size the match cap from the output budget: stop once `out` approaches
    // `MAX_OUTPUT_CHARS` rather than at a fixed match count. This adapts to the
    // context window — short matches yield more results, long lines fewer —
    // without a config knob. A floor of 50 ensures we always show *some* matches
    // even when lines are very long (truncate will clip the final string).
    let budget = *crate::condense::MAX_OUTPUT_CHARS;
    let walker = builder.build();
    for entry in walker {
        let entry = entry.with_context(|| format!("walking {target}"))?;
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let file_path = entry.path();
        let rel = display_path(root, file_path);
        // Read a bounded byte window before splitting into lines. Calling
        // `BufRead::read_line` directly lets one newline-free record allocate
        // the entire file before the line-count guard can run.
        let file = std::fs::File::open(file_path)
            .with_context(|| format!("opening {} while searching", file_path.display()))?;
        let mut bytes = Vec::new();
        file.take((MAX_GREP_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .with_context(|| format!("reading {} while searching", file_path.display()))?;
        if bytes.len() > MAX_GREP_FILE_BYTES {
            out.push_str(&format!(
                "{rel}: (skipped — file exceeds {MAX_GREP_FILE_BYTES} bytes; install ripgrep for full search)\n"
            ));
            continue;
        }
        // A NUL byte means this isn't text. Avoid UTF-8 work on binary blobs.
        if bytes.contains(&0) {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        // Read lines into a bounded index for context matching. The rg fast
        // path handles larger files without buffering; this fallback only runs
        // when rg isn't installed.
        const MAX_LINES_PER_FILE: usize = 50_000;
        let mut lines: Vec<(usize, &str)> = Vec::new();
        let mut too_large = false;
        for (index, line) in text.lines().enumerate() {
            if index >= MAX_LINES_PER_FILE {
                too_large = true;
                break;
            }
            lines.push((index + 1, line));
        }
        if too_large {
            out.push_str(&format!(
                "{rel}: (skipped — file exceeds {MAX_LINES_PER_FILE} lines; install ripgrep for full search)\n"
            ));
            continue;
        }
        for (idx, (_, line)) in lines.iter().enumerate() {
            if re.is_match(line) {
                let line_no = lines[idx].0;
                if context > 0 {
                    let start = idx.saturating_sub(context);
                    let end = (idx + context + 1).min(lines.len());
                    for (ctx_i, (ctx_no, ctx_line)) in
                        lines.iter().enumerate().take(end).skip(start)
                    {
                        let marker = if ctx_i == idx { ":" } else { "-" };
                        out.push_str(&format!("{rel}{marker}{}: {}\n", ctx_no, ctx_line));
                    }
                    out.push_str("--\n");
                } else {
                    out.push_str(&format!("{rel}:{line_no}: {line}\n"));
                }
                count += 1;
                // Auto-size: stop when we've filled the output budget. The
                // final `truncate` will clip to exactly `budget`, but we stop
                // early so we don't scan needlessly after the cap is reached.
                if out.len() >= budget && count >= 50 {
                    out.push_str("… (truncated — output budget reached)\n");
                    break;
                }
            }
        }
        if out.ends_with("output budget reached)\n") {
            break;
        }
    }
    let out = if out.is_empty() {
        format!("no matches for {pattern}")
    } else {
        out
    };
    Ok(ToolOutcome::plain(truncate(&out)))
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

/// Render a file for the `read` tool: each line prefixed with its 1-based number
/// and a tab (so the model can cite and edit precisely), optionally restricted
/// to `[offset, offset+limit)`. When no limit is provided, return a bounded
/// page. A footer notes when lines were omitted so the model knows to page a
/// large file with `offset`/`limit` rather than assume it saw everything.
pub(crate) fn format_read(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    if content.is_empty() {
        return "(empty file)".to_string();
    }
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = offset.unwrap_or(1).max(1);
    if start > total {
        return format!("(file has {total} line(s); offset {start} is past the end)");
    }
    // Treat zero as the smallest useful page rather than producing the
    // misleading range "lines 1-0" and an empty result.
    let limit = limit.unwrap_or(DEFAULT_READ_LIMIT).max(1);
    let end = start.saturating_add(limit).saturating_sub(1).min(total);
    // Width from the file's total line count (not this page's end) so the gutter
    // is consistent across pages — reading lines 1-240 vs 9900-10000 shouldn't
    // shift the column.
    let width = total.to_string().len().max(4);
    let mut out = String::new();
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let n = start + i;
        out.push_str(&format!("{n:>width$}\t{line}\n"));
    }
    if start > 1 || end < total {
        out.push_str(&format!("… showing lines {start}-{end} of {total}"));
        if end < total {
            out.push_str(&format!(" — read more with offset {}", end + 1));
        }
    }
    out
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
pub(crate) struct ReadArgs {
    /// Path to a single file. Optional if `paths` is given instead.
    #[serde(default)]
    pub path: Option<String>,
    /// Multiple paths to read in one call. Each is returned under a header.
    /// Use this to pull a whole directory of files at once instead of one
    /// call per file.
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    /// 1-based first line to return (default: start of file). Applied to
    /// every file when `paths` is used.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Max number of lines to return per file (default: 2000, i.e. the whole
    /// file for most source files). Page with a smaller `limit` + `offset`.
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub(crate) struct ListArgs {
    #[serde(default)]
    pub path: Option<String>,
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
        run_grep_fallback_sync, run_read,
    };

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
}
