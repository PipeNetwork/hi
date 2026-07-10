use std::collections::HashMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;
use tokio::process::Command;

use crate::ToolOutput;
use crate::condense::truncate;
use crate::edit::sh_quote;
use crate::paths::{READ_CACHE, cache_key, is_vcs_metadata_dir, validate_workspace_path};

const DEFAULT_READ_LIMIT: usize = 2000;

/// Run the `read` tool against `arguments` (already-parsed JSON).
///
/// Accepts either a single `path` or a `paths` array. When `paths` is given,
/// every file is read and returned concatenated, each headed by its path —
/// so a model can pull a whole directory of files in one call instead of
/// one call per file. A per-file separator makes the boundary unambiguous.
pub(crate) async fn run_read(arguments: &str) -> Result<ToolOutput> {
    let args: ReadArgs = crate::tools::parse(arguments)?;
    // Multi-file mode: read each path and join with a header per file.
    if let Some(paths) = args.paths.as_deref() {
        if paths.is_empty() {
            bail!("`paths` must list at least one path");
        }
        let mut out = String::new();
        for path in paths {
            validate_workspace_path(path)?;
            let body = read_one(path).await?;
            out.push_str(&format!("──── {path} ────\n"));
            out.push_str(&truncate(&format_read(&body, args.offset, args.limit)));
            out.push('\n');
        }
        return Ok(ToolOutput::plain(out));
    }
    // Single-file mode.
    let path = args
        .path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("`read` requires `path` or `paths`"))?;
    validate_workspace_path(path)?;
    let content = read_one(path).await?;
    Ok(ToolOutput::plain(truncate(&format_read(
        &content,
        args.offset,
        args.limit,
    ))))
}

/// Read one file as UTF-8 text, using the per-turn cache and bailing clearly
/// on binary files. Shared by the single- and multi-path read paths.
async fn read_one(path: &str) -> Result<String> {
    let cached = match READ_CACHE.lock() {
        Ok(mut cache) => cache.get(&cache_key(path)).cloned(),
        // Poisoned lock — treat as a cache miss and re-read the file, rather than
        // turning every subsequent `read` into a panic (as `.unwrap()` did).
        Err(_) => None,
    };
    if let Some(cached) = cached {
        return Ok(cached);
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
    if let Ok(mut cache) = READ_CACHE.lock() {
        cache.insert(cache_key(path), content.clone());
    }
    Ok(content)
}

/// Run the `list` tool against `arguments` (already-parsed JSON).
pub(crate) async fn run_list(arguments: &str) -> Result<ToolOutput> {
    let args: ListArgs = crate::tools::parse(arguments)?;
    let path = args.path.as_deref().unwrap_or(".");
    // Use the `ignore` crate for gitignore-aware directory walking, same
    // semantics as `git ls-files` but without spawning a process.
    let mut out = String::new();
    let mut count = 0u32;
    let walker = ignore::WalkBuilder::new(path)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false) // fall back to all files outside a repo
        .hidden(false)
        .filter_entry(|e| !is_vcs_metadata_dir(e))
        .build();
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let rel = entry.path().to_string_lossy();
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
    Ok(ToolOutput::plain(truncate(&out)))
}

/// Run the `glob` tool against `arguments` (already-parsed JSON).
pub(crate) async fn run_glob(arguments: &str) -> Result<ToolOutput> {
    #[derive(Deserialize)]
    struct GlobArgs {
        pattern: String,
        path: Option<String>,
    }
    let args: GlobArgs = crate::tools::parse(arguments)?;
    let path = args.path.as_deref().unwrap_or(".");
    let mut out = String::new();
    let mut count = 0u32;
    let mut builder = ignore::WalkBuilder::new(path);
    builder
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .hidden(false)
        .filter_entry(|e| !is_vcs_metadata_dir(e));
    let mut override_builder = ignore::overrides::OverrideBuilder::new(path);
    if let Err(e) = override_builder.add(&args.pattern) {
        return Ok(ToolOutput::plain(format!(
            "invalid glob `{}`: {e}",
            args.pattern
        )));
    }
    match override_builder.build() {
        Ok(ov) => {
            let walker = builder.overrides(ov).build();
            for entry in walker {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                    continue;
                }
                let rel = entry.path().to_string_lossy();
                out.push_str(&rel);
                out.push('\n');
                count += 1;
                if count >= 500 {
                    out.push_str("… (truncated at 500 entries)\n");
                    break;
                }
            }
        }
        Err(e) => {
            return Ok(ToolOutput::plain(format!(
                "invalid glob `{}`: {e}",
                args.pattern
            )));
        }
    }
    let out = if out.is_empty() {
        format!("no files match `{}`", args.pattern)
    } else {
        out
    };
    Ok(ToolOutput::plain(truncate(&out)))
}

/// Run the `grep` tool against `arguments` (already-parsed JSON).
pub(crate) async fn run_grep(arguments: &str) -> Result<ToolOutput> {
    let args: GrepArgs = crate::tools::parse(arguments)?;
    let pattern = &args.pattern;
    let path = args.path.as_deref().unwrap_or(".");
    let context = args.context.unwrap_or(0);

    // Fast path: shell out to ripgrep when available — 5-20x faster than
    // the inline walker, with built-in .gitignore support and SIMD.
    if tool_available("rg").await {
        let mut cmd_args = vec![
            "--no-heading".to_string(),
            "--line-number".to_string(),
            "--color=never".to_string(),
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
        cmd_args.push(path.to_string());
        let output = Command::new("rg").args(&cmd_args).output().await;
        match output {
            Ok(o) if o.status.success() || !o.stdout.is_empty() => {
                let text = String::from_utf8_lossy(&o.stdout);
                let out = if text.trim().is_empty() {
                    format!("no matches for {}", args.pattern)
                } else {
                    text.into_owned()
                };
                return Ok(ToolOutput::plain(truncate(&out)));
            }
            Ok(o) if o.status.code() == Some(1) => {
                // rg exit 1 = no matches (not an error)
                return Ok(ToolOutput::plain(format!(
                    "no matches for {}",
                    args.pattern
                )));
            }
            // Fall through to inline walker on other rg errors.
            _ => {}
        }
    }

    // Fallback: inline walker with the `ignore` crate + `regex`.
    let re = match Regex::new(pattern) {
        Ok(re) => re,
        Err(e) => {
            return Ok(ToolOutput::plain(format!("invalid regex: {e}")));
        }
    };
    let mut builder = ignore::WalkBuilder::new(path);
    builder
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .hidden(false)
        .filter_entry(|e| !is_vcs_metadata_dir(e));
    if let Some(glob) = &args.glob {
        match ignore::overrides::OverrideBuilder::new(path).add(glob) {
            Ok(ovb) => match ovb.build() {
                Ok(ov) => {
                    builder.overrides(ov);
                }
                Err(e) => {
                    return Ok(ToolOutput::plain(format!("invalid glob `{glob}`: {e}")));
                }
            },
            Err(e) => {
                return Ok(ToolOutput::plain(format!("invalid glob `{glob}`: {e}")));
            }
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
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let file_path = entry.path();
        let rel = file_path.to_string_lossy();
        // Stream line-by-line so large files don't get fully buffered. Open
        // the file and read lines incrementally; skip binary files (detected
        // from the first chunk) and unreadable files.
        let file = match tokio::fs::File::open(file_path).await {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut reader = tokio::io::BufReader::new(file);
        use tokio::io::AsyncBufReadExt;
        let mut lines: Vec<(usize, String)> = Vec::new();
        let mut binary = false;
        // Read lines into a buffer for context matching. We need random access
        // for context lines, so we collect the file's lines — but cap the count
        // so a huge file with no matches can't exhaust memory. The rg fast path
        // (above) handles large files without buffering; this fallback only runs
        // when rg isn't installed, and a file past the cap is skipped with a note
        // rather than scanned.
        const MAX_LINES_PER_FILE: usize = 50_000;
        let mut line_no = 0usize;
        let mut too_large = false;
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    // Strip the trailing newline (read_line includes it).
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    // Binary detection: a NUL byte in the first 8 KB means this
                    // isn't text — skip the whole file (same heuristic as `read`).
                    if line.contains('\0') {
                        binary = true;
                        break;
                    }
                    line_no += 1;
                    if line_no > MAX_LINES_PER_FILE {
                        too_large = true;
                        break;
                    }
                    lines.push((line_no, line));
                }
                Err(_) => break,
            }
        }
        if binary {
            continue;
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
        format!("no matches for {}", args.pattern)
    } else {
        out
    };
    Ok(ToolOutput::plain(truncate(&out)))
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
    let limit = limit.unwrap_or(DEFAULT_READ_LIMIT);
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

/// Best-effort: run a file-scoped formatter if one is installed for this file
/// type, then return the file's final content (for the diff shown to the user).
/// Never fails the edit — a missing formatter, or a formatter that errors on
/// not-yet-valid code, just leaves the file exactly as written.
pub(crate) async fn maybe_format(path: &str, written: String) -> String {
    // Opt-in: formatters churn unrelated lines in repos that aren't
    // formatter-clean. Disabled by default; set `HI_FORMAT=1` to enable.
    if std::env::var_os("HI_FORMAT").is_none() {
        return written;
    }
    let Some((probe, command)) = formatter_for(path) else {
        return written;
    };
    if !tool_available(probe).await {
        return written;
    }
    let _ = crate::tools::run_bash(&format!("{command} {}", sh_quote(path))).await;
    tokio::fs::read_to_string(path).await.unwrap_or(written)
}

/// The (probe binary, command prefix) of a file-scoped formatter for `path`'s
/// extension, if we support one. The command is run as `<prefix> <file>`.
pub(crate) fn formatter_for(path: &str) -> Option<(&'static str, &'static str)> {
    match Path::new(path).extension()?.to_str()? {
        "rs" => Some(("rustfmt", "rustfmt")),
        "go" => Some(("gofmt", "gofmt -w")),
        "py" => Some(("ruff", "ruff format -q")),
        "js" | "jsx" | "ts" | "tsx" | "json" | "css" | "scss" | "md" | "html" | "yaml" | "yml" => {
            Some(("prettier", "prettier --write --log-level warn"))
        }
        _ => None,
    }
}

/// Cached results of `tool_available` probes — the answer never changes within
/// a session, so we avoid a fork+exec per edit.
static TOOL_AVAILABLE_CACHE: LazyLock<Mutex<HashMap<String, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Whether `prog` is on PATH (so we only invoke formatters that exist).
/// Results are cached per-session: the probe is a fork+exec that takes
/// ~5-20ms, and it's called on every write/edit. Cached after first call.
pub(crate) async fn tool_available(prog: &str) -> bool {
    if let Some(&result) = TOOL_AVAILABLE_CACHE.lock().unwrap().get(prog) {
        return result;
    }
    let result = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {}", sh_quote(prog)))
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);
    TOOL_AVAILABLE_CACHE
        .lock()
        .unwrap()
        .insert(prog.to_string(), result);
    result
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
    use super::{DEFAULT_READ_LIMIT, format_read, is_binary};

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
}
