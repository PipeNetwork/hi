use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::Deserialize;

use crate::edit::sh_quote;
use crate::paths::{FileVersion, ReadCache, cache_key};
use crate::{ProcessRunner, ToolOutcome, ToolStatus};

mod discovery;
mod formatting;
mod grep_fallback;
mod resource;

#[cfg(test)]
#[path = "read/cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
#[path = "read/search_tests.rs"]
mod search_tests;

pub(crate) use resource::workspace_path_from_read_arguments;
pub use resource::{ResourceReadRoutingError, route_resource_read};

#[cfg(test)]
use discovery::run_list_sync;
pub(crate) use discovery::{run_glob, run_list};
#[cfg(test)]
use formatting::{format_read, format_read_with_budget};
use formatting::{format_read_for_output, render_read_with_budget};
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
            let rendered = format_read_for_output(&content, offset, limit);
            Ok(crate::ToolOutcome::plain_read(
                rendered.content,
                content.len() as u64,
                rendered.truncated,
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
            let mut truncated = false;
            for (display, offset, limit, body) in reads {
                let body = body?;
                let header = format!("──── {display} ────\n");
                out.push_str(&header);
                let body_budget = remaining_budget
                    .saturating_sub(header.chars().count() + 1)
                    .checked_div(remaining_files.max(1))
                    .unwrap_or(0);
                let rendered = render_read_with_budget(&body, offset, limit, Some(body_budget));
                truncated |= rendered.truncated;
                let formatted = rendered.content;
                source_bytes = source_bytes.saturating_add(body.len() as u64);
                let used = header.chars().count() + formatted.chars().count() + 1;
                out.push_str(&formatted);
                out.push('\n');
                remaining_budget = remaining_budget.saturating_sub(used);
                remaining_files = remaining_files.saturating_sub(1);
            }
            Ok(crate::ToolOutcome::plain_read(out, source_bytes, truncated))
        }
    }
}

/// Read one file as UTF-8 text, using the per-turn cache and bailing clearly
/// on binary files. Shared by the single- and multi-path read paths.
pub(super) async fn read_one(cache: &std::sync::Mutex<ReadCache>, path: &str) -> Result<String> {
    let key = cache_key(std::path::Path::new(path));
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("reading metadata for {path}"))?;
    let version = FileVersion::from_metadata(&metadata);
    let cached = match (cache.lock(), version.as_ref()) {
        (Ok(mut cache), Some(version)) => cache.get_file(&key, version).cloned(),
        // Poisoned lock — treat as a cache miss and re-read the file, rather than
        // turning every subsequent `read` into a panic (as `.unwrap()` did).
        _ => None,
    };
    if let Some(cached) = cached {
        return Ok(cached);
    }
    // Read as bytes first so we can detect binary files and
    // give a clear message instead of an opaque UTF-8 error.
    let bytes = read_regular_file_bytes_async(path).await?;
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
    let after = tokio::fs::metadata(path)
        .await
        .ok()
        .and_then(|metadata| FileVersion::from_metadata(&metadata));
    if version.is_some()
        && version == after
        && let Ok(mut cache) = cache.lock()
    {
        // A background process/editor may mutate a file independently of the
        // agent's explicit cache clears. Never label an in-flight read with a
        // newer stamp than the contents it actually observed.
        cache.insert_file(key, content.clone(), version);
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
            "--hidden".to_string(),
        ];
        if context > 0 {
            cmd_args.push(format!("--context={context}"));
        }
        if let Some(glob) = &args.glob {
            cmd_args.push("--glob".to_string());
            cmd_args.push(glob.clone());
        }
        // Put exclusions after the user glob: rg gives later globs priority.
        // Include useful dotfiles without bringing VCS/dependency internals
        // back into a search when a caller supplies a broad pattern like `**`.
        cmd_args.extend(
            [
                "--glob=!.git",
                "--glob=!.hg",
                "--glob=!.svn",
                "--glob=!.jj",
                "--glob=!**/.cargo-home/**",
                "--glob=!**/.hi/state/cargo-home/**",
            ]
            .map(str::to_owned),
        );
        cmd_args.push("--".to_string());
        cmd_args.push(pattern.clone());
        cmd_args.push(target.clone());
        // Route the fast path through the same bounded process runner as bash
        // and verification. Bound retained output, not matches per file:
        // hundreds of short matches may still fit the configured model budget.
        let owned_runner = process_runner
            .is_none()
            .then(|| ProcessRunner::new(root))
            .transpose()?;
        let runner = process_runner
            .or(owned_runner.as_ref())
            .expect("grep runner is either borrowed or constructed above");
        let output = runner
            .run_program_plain_maybe_timeout("rg", &cmd_args, timeout)
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

/// Read a bounded regular file through one descriptor. Inspecting the opened
/// file closes the stat/read race, and the read cap still holds if it grows.
/// Nonblocking open prevents a FIFO from consuming a blocking worker forever
/// before we can reject its descriptor type.
pub fn read_regular_file_bytes_bounded(path: &std::path::Path, max_bytes: u64) -> Result<Vec<u8>> {
    use std::io::Read;

    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("reading metadata for {}", path.display()))?;
    if !metadata.is_file() {
        bail!(
            "{} is not a regular file; text tools cannot read special files or directories",
            path.display()
        );
    }
    if metadata.len() > max_bytes {
        bail!(
            "{} is too large for text tools ({} bytes; limit {max_bytes}). Use `bash` with a bounded command to inspect or modify it.",
            path.display(),
            metadata.len()
        );
    }
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!(
            "{} grew too large while reading (limit {max_bytes} bytes)",
            path.display()
        );
    }
    Ok(bytes)
}

pub(crate) fn read_regular_file_bytes(path: &std::path::Path) -> Result<Vec<u8>> {
    read_regular_file_bytes_bounded(path, MAX_READ_FILE_BYTES)
}

async fn read_regular_file_bytes_async(path: &str) -> Result<Vec<u8>> {
    let path = std::path::PathBuf::from(path);
    tokio::task::spawn_blocking(move || read_regular_file_bytes(&path))
        .await
        .context("text file reader failed")?
}

/// Read a file as UTF-8 text, bailing with a clear message if it's binary
/// (same heuristic as `read`) or not valid UTF-8. Used by the preserving-edit
/// paths (`edit`/`multi_edit`/`apply_patch`), which write the decoded string
/// back to disk — a lossy decode here would silently replace every invalid
/// byte in the whole file with U+FFFD on the write-back, corrupting e.g.
/// Latin-1 files even on lines the edit never touched.
pub(crate) async fn read_text_file(path: &str) -> Result<String> {
    let bytes = read_regular_file_bytes_async(path).await?;
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
#[path = "read/tests.rs"]
mod tests;
