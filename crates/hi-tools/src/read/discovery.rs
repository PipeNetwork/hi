//! Gitignore-aware repository listing and glob discovery.

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::ToolOutcome;
use crate::condense::truncate;
use crate::paths::is_vcs_metadata_dir;

use super::{display_path, resolve};

#[derive(Deserialize)]
struct ListArgs {
    #[serde(default)]
    path: Option<String>,
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

/// Keep generated dependency caches out of repository discovery even when a
/// bootstrap project has not added them to `.gitignore` yet.
pub(super) fn is_searchable_entry(entry: &ignore::DirEntry) -> bool {
    !(is_vcs_metadata_dir(entry)
        || entry.file_type().is_some_and(|kind| kind.is_dir())
            && entry.file_name().to_str() == Some(".cargo-home"))
}

pub(super) fn run_list_sync(root: &std::path::Path, target: &str) -> Result<ToolOutcome> {
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
        .filter_entry(is_searchable_entry)
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
        .filter_entry(is_searchable_entry);
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
