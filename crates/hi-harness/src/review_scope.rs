//! What code `/review` covers when the user named no scope, and how a scope
//! is cut into turns.
//!
//! With a plan/spec one audit turn covers the whole tree. Without one the
//! audit is defects-only, and a single turn cannot read a large tree (a live
//! run on this repo read six of 1175 files against a README, spent 106k
//! tokens, and reported "finding: none"), so:
//!
//! * a large workspace narrows to recent work: the uncommitted changes,
//!   else the files of the last commit ([`crate::review_git`]), split into
//!   turns of `chunk_files` files when there are more;
//! * a small workspace, an explicit directory, or `all` is cut into chunks
//!   of about `chunk_files` files: one per top-level directory (or per crate
//!   under a container such as `crates/`), big directories split, small
//!   ones packed together, ordered by recent git churn.
//!
//! `[review] exclude` applies to every one of these. Only when git has
//! nothing recent (or all of it is excluded) does the command refuse,
//! saying what to pass instead.

use std::path::Path;

use crate::review::{InputKind, ReviewInputs};
use crate::review_git::{GitEmpty, churn_by_path, git_scope};
use crate::review_settings::ReviewSettings;

/// Workspaces with more non-ignored files than this are not audited whole
/// without a plan/spec: `/review` narrows to recent work instead.
pub const LARGE_WORKSPACE_FILES: usize = 200;

/// Chunks named in the start notice before `… (+N more)`.
const CHUNKS_LISTED: usize = 8;

/// Separator between the directories of a packed chunk label.
const LABEL_JOIN: &str = ", ";

/// One prospective audit turn while a chunk plan is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Unit {
    pub label: String,
    /// Workspace-relative files, sorted.
    pub files: Vec<String>,
    /// Files touched by recent commits, for ordering.
    pub churn: usize,
}

/// Decide what code the audit covers, given the discovered `inputs` and
/// whether `all` was asked for. Sets the git scope or the chunk list on
/// `inputs`, drops a README fallback when the audit is defects-only, and
/// writes the notice; `Err` is the user-facing refusal when there is
/// nothing to audit against and nothing recent to audit.
pub fn resolve_target(
    root: &Path,
    inputs: &mut ReviewInputs,
    all: bool,
    settings: &ReviewSettings,
) -> Result<(), String> {
    inputs.exclude = settings.exclude.clone();
    let budget = settings.chunk_files();
    if all {
        // `all .` is `all`: the whole workspace, own top-level files included.
        let whole = inputs.scope.is_empty() || inputs.scope.iter().any(|dir| dir == ".");
        let dirs = if whole {
            inputs.scope.clear();
            top_level_dirs(root, &settings.exclude)
        } else {
            std::mem::take(&mut inputs.scope)
        };
        let chunks = chunk_plan(root, &dirs, whole, budget, &settings.exclude);
        if chunks.is_empty() {
            return Err("spec review: the workspace has no files to audit".into());
        }
        let where_ = if whole {
            "the whole workspace".to_string()
        } else {
            dirs.join(", ")
        };
        let what = if inputs.has_spec() {
            format!("auditing {} against {where_}", inputs.summary())
        } else {
            drop_readme(inputs);
            format!("no plan.md or spec.md found; auditing {where_} for defects")
        };
        inputs.notice = Some(chunk_notice(&what, &chunks));
        inputs.chunks = chunks;
        return Ok(());
    }
    if inputs.has_spec() {
        return Ok(());
    }
    if !inputs.scope.is_empty() {
        // Defects-only in named directories: cut into turns when they hold
        // more than one turn's worth. `.` is the whole workspace, as for `all`.
        let whole = inputs.scope.iter().any(|dir| dir == ".");
        let dirs = if whole {
            top_level_dirs(root, &settings.exclude)
        } else {
            inputs.scope.clone()
        };
        let chunks = chunk_plan(root, &dirs, whole, budget, &settings.exclude);
        if chunks.len() > 1 {
            drop_readme(inputs);
            inputs.scope.clear();
            let where_ = if whole {
                "the whole workspace".to_string()
            } else {
                dirs.join(", ")
            };
            inputs.notice = Some(chunk_notice(
                &format!("no plan.md or spec.md found; auditing {where_} for defects"),
                &chunks,
            ));
            inputs.chunks = chunks;
        }
        return Ok(());
    }
    if !workspace_exceeds(root, LARGE_WORKSPACE_FILES) {
        // Small workspace, nothing to compare against: whole tree, in turns
        // of about `budget` files when it is more than one turn.
        let dirs = top_level_dirs(root, &settings.exclude);
        let chunks = chunk_plan(root, &dirs, true, budget, &settings.exclude);
        if chunks.len() > 1 {
            drop_readme(inputs);
            inputs.notice = Some(chunk_notice(
                "no plan.md or spec.md found; auditing the workspace for defects",
                &chunks,
            ));
            inputs.chunks = chunks;
        }
        return Ok(());
    }
    match git_scope(root) {
        Ok(mut scope) => {
            scope
                .files
                .retain(|path| !excluded(path, &settings.exclude));
            scope
                .numstat
                .retain(|stat| !excluded(&stat.path, &settings.exclude));
            if scope.files.is_empty() {
                return Err(refusal(
                    "every recently changed file is under `[review] exclude`",
                ));
            }
            drop_readme(inputs);
            let chunks = scope.group(root, budget);
            let mut notice = format!(
                "no plan.md or spec.md found; auditing recent work for defects: {}",
                scope.summary()
            );
            if !chunks.is_empty() {
                notice.push_str(&format!(
                    " in {} turns of at most {budget} files: {}",
                    chunks.len(),
                    list_chunks(&chunks)
                ));
            }
            notice.push_str(" · `/review audit all` audits the whole repo in chunks");
            inputs.notice = Some(notice);
            inputs.git = Some(scope);
            inputs.chunks = chunks;
            Ok(())
        }
        Err(empty) => Err(refusal(match empty {
            GitEmpty::NotARepo => "this is not a git repository",
            GitEmpty::Nothing => {
                "git has no uncommitted changes and no commit whose files are still in the tree"
            }
        })),
    }
}

fn drop_readme(inputs: &mut ReviewInputs) {
    inputs.files.retain(|input| input.kind != InputKind::Readme);
}

fn chunk_notice(what: &str, chunks: &[String]) -> String {
    format!(
        "{what} in {} chunks, one turn each: {}",
        chunks.len(),
        list_chunks(chunks)
    )
}

fn list_chunks(chunks: &[String]) -> String {
    let mut text = chunks
        .iter()
        .take(CHUNKS_LISTED)
        .map(|chunk| chunk_label(chunk))
        .collect::<Vec<_>>()
        .join("; ");
    if chunks.len() > CHUNKS_LISTED {
        text.push_str(&format!(" (+{} more)", chunks.len() - CHUNKS_LISTED));
    }
    text
}

fn refusal(why: &str) -> String {
    format!(
        "spec review: no plan.md or spec.md found, the workspace has more than {LARGE_WORKSPACE_FILES} files, and {why}, \
so there is no recent work to audit. `/review audit all` audits the whole repo in chunks (one turn per top-level directory), \
`/review audit <dir>` one directory, or write plan.md (a `- [ ]` checklist of what should exist) for a coverage audit."
    )
}

/// The parts of a chunk label: one directory, or several packed together.
/// A part is `dir`, `dir (i/k)` for the i-th of k slices of a big
/// directory, `dir/.` for a directory's own files without its
/// subdirectories, or `.` for the workspace's own top-level files.
pub fn chunk_parts(chunk: &str) -> Vec<&str> {
    chunk
        .split(LABEL_JOIN)
        .filter(|part| !part.is_empty())
        .collect()
}

/// Human form of a chunk label: `.` is the chunk for the workspace's own
/// top-level files.
pub fn chunk_label(chunk: &str) -> String {
    chunk_parts(chunk)
        .into_iter()
        .map(|part| match part {
            "." => "top-level files".to_string(),
            part => match part.strip_suffix("/.") {
                Some(dir) => format!("{dir} (own files)"),
                None => part.to_string(),
            },
        })
        .collect::<Vec<_>>()
        .join(LABEL_JOIN)
}

/// The files a chunk label names, derived from the tree the same way the
/// plan was, so a label is all the drive needs to persist.
pub fn chunk_files(root: &Path, chunk: &str, exclude: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for part in chunk_parts(chunk) {
        let (name, slice) = match split_slice_label(part) {
            Some((name, slice)) => (name, Some(slice)),
            None => (part, None),
        };
        let all = if name == "." {
            own_files(root, ".", exclude)
        } else if let Some(dir) = name.strip_suffix("/.") {
            own_files(root, dir, exclude)
        } else {
            dir_files(root, name, exclude)
        };
        let files: Vec<String> = match slice {
            Some((index, count)) => {
                let size = all.len().div_ceil(count.max(1)).max(1);
                all.into_iter()
                    .skip(index.saturating_sub(1) * size)
                    .take(size)
                    .collect()
            }
            None => all,
        };
        for file in files {
            if !out.contains(&file) {
                out.push(file);
            }
        }
    }
    out
}

/// `crates/x (2/5)` -> `("crates/x", (2, 5))`.
fn split_slice_label(part: &str) -> Option<(&str, (usize, usize))> {
    let (dir, rest) = part.rsplit_once(" (")?;
    let body = rest.strip_suffix(')')?;
    let (index, count) = body.split_once('/')?;
    Some((dir, (index.parse().ok()?, count.parse().ok()?)))
}

/// Chunks for `dirs` (and, with `root_files`, the workspace's own top-level
/// files as a `.` unit): directories within `budget` files are one unit
/// each, bigger ones with subdirectories descend (their own files become a
/// `dir/.` unit), bigger ones without are sliced; then small units are
/// packed together up to the budget and the chunks are ordered by recent
/// git churn. Labels only; [`chunk_files`] derives the files again.
pub fn chunk_plan(
    root: &Path,
    dirs: &[String],
    root_files: bool,
    budget: usize,
    exclude: &[String],
) -> Vec<String> {
    let mut units = Vec::new();
    for dir in dirs {
        let dir = dir.trim_end_matches('/');
        if dir.is_empty() || dir == "." {
            continue;
        }
        expand(root, dir, budget, exclude, &mut units);
    }
    if root_files {
        let files = own_files(root, ".", exclude);
        if !files.is_empty() {
            units.push(Unit {
                label: ".".into(),
                files,
                churn: 0,
            });
        }
    }
    let mut packed = pack_units(units, budget);
    let churn = churn_by_path(root);
    for unit in &mut packed {
        unit.churn = unit.files.iter().filter_map(|file| churn.get(file)).sum();
    }
    // Stable: chunks nobody touched keep directory order.
    packed.sort_by_key(|unit| std::cmp::Reverse(unit.churn));
    packed.into_iter().map(|unit| unit.label).collect()
}

fn expand(root: &Path, dir: &str, budget: usize, exclude: &[String], units: &mut Vec<Unit>) {
    if excluded(dir, exclude) {
        return;
    }
    let files = dir_files(root, dir, exclude);
    if files.is_empty() {
        return;
    }
    let (subdirs, _) = children(&root.join(dir));
    if files.len() <= budget || subdirs.is_empty() {
        units.push(Unit {
            label: dir.to_string(),
            files,
            churn: 0,
        });
        return;
    }
    let own = own_files(root, dir, exclude);
    if !own.is_empty() {
        units.push(Unit {
            label: format!("{dir}/."),
            files: own,
            churn: 0,
        });
    }
    for sub in subdirs {
        expand(root, &format!("{dir}/{sub}"), budget, exclude, units);
    }
}

/// Slice units bigger than `budget` into `label (i/k)` pieces, then merge
/// adjacent units while the sum stays within the budget (labels joined by
/// `, `). Order is kept.
pub(crate) fn pack_units(units: Vec<Unit>, budget: usize) -> Vec<Unit> {
    let budget = budget.max(1);
    let mut sliced = Vec::new();
    for unit in units {
        if unit.files.len() <= budget {
            sliced.push(unit);
            continue;
        }
        let count = unit.files.len().div_ceil(budget);
        let size = unit.files.len().div_ceil(count);
        for (index, files) in unit.files.chunks(size).enumerate() {
            sliced.push(Unit {
                label: format!("{} ({}/{count})", unit.label, index + 1),
                files: files.to_vec(),
                churn: 0,
            });
        }
    }
    let mut packed: Vec<Unit> = Vec::new();
    for unit in sliced {
        match packed.last_mut() {
            Some(last) if last.files.len() + unit.files.len() <= budget => {
                last.label.push_str(LABEL_JOIN);
                last.label.push_str(&unit.label);
                last.files.extend(unit.files);
            }
            _ => packed.push(unit),
        }
    }
    packed
}

/// True when `path` (workspace-relative) is under an excluded directory:
/// an entry equal to it or one of its parents, or a bare name equal to one
/// of its components.
pub fn excluded(path: &str, exclude: &[String]) -> bool {
    let path = path.trim_matches('/');
    exclude.iter().any(|entry| {
        let entry = entry.trim().trim_matches('/');
        if entry.is_empty() {
            return false;
        }
        path == entry
            || path.starts_with(&format!("{entry}/"))
            || (!entry.contains('/') && path.split('/').any(|part| part == entry))
    })
}

/// Sorted names of the non-ignored, non-hidden top-level directories, minus
/// the excluded ones.
pub fn top_level_dirs(root: &Path, exclude: &[String]) -> Vec<String> {
    children(root)
        .0
        .into_iter()
        .filter(|dir| !excluded(dir, exclude))
        .collect()
}

/// A directory with subdirectories and no files of its own: `crates/`,
/// `packages/`.
pub(crate) fn is_container(dir: &Path) -> bool {
    let (subdirs, has_files) = children(dir);
    !has_files && !subdirs.is_empty()
}

/// Sorted names of the non-ignored, non-hidden directories directly in
/// `dir`, and whether it holds any such file of its own.
fn children(dir: &Path) -> (Vec<String>, bool) {
    let mut dirs = Vec::new();
    let mut has_files = false;
    for entry in ignore::WalkBuilder::new(dir)
        .require_git(false)
        .max_depth(Some(1))
        .build()
        .flatten()
    {
        if entry.depth() == 0 {
            continue;
        }
        match entry.file_type() {
            Some(kind) if kind.is_dir() => {
                dirs.push(entry.file_name().to_string_lossy().into_owned());
            }
            Some(kind) if kind.is_file() => has_files = true,
            _ => {}
        }
    }
    dirs.sort();
    (dirs, has_files)
}

/// Sorted workspace-relative paths of every non-ignored, non-hidden file
/// under `dir` (`.` for the workspace), minus excluded directories.
pub fn dir_files(root: &Path, dir: &str, exclude: &[String]) -> Vec<String> {
    walk_files(root, dir, None, exclude)
}

/// Like [`dir_files`], but only the files directly in `dir`.
pub fn own_files(root: &Path, dir: &str, exclude: &[String]) -> Vec<String> {
    walk_files(root, dir, Some(1), exclude)
}

fn walk_files(root: &Path, dir: &str, depth: Option<usize>, exclude: &[String]) -> Vec<String> {
    let base = if dir == "." || dir.is_empty() {
        root.to_path_buf()
    } else {
        root.join(dir)
    };
    let mut files: Vec<String> = ignore::WalkBuilder::new(&base)
        .require_git(false)
        .max_depth(depth)
        .build()
        .flatten()
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .filter_map(|entry| {
            let relative = entry.path().strip_prefix(root).ok()?;
            let path = relative.to_string_lossy().replace('\\', "/");
            (!excluded(&path, exclude)).then_some(path)
        })
        .collect();
    files.sort();
    files.dedup();
    files
}

/// True when more than `limit` non-ignored, non-hidden files are under
/// `root`. The walk stops at `limit + 1`, so a huge tree costs the same as
/// a large one.
pub fn workspace_exceeds(root: &Path, limit: usize) -> bool {
    let mut count = 0usize;
    for entry in ignore::WalkBuilder::new(root)
        .require_git(false)
        .build()
        .flatten()
    {
        if entry.file_type().is_some_and(|kind| kind.is_file()) {
            count += 1;
            if count > limit {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
#[path = "review_scope_tests.rs"]
mod tests;
