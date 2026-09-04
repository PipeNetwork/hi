//! File mutation prepare/commit path.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::edit::{apply_edit, plan_multi_patch};
use crate::paths::cache_key;
use crate::transaction::{MutationPlan, PlannedFileMutation};
use crate::{ToolEffects, ToolOutcome, format_lsp_error_feedback};

use super::parse;

/// A completely parsed and materialized file-tool invocation.
///
/// The contained [`MutationPlan`] owns the exact postimages shown by
/// [`PreparedMutation::preview`] and the preimage digests that must still match
/// when it is committed. Consuming this value is therefore the only supported
/// way to execute an edit after an interactive confirmation: the tool call is
/// never reparsed or rebuilt after approval.
#[derive(Debug)]
pub struct PreparedMutation {
    plan: MutationPlan,
    kind: PreparedMutationKind,
}

#[derive(Debug)]
enum PreparedMutationKind {
    Write {
        target: std::path::PathBuf,
        path: String,
        after: String,
    },
    Edit {
        target: std::path::PathBuf,
        path: String,
        after: String,
        replacements: usize,
        replace_all: bool,
    },
    MultiEdit {
        target: std::path::PathBuf,
        path: String,
        after: String,
        edit_count: usize,
    },
    ApplyPatch {
        summary: String,
    },
}

impl PreparedMutation {
    /// Render the exact postimages held by this prepared plan.
    pub fn preview(&self) -> String {
        self.plan.preview()
    }

    /// The exact canonical workspace-relative target for a single-file request.
    /// Approval scopes must use this sealed target instead of model arguments,
    /// which may contain parent traversal or symlink aliases. Multi-file plans
    /// return `None` and require approval for the complete operation.
    pub fn single_target_path(&self) -> Option<String> {
        self.plan.single_target_path()
    }
}
/// Refuse `write` overwrites of existing files larger than this (bytes). Forces
/// the model onto `edit` / `multi_edit` / `apply_patch` for real source rewrites.
/// Creates and small-file overwrites still go through `write`.
pub const MAX_WRITE_OVERWRITE_BYTES: u64 = 16 * 1024;

/// Parse and materialize one built-in file mutation without touching its
/// targets. Preparation errors are returned to the caller and must not be
/// discarded before asking for confirmation.
pub async fn prepare_mutation_in_with_state(
    root: &Path,
    state_root: &Path,
    name: &str,
    arguments: &str,
) -> Result<PreparedMutation> {
    match name {
        "write" => {
            let args: WriteArgs = parse(arguments)?;
            refuse_internal_elision_placeholder(&args.content)?;
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            refuse_large_write_overwrite(&target, &args.path)?;
            let recorded = crate::transaction::workspace_display_path(root, &target);
            let after = args.content;
            crate::read::refuse_oversized_text(&args.path, after.len())?;
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::write(
                    &recorded,
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::Write {
                    target,
                    path: recorded,
                    after,
                },
            })
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
            refuse_internal_elision_placeholder(&args.new_string)?;
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            let (before, after, replacements) = apply_edit_with_disk_retry(
                &target,
                &args.path,
                &args.old_string,
                &args.new_string,
                args.replace_all,
            )
            .await
            .with_context(|| format!("editing {}", args.path))?;
            crate::read::refuse_oversized_text(&args.path, after.len())?;
            let recorded = crate::transaction::workspace_display_path(root, &target);
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::update_from_preimage(
                    &recorded,
                    before.as_bytes(),
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::Edit {
                    target,
                    path: recorded,
                    after,
                    replacements,
                    replace_all: args.replace_all,
                },
            })
        }
        "multi_edit" => {
            let args: MultiEditArgs = parse(arguments)?;
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            if args.edits.is_empty() {
                bail!("no edits provided");
            }
            if args.edits.len() > MAX_MULTI_EDITS {
                bail!(
                    "multi_edit accepts at most {MAX_MULTI_EDITS} edits per call (got {})",
                    args.edits.len()
                );
            }
            for edit in &args.edits {
                refuse_internal_elision_placeholder(&edit.new_string)?;
            }
            let (before, after) =
                apply_multi_edit_with_disk_retry(&target, &args.path, &args.edits).await?;
            crate::read::refuse_oversized_text(&args.path, after.len())?;
            let recorded = crate::transaction::workspace_display_path(root, &target);
            let edit_count = args.edits.len();
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::update_from_preimage(
                    &recorded,
                    before.as_bytes(),
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::MultiEdit {
                    target,
                    path: recorded,
                    after,
                    edit_count,
                },
            })
        }
        "apply_patch" => {
            #[derive(Deserialize)]
            struct PatchArgs {
                patch: String,
            }
            let args: PatchArgs = parse(arguments)?;
            refuse_internal_elision_placeholder(&args.patch)?;
            if args.patch.len() > MAX_PATCH_BYTES {
                bail!(
                    "apply_patch payload is too large ({} bytes; limit {MAX_PATCH_BYTES})",
                    args.patch.len()
                );
            }
            let (plan, summary) =
                plan_multi_patch_with_disk_retry(root, state_root, &args.patch).await?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::ApplyPatch { summary },
            })
        }
        _ => bail!("{name} is not a preparable file mutation"),
    }
}

/// Cap on `multi_edit` hunks in one call. Each hunk is applied sequentially to
/// a growing buffer; an unbounded list is a CPU/memory bomb.
const MAX_MULTI_EDITS: usize = 64;
/// Cap on an `apply_patch` payload. Stream ingestion already stops at 4 MiB;
/// this is a tighter, earlier refuse so we never materialize a giant plan.
const MAX_PATCH_BYTES: usize = 1024 * 1024;

/// Compaction replaces old, bulky tool arguments with this internal marker.
/// It is context metadata, never valid replacement file content. A model can
/// quote an old tool call back to us, so reject the exact marker at the final
/// mutation boundary instead of allowing it to overwrite a source file.
fn refuse_internal_elision_placeholder(text: &str) -> Result<()> {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("[elided — ") else {
        return Ok(());
    };
    let Some(count) = rest.strip_suffix(" chars]") else {
        return Ok(());
    };
    if !count.is_empty() && count.chars().all(|ch| ch.is_ascii_digit()) {
        bail!(
            "refusing to write an internal transcript-elision placeholder; regenerate the actual content"
        );
    }
    Ok(())
}

fn refuse_large_write_overwrite(target: &Path, display_path: &str) -> Result<()> {
    if !target.is_file() {
        return Ok(());
    }
    let meta = std::fs::metadata(target)
        .with_context(|| format!("statting existing file {display_path}"))?;
    if meta.len() > MAX_WRITE_OVERWRITE_BYTES {
        bail!(
            "refusing to overwrite existing `{display_path}` ({} bytes) via `write` — \
             use `edit`, `multi_edit`, or `apply_patch` for in-place changes to large files \
             (limit {} bytes). `write` is for creates and small files only.",
            meta.len(),
            MAX_WRITE_OVERWRITE_BYTES
        );
    }
    Ok(())
}

/// Apply one edit; if the anchor miss looks like a stale disk race, re-read once
/// and retry. Ambiguous matches are never auto-picked.
async fn apply_edit_with_disk_retry(
    target: &Path,
    display_path: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<(String, String, usize)> {
    let path_str = target.to_string_lossy().into_owned();
    let before = crate::read::read_text_file(&path_str).await?;
    match apply_edit(&before, old, new, replace_all) {
        Ok(after) => {
            let replacements = if replace_all {
                before.matches(old).count().max(1)
            } else {
                1
            };
            Ok((before, after, replacements))
        }
        Err(first) if is_retryable_edit_miss(&first) => {
            // Brief yield so a concurrent writer can finish; then re-read.
            tokio::task::yield_now().await;
            let refreshed = crate::read::read_text_file(&path_str).await?;
            if refreshed == before {
                return Err(first).with_context(|| format!("editing {display_path}"));
            }
            let after = apply_edit(&refreshed, old, new, replace_all).with_context(|| {
                format!(
                    "editing {display_path} (retried after on-disk change; \
                     original miss: {first:#})"
                )
            })?;
            let replacements = if replace_all {
                refreshed.matches(old).count().max(1)
            } else {
                1
            };
            Ok((refreshed, after, replacements))
        }
        Err(err) => Err(err).with_context(|| format!("editing {display_path}")),
    }
}

async fn apply_multi_edit_with_disk_retry(
    target: &Path,
    display_path: &str,
    edits: &[EditOp],
) -> Result<(String, String)> {
    let path_str = target.to_string_lossy().into_owned();
    let before = crate::read::read_text_file(&path_str).await?;
    match apply_edit_chain(&before, edits, display_path) {
        Ok(after) => Ok((before, after)),
        Err(first) if is_retryable_edit_miss(&first) => {
            tokio::task::yield_now().await;
            let refreshed = crate::read::read_text_file(&path_str).await?;
            if refreshed == before {
                return Err(first);
            }
            let after = apply_edit_chain(&refreshed, edits, display_path).with_context(|| {
                format!("multi_edit {display_path} retried after on-disk change")
            })?;
            Ok((refreshed, after))
        }
        Err(err) => Err(err),
    }
}

fn apply_edit_chain(before: &str, edits: &[EditOp], display_path: &str) -> Result<String> {
    let mut after = before.to_string();
    for (index, edit) in edits.iter().enumerate() {
        after = apply_edit(&after, &edit.old_string, &edit.new_string, false)
            .with_context(|| format!("editing {display_path} (edit #{})", index + 1))?;
    }
    Ok(after)
}

async fn plan_multi_patch_with_disk_retry(
    root: &Path,
    state_root: &Path,
    patch: &str,
) -> Result<(MutationPlan, String)> {
    match plan_multi_patch(root, state_root, patch) {
        Ok(ok) => Ok(ok),
        Err(first) if is_retryable_patch_miss(&first) => {
            tokio::task::yield_now().await;
            // Re-plan reads files fresh from disk; a second attempt only helps
            // when the underlying files changed underfoot.
            match plan_multi_patch(root, state_root, patch) {
                Ok(ok) => Ok(ok),
                Err(second) => {
                    Err(first).with_context(|| format!("apply_patch failed ({second:#})"))
                }
            }
        }
        Err(err) => Err(err),
    }
}

pub(crate) fn is_retryable_edit_miss(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("old_string not found") || msg.contains("replace_all found no exact occurrences")
}

pub(crate) fn is_retryable_patch_miss(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    // found 0 → stale; found >1 → ambiguous — only retry stale (0).
    msg.contains("hunk context must match one unique contiguous region (found 0)")
        || msg.contains("addition-only hunk has no unique insertion anchor")
}

/// Commit the exact mutation plan previously displayed for confirmation.
/// Preimage changes made while the confirmation UI was open cause a typed
/// failure and are never overwritten.
pub async fn execute_prepared_in_runtime(
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    prepared: PreparedMutation,
) -> ToolOutcome {
    match run_prepared_mutation(lsp, read_cache, None, prepared).await {
        Ok(outcome) => outcome,
        Err(error) => {
            // A failed digest precondition means something else changed the
            // workspace while confirmation was open. Do not let a later read
            // reuse content cached before that external edit.
            if let Ok(mut cache) = read_cache.lock() {
                cache.clear();
            }
            let mut outcome = ToolOutcome::failed(format!("Error: {error:#}"));
            outcome.effects.mutation_attempted = true;
            outcome
        }
    }
}

pub(super) async fn run_prepared_mutation(
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    hunk_tracker: Option<&hi_hunk_tracker::HunkTrackerHandle>,
    prepared: PreparedMutation,
) -> Result<ToolOutcome> {
    let display = prepared.preview();
    // Record agent writes for hunk-level attribution. Best-effort: a closed
    // handle or send error must not fail the mutation. We record before
    // commit so the hunk-tracker sees the postimage even if commit fails.
    if let Some(tracker) = hunk_tracker {
        let (target, after) = match &prepared.kind {
            PreparedMutationKind::Write { target, after, .. }
            | PreparedMutationKind::Edit { target, after, .. }
            | PreparedMutationKind::MultiEdit { target, after, .. } => {
                (target.clone(), after.clone())
            }
            PreparedMutationKind::ApplyPatch { .. } => (std::path::PathBuf::new(), String::new()),
        };
        if !after.is_empty() {
            tracker.record_agent_write(target, after, 0, None);
        }
    }
    let changes = prepared.plan.commit()?;
    let mut outcome = match prepared.kind {
        PreparedMutationKind::Write {
            target,
            path,
            after,
        } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.remove(&cache_key(&target));
            }
            sync_lsp_document(lsp, &target, &after).await;
            let mut outcome =
                ToolOutcome::shown(format!("Wrote {} bytes to {path}", after.len()), display);
            attach_lsp_diagnostics(lsp, &target, &mut outcome).await;
            outcome
        }
        PreparedMutationKind::Edit {
            target,
            path,
            after,
            replacements,
            replace_all,
        } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.remove(&cache_key(&target));
            }
            sync_lsp_document(lsp, &target, &after).await;
            let message = if replace_all && replacements > 1 {
                format!("Replaced {replacements} occurrences in {path}")
            } else {
                format!("Edited {path}")
            };
            let mut outcome = ToolOutcome::shown(message, display);
            attach_lsp_diagnostics(lsp, &target, &mut outcome).await;
            outcome
        }
        PreparedMutationKind::MultiEdit {
            target,
            path,
            after,
            edit_count,
        } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.remove(&cache_key(&target));
            }
            sync_lsp_document(lsp, &target, &after).await;
            let mut outcome =
                ToolOutcome::shown(format!("Applied {edit_count} edits to {path}"), display);
            attach_lsp_diagnostics(lsp, &target, &mut outcome).await;
            outcome
        }
        PreparedMutationKind::ApplyPatch { summary } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.clear();
            }
            ToolOutcome::plain(summary)
        }
    };
    outcome.effects = mutation_effects(changes);
    Ok(outcome)
}

async fn sync_lsp_document(lsp: &std::sync::Arc<hi_lsp::LspManager>, path: &Path, text: &str) {
    let _ = lsp.sync_document(path, text).await;
}

const MUTATION_DIAGNOSTICS_BUDGET: Duration = Duration::from_millis(1500);

/// Wait briefly for LSP diagnostics on the edited path (and a few siblings).
/// Timeout or a disabled/fake-empty server leaves the mutation successful.
async fn attach_lsp_diagnostics(
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    edited: &Path,
    outcome: &mut ToolOutcome,
) {
    let mut paths = vec![edited.to_path_buf()];
    paths.extend(sibling_source_paths(edited, 3));
    let wait = lsp.diagnostics_batch(&paths);
    let states = match tokio::time::timeout(MUTATION_DIAGNOSTICS_BUDGET, wait).await {
        Ok(states) => states,
        Err(_) => return,
    };
    let items = flatten_mutation_diagnostics(lsp.root(), edited, &states);
    if items.is_empty() {
        return;
    }
    let body = format_lsp_error_feedback(&items);
    if body.is_empty() {
        return;
    }
    outcome.content.push_str("\n\n<diagnostics>\n");
    outcome.content.push_str(&body);
    outcome.content.push_str("\n</diagnostics>");
}

fn sibling_source_paths(edited: &Path, max: usize) -> Vec<PathBuf> {
    let Some(parent) = edited.parent() else {
        return Vec::new();
    };
    let Some(ext) = edited.extension() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        if out.len() >= max {
            break;
        }
        let path = entry.path();
        if path == edited {
            continue;
        }
        if path.extension() != Some(ext) || !path.is_file() {
            continue;
        }
        out.push(path);
    }
    out
}

fn flatten_mutation_diagnostics(
    root: &Path,
    edited: &Path,
    states: &[(PathBuf, hi_lsp::DiagnosticState)],
) -> Vec<(String, u32, u32, String)> {
    let mut items = Vec::new();
    for (path, state) in states {
        let hi_lsp::DiagnosticState::DiagnosticsPresent { diagnostics, .. } = state else {
            continue;
        };
        let is_edited = paths_match(path, edited);
        let display = path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string();
        for d in diagnostics {
            let include = d.severity == "error" || (is_edited && d.severity == "warning");
            if !include {
                continue;
            }
            let message = if d.severity == "warning" {
                format!("warning: {}", d.message)
            } else {
                d.message.clone()
            };
            items.push((
                display.clone(),
                d.line.saturating_add(1),
                d.col.saturating_add(1),
                message,
            ));
        }
    }
    items
}

fn paths_match(a: &Path, b: &Path) -> bool {
    a == b
        || a.canonicalize()
            .ok()
            .zip(b.canonicalize().ok())
            .is_some_and(|(a, b)| a == b)
}

fn mutation_effects(changes: Vec<crate::FileChange>) -> ToolEffects {
    ToolEffects {
        mutation_attempted: true,
        mutation_applied: !changes.is_empty(),
        file_changes: changes,
    }
}
/// Render a mutation preview without applying it (test helper).
#[cfg(test)]
pub(crate) async fn preview_edit_in(root: &Path, name: &str, arguments: &str) -> Option<String> {
    prepare_mutation_in_with_state(root, &root.join(".hi-test-state"), name, arguments)
        .await
        .ok()
        .map(|prepared| prepared.preview())
}

#[derive(Deserialize)]
pub(crate) struct MultiEditArgs {
    pub path: String,
    pub edits: Vec<EditOp>,
}

#[derive(Deserialize)]
pub(crate) struct EditOp {
    pub old_string: String,
    pub new_string: String,
}

#[derive(Deserialize)]
pub(crate) struct WriteArgs {
    pub path: String,
    pub content: String,
}

#[derive(Deserialize)]
pub(crate) struct EditArgs {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
    /// If true, replace every occurrence of `old_string` (default: false).
    #[serde(default)]
    pub replace_all: bool,
}
