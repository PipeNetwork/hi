//! File mutation prepare/commit path.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::edit::{apply_edit, plan_multi_patch};
use crate::paths::cache_key;
use crate::transaction::{MutationPlan, PlannedFileMutation};
use crate::{ToolEffects, ToolOutcome};

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
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            refuse_large_write_overwrite(&target, &args.path)?;
            let after = args.content;
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::write(
                    &args.path,
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::Write {
                    target,
                    path: args.path,
                    after,
                },
            })
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
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
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::update_from_preimage(
                    &args.path,
                    before.as_bytes(),
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::Edit {
                    target,
                    path: args.path,
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
            let (before, after) =
                apply_multi_edit_with_disk_retry(&target, &args.path, &args.edits).await?;
            let edit_count = args.edits.len();
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::update_from_preimage(
                    &args.path,
                    before.as_bytes(),
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::MultiEdit {
                    target,
                    path: args.path,
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
    match run_prepared_mutation(lsp, read_cache, prepared).await {
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
    prepared: PreparedMutation,
) -> Result<ToolOutcome> {
    let display = prepared.preview();
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
            ToolOutcome::shown(format!("Wrote {} bytes to {path}", after.len()), display)
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
            ToolOutcome::shown(message, display)
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
            ToolOutcome::shown(format!("Applied {edit_count} edits to {path}"), display)
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

fn mutation_effects(changes: Vec<crate::FileChange>) -> ToolEffects {
    ToolEffects {
        mutation_attempted: true,
        mutation_applied: !changes.is_empty(),
        file_changes: changes,
    }
}
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
