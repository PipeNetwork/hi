//! Admission and settlement for harness-owned workspace files.
//!
//! Curation and memory maintenance are not model-invoked tools, but their
//! bytes are still part of the authoritative workspace. This module gives
//! those small synchronous writers the same permit → effect → reconcile →
//! transcript → settlement boundary as ordinary tools. Explicit host-local
//! paths (for example global user memory) stay outside PipeFS publication.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

impl crate::Agent {
    /// Run a write which is deliberately host-local, never part of the
    /// authoritative workspace. Refuse an override that points into the
    /// workspace rather than silently bypassing controller admission.
    pub(crate) fn run_host_local_file_mutation<T, F>(
        &self,
        description: &str,
        paths: &[PathBuf],
        mutation: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        ensure_host_local_paths(self.workspace_root(), paths, description)?;
        mutation()
    }

    pub(crate) async fn run_internal_file_mutation<T, F>(
        &mut self,
        operation_name: &'static str,
        description: &str,
        paths: &[PathBuf],
        input: serde_json::Value,
        mutation: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<(T, String)>,
    {
        // Legacy PipeFS can CAS workspace bytes but has no durable operation
        // slot for this synthetic transcript. Optional maintenance must stay
        // disabled rather than publishing bytes with missing causal evidence.
        anyhow::ensure!(
            !self.pipefs_workspace_active() || self.config.harness.features.workspace_controller_v2,
            "{description} requires workspace_controller_v2 under PipeFS"
        );
        let dirty_paths = classify_paths(self.workspace_root(), paths)?.ok_or_else(|| {
            anyhow!(
                "{description} targets host-local state; use the explicit host-local mutation path"
            )
        })?;

        let mut intent = hi_workspace::MutationIntent::workspace(description);
        intent.dirty_paths = Some(dirty_paths.clone());
        self.begin_classified_workspace_operation(intent)
            .await
            .with_context(|| format!("workspace controller refused {description}"))?;

        let ledger_revision = self.runtime.ledger().revision();
        let (value, result_text, mutation_error) = match mutation() {
            Ok((value, result)) => (Some(value), result, None),
            Err(error) => {
                let result = format!("{description} failed: {error:#}");
                (None, result, Some(error))
            }
        };
        let after_effect = hi_workspace::hit_harness_failpoint(
            hi_workspace::HarnessFailpoint::ExecutionAfterEffect,
        )
        .map_err(anyhow::Error::from);
        let reconciliation = self.reconcile_workspace_changes().await;
        let changes = if reconciliation.is_ok() {
            self.runtime.ledger().changes_since(ledger_revision)
        } else {
            Vec::new()
        };

        let indeterminate = after_effect.is_err() || reconciliation.is_err();
        let mut execution = hi_workspace::ExecutionReport {
            disposition: if indeterminate {
                hi_workspace::ExecutionDisposition::Indeterminate
            } else if mutation_error.is_some() {
                hi_workspace::ExecutionDisposition::Failed
            } else {
                hi_workspace::ExecutionDisposition::Succeeded
            },
            workspace_may_have_changed: indeterminate || !changes.is_empty(),
            external_effect_may_have_occurred: false,
            content_digest: None,
            changed_paths: changes
                .iter()
                .map(|change| PathBuf::from(&change.path))
                .collect(),
            artifacts: Vec::new(),
            detail: operation_detail(
                description,
                mutation_error.as_ref(),
                after_effect.as_ref().err(),
                reconciliation.as_ref().err(),
            ),
        };

        // Controller-v2 admission has a durable operation id. During the
        // compatibility rollout the legacy backend can still admit without
        // installing a controller permit; retain a unique transcript call id
        // instead of panicking while that feature gate remains supported.
        let operation_label = self
            .workspace_coordination
            .active_parent_operation()
            .map(|operation_id| operation_id.to_string())
            .unwrap_or_else(|| format!("legacy-{}", uuid::Uuid::new_v4()));
        let call_id = format!("internal-workspace:{operation_name}:{operation_label}");
        let arguments = serde_json::json!({
            "operation": operation_name,
            "paths": dirty_paths,
            "input": input,
        })
        .to_string();
        let calls = [(
            call_id.clone(),
            operation_name.to_owned(),
            arguments.clone(),
        )];
        let assistant_content = [hi_ai::Content::ToolCall {
            id: call_id.clone(),
            name: operation_name.to_owned(),
            arguments,
        }];
        let results = [(call_id, result_text)];
        let stage_error = self
            .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
            .err();
        if let Some(error) = &stage_error {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.content_digest = None;
            execution.detail = Some(match execution.detail.take() {
                Some(detail) => format!(
                    "{detail}; exact internal-mutation transcript could not be staged: {error:#}"
                ),
                None => {
                    format!("exact internal-mutation transcript could not be staged: {error:#}")
                }
            });
        }
        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;

        let mut publication_failures = Vec::new();
        if let Err(error) = after_effect {
            publication_failures.push(format!("post-effect crash boundary: {error:#}"));
        }
        if let Err(error) = reconciliation {
            publication_failures.push(format!("workspace reconciliation failed: {error:#}"));
        }
        if let Some(error) = stage_error {
            publication_failures.push(format!("transcript staging failed: {error:#}"));
        }
        if let Err(error) = settlement {
            publication_failures.push(format!("durable settlement failed: {error:#}"));
        }

        match (value, mutation_error, publication_failures.is_empty()) {
            (Some(value), None, true) => Ok(value),
            (_, Some(error), true) => Err(error),
            (_, Some(error), false) => Err(error.context(publication_failures.join("; "))),
            (Some(_), None, false) | (None, None, false) => {
                bail!(
                    "{description} was not published: {}",
                    publication_failures.join("; ")
                )
            }
            (None, None, true) => Err(anyhow!("{description} returned no result")),
        }
    }
}

fn operation_detail(
    description: &str,
    mutation: Option<&anyhow::Error>,
    after_effect: Option<&anyhow::Error>,
    reconciliation: Option<&anyhow::Error>,
) -> Option<String> {
    let mut details = Vec::new();
    if let Some(error) = mutation {
        details.push(format!("{description} failed: {error:#}"));
    }
    if let Some(error) = after_effect {
        details.push(format!("post-effect crash boundary failed: {error:#}"));
    }
    if let Some(error) = reconciliation {
        details.push(format!("workspace reconciliation failed: {error:#}"));
    }
    (!details.is_empty()).then(|| details.join("; "))
}

fn ensure_host_local_paths(root: &Path, paths: &[PathBuf], description: &str) -> Result<()> {
    anyhow::ensure!(
        classify_paths(root, paths)?.is_none(),
        "{description} is host-local but its target overlaps the authoritative workspace"
    );
    Ok(())
}

/// `Some` means every path is safely inside the workspace and contains the
/// resolved relative paths for controller admission. `None` means every path
/// is an explicit absolute host-local target. Mixed scopes fail closed.
fn classify_paths(root: &Path, paths: &[PathBuf]) -> Result<Option<Vec<PathBuf>>> {
    anyhow::ensure!(
        !paths.is_empty(),
        "internal mutation declared no target paths"
    );
    let root_abs = lexical_absolute(root)?;
    let root_resolved = resolve_existing_ancestor(&root_abs)?;
    let mut workspace_paths = Vec::new();
    let mut host_local = false;

    for path in paths {
        let candidate = lexical_normalize(&if path.is_absolute() {
            path.clone()
        } else {
            root_abs.join(path)
        });
        let resolved = resolve_existing_ancestor(&candidate)?;
        let intended_workspace = !path.is_absolute()
            || candidate.starts_with(&root_abs)
            || resolved.starts_with(&root_resolved);
        if !intended_workspace {
            host_local = true;
            continue;
        }
        let relative = resolved.strip_prefix(&root_resolved).with_context(|| {
            format!(
                "internal workspace path escapes through a parent or symlink: {}",
                path.display()
            )
        })?;
        anyhow::ensure!(
            !relative.as_os_str().is_empty(),
            "internal mutation cannot target the workspace root"
        );
        workspace_paths.push(relative.to_path_buf());
    }

    if host_local && !workspace_paths.is_empty() {
        bail!("internal mutation cannot mix workspace and host-local paths");
    }
    if host_local {
        Ok(None)
    } else {
        Ok(Some(workspace_paths))
    }
}

fn lexical_absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(lexical_normalize(path))
    } else {
        Ok(lexical_normalize(&std::env::current_dir()?.join(path)))
    }
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn resolve_existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut cursor = path;
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(cursor) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = cursor.file_name().with_context(|| {
                    format!(
                        "internal mutation path has no existing ancestor: {}",
                        path.display()
                    )
                })?;
                missing.push(name.to_os_string());
                cursor = cursor.parent().with_context(|| {
                    format!("internal mutation path has no parent: {}", path.display())
                })?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("inspecting internal mutation path {}", cursor.display())
                });
            }
        }
    }
    let mut resolved = cursor.canonicalize().with_context(|| {
        format!(
            "internal mutation path has an unresolved or dangling symlink: {}",
            cursor.display()
        )
    })?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_classification_separates_workspace_and_host_local_targets() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(
            classify_paths(root.path(), &[root.path().join(".hi/memory.md")]).unwrap(),
            Some(vec![PathBuf::from(".hi/memory.md")])
        );
        assert_eq!(
            classify_paths(
                root.path(),
                &[std::env::temp_dir().join("global-memory.md")]
            )
            .unwrap(),
            None
        );
        assert!(
            classify_paths(
                root.path(),
                &[
                    root.path().join(".hi/memory.md"),
                    std::env::temp_dir().join("global-memory.md")
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn host_local_mutation_refuses_a_workspace_target_before_effect() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join(".hi/global-memory.md");
        let error = ensure_host_local_paths(root.path(), &[target], "global memory").unwrap_err();

        assert!(error.to_string().contains("overlaps"));
    }

    #[cfg(unix)]
    #[test]
    fn workspace_symlink_escape_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join(".hi")).unwrap();
        let error = classify_paths(root.path(), &[root.path().join(".hi/memory.md")]).unwrap_err();
        assert!(
            error.to_string().contains("escapes") || error.to_string().contains("dangling symlink")
        );
    }

    #[cfg(unix)]
    #[test]
    fn memory_undo_symlink_escape_is_rejected_before_publication() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join(".hi")).unwrap();
        let memory = root.path().join(".hi/memory.md");
        symlink(
            outside.path().join("captured.md"),
            root.path().join(".hi/memory.undo.md"),
        )
        .unwrap();

        let error =
            classify_paths(root.path(), &crate::memory::memory_write_paths(&memory)).unwrap_err();

        assert!(
            error.to_string().contains("escapes") || error.to_string().contains("dangling symlink")
        );
        assert!(!outside.path().join("captured.md").exists());
    }
}
