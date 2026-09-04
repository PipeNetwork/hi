use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};

pub(super) fn canonical_root(root: &Path) -> Result<PathBuf> {
    let metadata =
        fs::metadata(root).with_context(|| format!("reading workspace root {}", root.display()))?;
    ensure!(
        metadata.is_dir(),
        "workspace root is not a directory: {}",
        root.display()
    );
    root.canonicalize()
        .with_context(|| format!("canonicalizing workspace root {}", root.display()))
}

pub(crate) fn resolve_workspace_target(root: &Path, requested: &Path) -> Result<PathBuf> {
    // Callers pass the workspace root in whatever form they hold (the agent
    // runtime pre-canonicalizes; direct tool entry points like `execute_in`
    // may not). Canonicalize here so the containment check and the returned
    // target never depend on caller hygiene.
    let root = &canonical_root(root)?;
    let joined = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    let resolved = resolve_components(&joined)?;
    ensure!(
        resolved.starts_with(root),
        "path '{}' is outside workspace {}",
        requested.display(),
        root.display()
    );
    // Return the path that was actually validated. The lexical form can name
    // the workspace through a symlink alias (macOS `/var` → `/private/var`,
    // so any `$TMPDIR` path) — every later containment check compares against
    // the canonical root and would reject it ("parent escaped workspace"),
    // failing all mutations in such workspaces. It would also let the write
    // land on a different file than the one the escape check resolved.
    Ok(resolved)
}

/// Checkpoint postimages replace a directory entry itself, including a live
/// symlink. Resolve parents for containment without following the final name.
pub(super) fn resolve_restore_target(root: &Path, requested: &Path) -> Result<PathBuf> {
    let name = requested.file_name().with_context(|| {
        format!(
            "restore target must name a workspace entry: {}",
            requested.display()
        )
    })?;
    let parent = resolve_workspace_target(root, requested.parent().unwrap_or(Path::new("")))?;
    match fs::symlink_metadata(&parent) {
        Ok(metadata) => ensure!(
            metadata.is_dir(),
            "restore target parent is not a directory: {}",
            parent.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading restore parent {}", parent.display()));
        }
    }
    Ok(parent.join(name))
}

/// The workspace-relative display form of a resolved mutation target. Effects
/// and ledger records must use this — never the caller's verbatim path string
/// — so a model-supplied alias (`./x`, an absolute path, or a path resolved
/// through any fallback) can never make the recorded change disagree with the
/// file that actually changed.
pub(crate) fn workspace_display_path(root: &Path, target: &Path) -> String {
    let stripped = canonical_root(root)
        .ok()
        .and_then(|canonical| target.strip_prefix(&canonical).ok().map(Path::to_path_buf));
    match stripped {
        Some(relative) if !relative.as_os_str().is_empty() => {
            relative.to_string_lossy().replace('\\', "/")
        }
        _ => target.to_string_lossy().into_owned(),
    }
}

fn resolve_components(path: &Path) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // Existing symlinks have already been resolved, so `alias/..`
                // selects the target's parent just as a normal filesystem call
                // does. Only not-yet-existing components are collapsed lexically.
                out.pop();
            }
            Component::RootDir => out.push(Path::new("/")),
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::Normal(name) => {
                out.push(name);
                match fs::symlink_metadata(&out) {
                    Ok(metadata) => {
                        match out.canonicalize() {
                            Ok(canonical) => out = canonical,
                            // Keep an unresolved final symlink's identity for
                            // no-follow checkpoint inspection. Ordinary file
                            // mutations reject this non-regular target later.
                            Err(_) if metadata.is_symlink() && components.peek().is_none() => {}
                            Err(error) => {
                                return Err(error).with_context(|| {
                                    format!("resolving path component {}", out.display())
                                });
                            }
                        }
                        if components.peek().is_some() {
                            ensure!(
                                fs::metadata(&out)?.is_dir(),
                                "path component is not a directory: {}",
                                out.display()
                            );
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error)
                            .with_context(|| format!("resolving path {}", out.display()));
                    }
                }
            }
        }
    }
    Ok(out)
}
