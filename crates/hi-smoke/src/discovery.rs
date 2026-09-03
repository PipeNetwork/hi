use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::scenario::Scenario;

#[derive(Clone, Debug)]
pub(crate) struct DiscoveredScenario {
    pub path: PathBuf,
    pub scenario: Scenario,
}

pub(crate) fn discover(root: &Path) -> Result<Vec<DiscoveredScenario>> {
    let root_metadata = std::fs::symlink_metadata(root)
        .with_context(|| format!("reading scenario root {}", root.display()))?;
    ensure!(
        !root_metadata.file_type().is_symlink(),
        "scenario root must not be a symlink: {}",
        root.display()
    );
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing scenario root {}", root.display()))?;
    let mut paths = Vec::new();
    collect(root, &canonical_root, root_metadata.is_file(), &mut paths)?;
    paths.sort();
    let mut found = Vec::with_capacity(paths.len());
    for path in paths {
        let scenario = Scenario::parse(&path)?;
        found.push(DiscoveredScenario { path, scenario });
    }
    if found.is_empty() {
        bail!("no scenario.toml files found under {}", root.display());
    }
    let mut names = std::collections::BTreeSet::new();
    for entry in &found {
        if !names.insert(entry.scenario.name.as_str()) {
            bail!("duplicate scenario name {:?}", entry.scenario.name);
        }
    }
    Ok(found)
}

fn collect(
    path: &Path,
    canonical_root: &Path,
    root_is_file: bool,
    found: &mut Vec<PathBuf>,
) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading scenario path {}", path.display()))?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "scenario suites may not contain symlinks: {}",
        path.display()
    );
    let canonical_path = path
        .canonicalize()
        .with_context(|| format!("canonicalizing scenario path {}", path.display()))?;
    ensure!(
        if root_is_file {
            canonical_path == canonical_root
        } else {
            canonical_path.starts_with(canonical_root)
        },
        "scenario path escaped suite root {}: {}",
        canonical_root.display(),
        path.display()
    );

    if metadata.is_file() {
        if path.file_name().is_some_and(|name| name == "scenario.toml")
            || path
                .extension()
                .is_some_and(|extension| extension == "toml")
        {
            found.push(path.to_path_buf());
        }
        return Ok(());
    }
    ensure!(
        metadata.is_dir(),
        "scenario path is neither a file nor a directory: {}",
        path.display()
    );
    let entries = std::fs::read_dir(path)
        .with_context(|| format!("reading scenario directory {}", path.display()))?;
    for entry in entries {
        let entry = entry?;
        let child = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("reading scenario entry type {}", child.display()))?;
        ensure!(
            !file_type.is_symlink(),
            "scenario suites may not contain symlinks: {}",
            child.display()
        );
        if file_type.is_dir() {
            collect(&child, canonical_root, root_is_file, found)?;
        } else if child
            .file_name()
            .is_some_and(|name| name == "scenario.toml")
        {
            found.push(child);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn discovery_rejects_external_and_cyclic_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let suite = temporary.path().join("suite");
        let outside = temporary.path().join("outside");
        std::fs::create_dir(&suite).unwrap();
        std::fs::create_dir(&outside).unwrap();
        symlink(&outside, suite.join("external")).unwrap();

        let error = discover(&suite).unwrap_err();
        assert!(error.to_string().contains("may not contain symlinks"));

        std::fs::remove_file(suite.join("external")).unwrap();
        symlink(&suite, suite.join("cycle")).unwrap();
        let error = discover(&suite).unwrap_err();
        assert!(error.to_string().contains("may not contain symlinks"));
    }

    #[cfg(unix)]
    #[test]
    fn discovery_rejects_a_symlinked_suite_root() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let suite = temporary.path().join("suite");
        let linked = temporary.path().join("linked-suite");
        std::fs::create_dir(&suite).unwrap();
        symlink(&suite, &linked).unwrap();

        let error = discover(&linked).unwrap_err();
        assert!(error.to_string().contains("root must not be a symlink"));
    }
}
