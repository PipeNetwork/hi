//! File-based hook discovery — scans `.hi/hooks/` directories for TOML files.

use std::collections::HashMap;
use std::path::Path;

use crate::config::{HookFile, HookSpec};
use crate::event::HookEvent;

/// The loaded set of hooks, indexed by event type for fast lookup.
///
/// This is a point-in-time snapshot. Edits to hook files on disk are only
/// picked up by new sessions.
#[derive(Debug, Clone, Default)]
pub struct HookRegistry {
    hooks: HashMap<HookEvent, Vec<HookSpec>>,
}

impl HookRegistry {
    /// Hooks registered under the exact event key.
    pub fn hooks_for(&self, event: HookEvent) -> &[HookSpec] {
        self.hooks.get(&event).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// True when any enabled hook is registered for `event`.
    pub fn has_enabled_hooks(&self, event: HookEvent) -> bool {
        self.hooks_for(event).iter().any(|s| s.enabled)
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.values().all(|v| v.is_empty())
    }

    pub fn len(&self) -> usize {
        self.hooks.values().map(|v| v.len()).sum()
    }

    pub fn append_specs(&mut self, specs: Vec<HookSpec>) {
        for spec in specs {
            self.hooks.entry(spec.event).or_default().push(spec);
        }
    }
}

/// Discover hooks from global and project-local hook directories.
///
/// Scans `global_dir` and `project_dir` for `*.toml` files, parses each as a
/// [`HookFile`], and builds a [`HookRegistry`]. Returns the registry and a list
/// of non-fatal errors (bad files are skipped, not fatal).
pub fn discover_hooks(
    global_dir: Option<&Path>,
    project_dir: Option<&Path>,
) -> (HookRegistry, Vec<String>) {
    let mut specs = Vec::new();
    let mut errors = Vec::new();

    for dir in global_dir.into_iter().chain(project_dir) {
        let (dir_specs, dir_errors) = load_hooks_from_dir(dir);
        specs.extend(dir_specs);
        errors.extend(dir_errors);
    }

    let mut registry = HookRegistry::default();
    registry.append_specs(specs);
    (registry, errors)
}

fn load_hooks_from_dir(dir: &Path) -> (Vec<HookSpec>, Vec<String>) {
    let mut specs = Vec::new();
    let mut errors = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return (specs, errors), // dir doesn't exist — not an error
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors.push(format!("reading hook dir entry: {e}"));
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                errors.push(format!("reading {}: {e}", path.display()));
                continue;
            }
        };
        let file: HookFile = match toml::from_str(&content) {
            Ok(f) => f,
            Err(e) => {
                errors.push(format!("parsing {}: {e}", path.display()));
                continue;
            }
        };
        let source_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        match HookSpec::from_file(file, source_dir) {
            Ok(spec) => specs.push(spec),
            Err(e) => {
                errors.push(format!("building hook from {}: {e}", path.display()));
            }
        }
    }

    (specs, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn discovers_hooks_from_toml_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("format.toml"),
            r#"
name = "rustfmt"
event = "post_tool_use"
command = "rustfmt {{file}}"
matcher = "edit"
timeout = 10
"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("guard.toml"),
            r#"
name = "no-rm-rf"
event = "pre_tool_use"
command = "echo denied"
matcher = "bash*"
"#,
        )
        .unwrap();

        let (registry, errors) = discover_hooks(None, Some(dir.path()));
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(registry.len(), 2);
        assert!(registry.has_enabled_hooks(HookEvent::PreToolUse));
        assert!(registry.has_enabled_hooks(HookEvent::PostToolUse));
    }

    #[test]
    fn skips_non_toml_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("readme.md"), "# hooks").unwrap();
        fs::write(dir.path().join("script.sh"), "#!/bin/sh").unwrap();

        let (registry, errors) = discover_hooks(None, Some(dir.path()));
        assert!(errors.is_empty());
        assert!(registry.is_empty());
    }

    #[test]
    fn reports_parse_errors_without_failing() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("bad.toml"), "not valid toml = = =").unwrap();

        let (registry, errors) = discover_hooks(None, Some(dir.path()));
        assert!(registry.is_empty());
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("parsing"));
    }

    #[test]
    fn missing_dir_is_not_an_error() {
        let (registry, errors) = discover_hooks(Some(Path::new("/nonexistent/hi/hooks")), None);
        assert!(registry.is_empty());
        assert!(errors.is_empty());
    }
}
