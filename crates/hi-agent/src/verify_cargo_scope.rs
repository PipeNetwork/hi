//! Identity for the default Cargo checks dispatched by native validators.
//!
//! Package/features/target/workspace flags and shell programs remain exact
//! command scopes. Cargo's feature unification makes inferred broader coverage
//! unsafe, even when another invocation compiles more packages or tests.

use std::path::{Path, PathBuf};

use crate::recovery::ValidationObservation;

pub(crate) fn canonicalize_package(
    observation: &mut ValidationObservation,
    root: &Path,
    command: &str,
    package: &str,
) {
    if command == "cargo check"
        && let Some(scope) = NativeCheck::manifest(root, &root.join(package).join("Cargo.toml"))
    {
        scope.apply(observation);
    }
}

pub(crate) fn canonicalize_command(
    observation: &mut ValidationObservation,
    root: &Path,
    command: &str,
) {
    if let Some(scope) = NativeCheck::command(root, command) {
        scope.apply(observation);
    }
}

struct NativeCheck {
    package: String,
    manifest: PathBuf,
}

impl NativeCheck {
    fn command(root: &Path, command: &str) -> Option<Self> {
        let command = command.trim();
        if matches!(
            command,
            "cargo check" | "cargo check --quiet" | "cargo check -q"
        ) {
            return Self::manifest(root, &root.join("Cargo.toml"));
        }
        let argument = [
            "cargo check --quiet --manifest-path ",
            "cargo check -q --manifest-path ",
            "cargo check --manifest-path ",
        ]
        .iter()
        .find_map(|prefix| command.strip_prefix(prefix))?;
        let manifest = literal_argument(argument)?;
        Self::manifest(root, &root.join(manifest))
    }

    fn manifest(root: &Path, manifest: &Path) -> Option<Self> {
        let root = root.canonicalize().ok()?;
        let manifest = manifest.canonicalize().ok()?;
        if !manifest.is_file() || manifest.file_name()? != "Cargo.toml" {
            return None;
        }
        let relative = manifest.strip_prefix(&root).ok()?;
        let package = relative.parent()?.to_str()?;
        Some(Self {
            package: if package.is_empty() {
                ".".into()
            } else {
                package.replace('\\', "/")
            },
            manifest,
        })
    }

    fn apply(self, observation: &mut ValidationObservation) {
        // Retain the historical native fast-check key. The equivalent literal
        // commands are only discharged after a real, stable native pass; saved
        // recovery records and their integrity hashes are never rewritten.
        let scope = format!("cargo check [package:{}]", self.package);
        let mut aliases = vec![std::mem::replace(&mut observation.scope, scope.clone())];
        if self.package == "." {
            aliases.extend(
                ["cargo check", "cargo check --quiet", "cargo check -q"].map(str::to_owned),
            );
        }
        let relative = if self.package == "." {
            "Cargo.toml".to_owned()
        } else {
            format!("{}/Cargo.toml", self.package)
        };
        let mut manifests = vec![relative];
        if let Some(absolute) = self.manifest.to_str() {
            manifests.push(absolute.to_owned());
        }
        for manifest in manifests {
            for verbosity in ["", " --quiet", " -q"] {
                aliases.push(format!(
                    "cargo check{verbosity} --manifest-path {}",
                    super::shell_quote(&manifest)
                ));
            }
        }
        aliases.retain(|alias| alias != &scope);
        aliases.sort();
        aliases.dedup();
        observation.equivalent_scopes = aliases;
    }
}

/// Parse only a literal argument, including the exact quote escaping emitted
/// by our stage builder. No expansions, environment assignments or shell chains.
fn literal_argument(argument: &str) -> Option<String> {
    if argument.starts_with('\'') && argument.ends_with('\'') && argument.len() >= 2 {
        let decoded = argument[1..argument.len() - 1].replace("'\"'\"'", "'");
        return (super::shell_quote(&decoded) == argument).then_some(decoded);
    }
    (!argument.is_empty()
        && argument
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')))
    .then(|| argument.to_owned())
}

#[cfg(test)]
#[path = "verify_cargo_scope_tests.rs"]
mod tests;
