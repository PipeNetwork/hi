//! Deterministic diff-hygiene checks after a green WorkspaceRepair pass.
//!
//! These are conservative merge-quality signals, not compile/test authority.
//! Findings re-enter the model like an independent-review OBJECT.

use hi_tools::{FileChange, FileChangeKind};

use crate::task_contract::TaskContract;

/// Cap (bytes) above which a single Create/Modify is a hygiene finding.
pub(crate) const LARGE_FILE_BYTES: u64 = 32 * 1024;
/// Unreferenced Creates that trip the sprawl check on a narrow mutation.
const UNREFERENCED_CREATE_THRESHOLD: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HygieneFinding {
    pub reason: String,
}

/// Assess a green turn's file changes against the task contract.
pub(crate) fn assess(
    contract: &TaskContract,
    changes: &[FileChange],
    prompt: &str,
) -> Vec<HygieneFinding> {
    let mut findings = Vec::new();
    if let Some(finding) = unreferenced_creates(contract, changes) {
        findings.push(finding);
    }
    if let Some(finding) = unexpected_dependency_manifest(contract, changes, prompt) {
        findings.push(finding);
    }
    if let Some(finding) = oversized_file(changes) {
        findings.push(finding);
    }
    findings
}

fn unreferenced_creates(contract: &TaskContract, changes: &[FileChange]) -> Option<HygieneFinding> {
    if contract.referenced_paths.is_empty() {
        return None;
    }
    let extras: Vec<&str> = changes
        .iter()
        .filter(|change| change.kind == FileChangeKind::Create)
        .map(|change| change.path.as_str())
        .filter(|path| !path_is_referenced(path, &contract.referenced_paths))
        .collect();
    if extras.len() < UNREFERENCED_CREATE_THRESHOLD {
        return None;
    }
    Some(HygieneFinding {
        reason: format!(
            "narrow mutation named {} but created {} unreferenced files ({})",
            contract.referenced_paths.join(", "),
            extras.len(),
            extras
                .iter()
                .take(6)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

fn path_is_referenced(path: &str, referenced: &[String]) -> bool {
    let normalized = path.replace('\\', "/");
    referenced.iter().any(|named| {
        let named = named.replace('\\', "/");
        normalized == named
            || normalized.starts_with(&format!("{named}/"))
            || named.ends_with('/') && normalized.starts_with(&named)
            || normalized.rsplit('/').next() == Some(named.as_str())
    })
}

fn unexpected_dependency_manifest(
    contract: &TaskContract,
    changes: &[FileChange],
    prompt: &str,
) -> Option<HygieneFinding> {
    // Dependency-manifest changes are suspicious only for a narrow task that
    // named concrete files. Broad feature/integration work routinely needs a
    // client library or runtime package even when the user did not prescribe
    // the implementation detail. Challenging that natural change caused a
    // completed turn to re-enter the repair loop after dozens of inspections.
    if contract.referenced_paths.is_empty() {
        return None;
    }
    if prompt_mentions_dependencies(prompt, contract) {
        return None;
    }
    let manifests: Vec<&str> = changes
        .iter()
        .filter(|change| {
            change.kind == FileChangeKind::Modify && is_dependency_manifest(&change.path)
        })
        .map(|change| change.path.as_str())
        .collect();
    if manifests.is_empty() {
        return None;
    }
    Some(HygieneFinding {
        reason: format!(
            "modified dependency manifest(s) {} without the task asking to add a dependency",
            manifests.join(", ")
        ),
    })
}

fn prompt_mentions_dependencies(prompt: &str, contract: &TaskContract) -> bool {
    let mut blob = prompt.to_ascii_lowercase();
    for line in &contract.acceptance_text {
        blob.push(' ');
        blob.push_str(&line.to_ascii_lowercase());
    }
    [
        "dependenc",
        "add crate",
        "add a crate",
        "add package",
        "add a package",
        "install ",
        "cargo add",
        "npm install",
        "pnpm add",
        "yarn add",
        "pip install",
        "go get",
        "requirements",
    ]
    .iter()
    .any(|needle| blob.contains(needle))
}

fn is_dependency_manifest(path: &str) -> bool {
    let name = path.replace('\\', "/");
    let file = name.rsplit('/').next().unwrap_or(&name);
    matches!(
        file,
        "Cargo.toml"
            | "Cargo.lock"
            | "package.json"
            | "package-lock.json"
            | "pnpm-lock.yaml"
            | "yarn.lock"
            | "pyproject.toml"
            | "go.mod"
            | "go.sum"
            | "requirements.txt"
    )
}

/// Bytes written or removed this turn. `None` when the baseline is unknown
/// (a Modify without `before_len`) — those are not flagged, because an
/// already-large file with a small patch is not a rewrite.
fn rewrite_delta(change: &FileChange) -> Option<u64> {
    match change.kind {
        FileChangeKind::Create => change.after_len,
        FileChangeKind::Modify => {
            let after = change.after_len?;
            Some(after.abs_diff(change.before_len?))
        }
        FileChangeKind::Delete => None,
    }
}

fn oversized_file(changes: &[FileChange]) -> Option<HygieneFinding> {
    changes.iter().find_map(|change| {
        let delta = rewrite_delta(change).filter(|&delta| delta > LARGE_FILE_BYTES)?;
        Some(HygieneFinding {
            reason: format!(
                "{} changed by {} bytes this turn (file is {} bytes) — prefer edit/patch over rewriting large files",
                change.path,
                delta,
                change.after_len.unwrap_or(0)
            ),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VerificationMode;

    fn change(path: &str, kind: FileChangeKind, after_len: Option<u64>) -> FileChange {
        change_with_before(path, kind, None, after_len)
    }

    fn change_with_before(
        path: &str,
        kind: FileChangeKind,
        before_len: Option<u64>,
        after_len: Option<u64>,
    ) -> FileChange {
        FileChange {
            path: path.into(),
            kind,
            before_digest: None,
            after_digest: None,
            before_len,
            after_len,
            before_mode: None,
            after_mode: None,
        }
    }

    #[test]
    fn unreferenced_creates_need_a_narrow_contract() {
        let contract = TaskContract::derive("fix src/parser.rs", VerificationMode::Auto);
        assert!(!contract.referenced_paths.is_empty());
        let changes = vec![
            change("src/parser.rs", FileChangeKind::Modify, Some(100)),
            change("a.rs", FileChangeKind::Create, Some(10)),
            change("b.rs", FileChangeKind::Create, Some(10)),
            change("c.rs", FileChangeKind::Create, Some(10)),
        ];
        let findings = assess(&contract, &changes, "fix src/parser.rs");
        assert!(
            findings.iter().any(|f| f.reason.contains("unreferenced")),
            "{findings:?}"
        );
        let broad = TaskContract::derive("implement the feature", VerificationMode::Auto);
        assert!(broad.referenced_paths.is_empty());
        assert!(
            assess(&broad, &changes, "implement the feature")
                .iter()
                .all(|f| !f.reason.contains("unreferenced"))
        );
    }

    #[test]
    fn dependency_manifest_is_flagged_unless_asked() {
        let contract = TaskContract::derive("fix the parser in src/lib.rs", VerificationMode::Auto);
        let changes = vec![change("Cargo.toml", FileChangeKind::Modify, Some(200))];
        let findings = assess(&contract, &changes, "fix the parser in src/lib.rs");
        assert!(
            findings
                .iter()
                .any(|f| f.reason.contains("dependency manifest")),
            "{findings:?}"
        );
        let asked = assess(
            &contract,
            &changes,
            "add crate serde to Cargo.toml for the parser",
        );
        assert!(
            asked
                .iter()
                .all(|f| !f.reason.contains("dependency manifest")),
            "{asked:?}"
        );

        let broad = TaskContract::derive(
            "connect the app to the inference API and make it work",
            VerificationMode::Auto,
        );
        assert!(broad.referenced_paths.is_empty());
        let broad_findings = assess(
            &broad,
            &changes,
            "connect the app to the inference API and make it work",
        );
        assert!(
            broad_findings
                .iter()
                .all(|f| !f.reason.contains("dependency manifest")),
            "broad integration work may naturally require a dependency: {broad_findings:?}"
        );
    }

    #[test]
    fn oversized_create_is_flagged() {
        let contract = TaskContract::derive("add a helper", VerificationMode::Auto);
        let changes = vec![change(
            "src/huge.rs",
            FileChangeKind::Create,
            Some(LARGE_FILE_BYTES + 1),
        )];
        let findings = assess(&contract, &changes, "add a helper");
        assert!(
            findings.iter().any(|f| f.reason.contains("bytes")),
            "{findings:?}"
        );
    }

    #[test]
    fn small_patch_on_already_large_file_is_not_flagged() {
        let contract =
            TaskContract::derive("fold stream_area into the Run row", VerificationMode::Auto);
        let changes = vec![change_with_before(
            "crates/hi-tui/src/app/render.rs",
            FileChangeKind::Modify,
            Some(113_013),
            Some(113_050),
        )];
        let findings = assess(&contract, &changes, "fold stream_area into the Run row");
        assert!(
            findings
                .iter()
                .all(|f| !f.reason.contains("rewriting large files")),
            "small delta on a large file must not look like a rewrite: {findings:?}"
        );
    }

    #[test]
    fn modify_without_before_len_is_not_flagged() {
        let contract = TaskContract::derive("edit render.rs", VerificationMode::Auto);
        let changes = vec![change(
            "crates/hi-tui/src/app/render.rs",
            FileChangeKind::Modify,
            Some(LARGE_FILE_BYTES + 1),
        )];
        let findings = assess(&contract, &changes, "edit render.rs");
        assert!(
            findings
                .iter()
                .all(|f| !f.reason.contains("rewriting large files")),
            "unknown baseline must not flag an already-large file: {findings:?}"
        );
    }

    #[test]
    fn large_growth_on_modify_is_flagged() {
        let contract = TaskContract::derive("rewrite the renderer", VerificationMode::Auto);
        let changes = vec![change_with_before(
            "src/render.rs",
            FileChangeKind::Modify,
            Some(1_024),
            Some(LARGE_FILE_BYTES + 2_048),
        )];
        let findings = assess(&contract, &changes, "rewrite the renderer");
        assert!(
            findings.iter().any(|f| f.reason.contains("changed by")),
            "{findings:?}"
        );
    }
}
