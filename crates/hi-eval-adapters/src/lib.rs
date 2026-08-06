//! Third-party evaluation source adapters.
//!
//! The initial adapter boundary accepts normalized task-package directories
//! for every supported route. This makes the route catalog useful immediately
//! while individual upstream readers are added without coupling them to the
//! evaluator or runtime backend.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hi_eval::{CasePlan, ClaimLevel, DatasetPlan, SourceIdentity};

pub const ADAPTER_API_VERSION: &str = "hi-eval-adapters/1";

/// Route metadata keeps benchmark-specific claim and topology caveats out of
/// the generic runner while still making them visible to manifests and tools.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterRouteSpec {
    pub id: String,
    pub default_claim_level: ClaimLevel,
    pub input_mode: &'static str,
    pub output_mode: &'static str,
    pub requires_external_harness: bool,
    pub gated_data: bool,
}

pub fn route_spec(route: &str) -> Option<AdapterRouteSpec> {
    let (input_mode, output_mode, requires_external_harness, gated_data, default_claim_level) =
        match route {
            "legacy" | "terminal-bench" | "harbor" | "deepswe" | "stablebench"
            | "frontier-bench" | "swe-bench" | "genebench-pro" | "swe-atlas-qna"
            | "agents-last-exam" => ("prompt", "workspace", false, false, ClaimLevel::Smoke),
            "arena-hard"
            | "openai-evals-match"
            | "openai-evals-includes"
            | "gpqa-diamond"
            | "browsecomp" => ("prompt", "final_message", false, false, ClaimLevel::Smoke),
            "graphwalks" | "mrcr" | "healthbench-professional" => (
                "transcript",
                "final_message",
                true,
                false,
                ClaimLevel::PublicReproduction,
            ),
            "gdpval" => (
                "prompt",
                "workspace",
                true,
                false,
                ClaimLevel::PublicReproduction,
            ),
            "arc-agi-3" => (
                "interactive",
                "actions",
                true,
                false,
                ClaimLevel::EvidenceOnly,
            ),
            "external" => (
                "prompt_or_transcript",
                "workspace_or_final_message",
                true,
                false,
                ClaimLevel::Smoke,
            ),
            _ => return None,
        };
    Some(AdapterRouteSpec {
        id: route.to_string(),
        default_claim_level,
        input_mode,
        output_mode,
        requires_external_harness,
        gated_data,
    })
}

pub const UNSUPPORTED_OPENAI_EVAL_CLASSES: &[&str] =
    &["modelgraded_closedqa", "cotqa", "web_of_lies"];

/// Current adapter route names.
pub const SUPPORTED_ROUTES: &[&str] = &[
    "legacy",
    "external",
    "harbor",
    "terminal-bench",
    "deepswe",
    "stablebench",
    "frontier-bench",
    "arena-hard",
    "openai-evals-match",
    "openai-evals-includes",
    "swe-bench",
    "genebench-pro",
    "graphwalks",
    "mrcr",
    "healthbench-professional",
    "gdpval",
    "swe-atlas-qna",
    "gpqa-diamond",
    "browsecomp",
    "arc-agi-3",
    "agents-last-exam",
];

/// Adapter output for a source that already contains complete task packages.
#[derive(Clone, Debug)]
pub struct DirectoryAdapter {
    pub name: String,
    pub route: String,
    pub source: PathBuf,
    pub revision: String,
    pub claim_level: ClaimLevel,
}

impl DirectoryAdapter {
    pub fn plan(&self) -> Result<DatasetPlan> {
        if route_spec(&self.route).is_none() {
            bail!("unsupported evaluation adapter route {:?}", self.route);
        }
        if fs::symlink_metadata(&self.source)
            .with_context(|| format!("reading adapter source {}", self.source.display()))?
            .file_type()
            .is_symlink()
        {
            bail!(
                "adapter source must not be a symlink: {}",
                self.source.display()
            );
        }
        let source = self.source.canonicalize().with_context(|| {
            format!(
                "resolving {} adapter source {}",
                self.route,
                self.source.display()
            )
        })?;
        let mut cases = Vec::new();
        if is_task_package(&source) {
            cases.push(CasePlan::new(case_id(&source)?, source.clone()));
        } else {
            discover_cases(&source, &mut cases)?;
        }
        if cases.is_empty() {
            bail!(
                "{} adapter found no task.toml, package.toml, or package.json cases under {}",
                self.route,
                self.source.display()
            );
        }
        Ok(DatasetPlan::new(
            &self.name,
            SourceIdentity {
                kind: self.route.clone(),
                revision: self.revision.clone(),
                digest: digest_tree(&source)?,
            },
        )
        .with_cases(cases)
        .with_claim_level(self.claim_level))
    }
}

/// Build a plan for any catalog route from a directory of normalized cases.
pub fn plan_directory(
    name: impl Into<String>,
    route: impl Into<String>,
    source: impl Into<PathBuf>,
    revision: impl Into<String>,
    claim_level: ClaimLevel,
) -> Result<DatasetPlan> {
    DirectoryAdapter {
        name: name.into(),
        route: route.into(),
        source: source.into(),
        revision: revision.into(),
        claim_level,
    }
    .plan()
}

fn discover_cases(root: &Path, cases: &mut Vec<CasePlan>) -> Result<()> {
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            bail!("adapter source contains a symlink: {}", path.display());
        }
        if !path.is_dir() {
            continue;
        }
        if is_task_package(&path) {
            cases.push(CasePlan::new(case_id(&path)?, path));
        } else {
            discover_cases(&path, cases)?;
        }
    }
    Ok(())
}

fn is_task_package(path: &Path) -> bool {
    ["task.toml", "package.toml", "package.json"]
        .iter()
        .any(|name| path.join(name).is_file())
}

fn case_id(path: &Path) -> Result<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty() && name != "." && name != "..")
        .context("task package has no usable directory name")
}

fn digest_tree(path: &Path) -> Result<String> {
    let mut entries = fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    let mut bytes = Vec::new();
    for entry in entries {
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            bail!(
                "adapter source contains a symlink: {}",
                entry.path().display()
            );
        }
        bytes.push(if metadata.is_dir() { b'D' } else { b'F' });
        bytes.extend_from_slice(entry.file_name().to_string_lossy().as_bytes());
        bytes.push(0);
        if metadata.is_dir() {
            bytes.extend_from_slice(digest_tree(&entry.path())?.as_bytes());
        } else if metadata.is_file() {
            bytes.extend_from_slice(&fs::read(entry.path())?);
        } else {
            bail!(
                "adapter source contains unsupported node: {}",
                entry.path().display()
            );
        }
        bytes.push(0xff);
    }
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_adapter_discovers_nested_cases() {
        let root = std::env::temp_dir().join(format!("hi-eval-adapters-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("suite/case-a/fixture")).unwrap();
        fs::write(
            root.join("suite/case-a/task.toml"),
            "schema_version = 2\nprompt = 'x'\nallowed_changes = ['**']\n[final_oracle]\ncommand = 'true'\n",
        )
        .unwrap();
        let plan = plan_directory(
            "suite",
            "external",
            root.join("suite"),
            "test@1",
            ClaimLevel::Smoke,
        )
        .unwrap();
        assert_eq!(plan.cases.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn directory_adapter_discovers_json_task_packages() {
        let root =
            std::env::temp_dir().join(format!("hi-eval-adapters-json-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("suite/case-json")).unwrap();
        fs::write(root.join("suite/case-json/package.json"), "{}").unwrap();
        let plan = plan_directory(
            "suite",
            "external",
            root.join("suite"),
            "test@1",
            ClaimLevel::Smoke,
        )
        .unwrap();
        assert_eq!(plan.cases.len(), 1);
        assert_eq!(plan.cases[0].id, "case-json");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn route_catalog_labels_transcripts_and_gated_judges() {
        let route = route_spec("mrcr").unwrap();
        assert_eq!(route.input_mode, "transcript");
        assert!(route.requires_external_harness);
        assert_eq!(route.default_claim_level, ClaimLevel::PublicReproduction);
        assert!(route_spec("not-a-route").is_none());
    }
}
