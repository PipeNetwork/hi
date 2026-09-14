//! Session-policy types previously owned by the old agent crate.

use serde::{Deserialize, Serialize};

/// Sentinel for "no repair ceiling" in quality settings.
pub const UNLIMITED_REPAIR_CYCLES: u32 = u32::MAX;

/// One stage of layered verification: a short label and the shell command to
/// run. Stages run in order; the first to fail stops the turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyStage {
    pub name: String,
    pub command: String,
}

impl VerifyStage {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
        }
    }
}

/// How deterministic verification is selected for a turn.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "stages", rename_all = "snake_case")]
pub enum VerificationMode {
    /// Detect a project-appropriate pipeline from the workspace.
    #[default]
    Auto,
    /// Run exactly these stages, in order.
    Explicit(Vec<VerifyStage>),
    /// Do not run deterministic verification.
    Disabled,
}

impl VerificationMode {
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Self::Explicit(stages) = self {
            anyhow::ensure!(
                !stages.is_empty() && stages.iter().all(|stage| !stage.command.trim().is_empty()),
                "explicit verification requires non-empty command stages"
            );
        }
        Ok(())
    }

    pub fn resolved_stages(&self, root: &std::path::Path) -> Vec<VerifyStage> {
        match self {
            Self::Auto => detect_verify_pipeline(root),
            Self::Explicit(stages) => stages.clone(),
            Self::Disabled => Vec::new(),
        }
    }
}

/// Post-mutation completion-review policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPolicy {
    #[default]
    Risk,
    Always,
    Off,
}

impl ReviewPolicy {
    pub fn label(self) -> &'static str {
        match self {
            ReviewPolicy::Risk => "risk",
            ReviewPolicy::Always => "always",
            ReviewPolicy::Off => "off",
        }
    }
}

/// Workspace-local language-server policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LspMode {
    #[default]
    Auto,
    On,
    Off,
}

impl LspMode {
    pub fn label(self) -> &'static str {
        match self {
            LspMode::Auto => "auto",
            LspMode::On => "on",
            LspMode::Off => "off",
        }
    }
}

/// When the write-capable `delegate` subagent is advertised.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteSubagentPolicy {
    Off,
    #[default]
    Risk,
    On,
}

impl WriteSubagentPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Risk => "risk",
            Self::On => "on",
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Tool advertisement policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSet {
    #[default]
    Dynamic,
    Minimal,
    Full,
}

impl ToolSet {
    pub fn label(self) -> &'static str {
        match self {
            ToolSet::Dynamic => "dynamic",
            ToolSet::Minimal => "minimal",
            ToolSet::Full => "full",
        }
    }
}

/// Whether task progress is checkpointed at durable execution boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    #[default]
    Ephemeral,
    Durable,
}

impl ExecutionMode {
    pub fn is_durable(self) -> bool {
        matches!(self, Self::Durable)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ephemeral => "ephemeral",
            Self::Durable => "durable",
        }
    }
}

/// Guess a layered deterministic verification pipeline from marker files.
pub fn detect_verify_pipeline(dir: &std::path::Path) -> Vec<VerifyStage> {
    detect_verify_pipeline_with(dir, false)
}

/// Like [`detect_verify_pipeline`], optionally inserting a clippy stage for
/// Cargo workspaces.
pub fn detect_verify_pipeline_with(dir: &std::path::Path, clippy: bool) -> Vec<VerifyStage> {
    let has = |name: &str| dir.join(name).exists();
    let stage = |name: &str, command: &str| VerifyStage::new(name, command);
    if has("Cargo.toml") {
        let mut stages = vec![stage("check", "cargo check --quiet")];
        if clippy {
            stages.push(stage("clippy", "cargo clippy --quiet --all-targets"));
        }
        stages.push(stage("test", "cargo test --quiet"));
        stages
    } else if has("go.mod") {
        vec![
            stage("build", "go build ./..."),
            stage("test", "go test ./..."),
        ]
    } else if has("package.json") {
        javascript_pipeline(dir)
    } else if has("pyproject.toml") || has("setup.py") || has("pytest.ini") || has("tox.ini") {
        let mut stages = Vec::new();
        if has("ruff.toml") || has(".ruff.toml") {
            stages.push(stage("lint", "ruff check ."));
        }
        if has_python_tests(dir) {
            stages.push(stage("test", "pytest -q"));
        }
        stages
    } else {
        makefile_pipeline(dir).unwrap_or_default()
    }
}

fn javascript_pipeline(dir: &std::path::Path) -> Vec<VerifyStage> {
    let runner = if dir.join("pnpm-lock.yaml").exists() {
        "pnpm"
    } else if dir.join("yarn.lock").exists() {
        "yarn"
    } else if dir.join("bun.lockb").exists() || dir.join("bun.lock").exists() {
        "bun"
    } else {
        "npm"
    };
    let scripts: std::collections::BTreeSet<String> =
        std::fs::read_to_string(dir.join("package.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|package| {
                package.get("scripts").and_then(|scripts| {
                    scripts.as_object().map(|map| map.keys().cloned().collect())
                })
            })
            .unwrap_or_default();
    let mut stages = Vec::new();
    if dir.join("tsconfig.json").exists() {
        stages.push(VerifyStage::new(
            "typecheck",
            "npx --no-install tsc --noEmit",
        ));
    } else if scripts.contains("typecheck") {
        stages.push(VerifyStage::new(
            "typecheck",
            format!("{runner} run typecheck"),
        ));
    }
    if scripts.contains("lint") {
        stages.push(VerifyStage::new("lint", format!("{runner} run lint")));
    }
    if scripts.contains("test") {
        let command = match runner {
            "npm" => "npm test --silent".to_string(),
            other => format!("{other} test"),
        };
        stages.push(VerifyStage::new("test", command));
    }
    stages
}

fn makefile_pipeline(dir: &std::path::Path) -> Option<Vec<VerifyStage>> {
    let makefile = ["Makefile", "makefile"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())?;
    let text = std::fs::read_to_string(makefile).ok()?;
    let has_target = |target: &str| {
        text.lines().any(|line| {
            line.strip_prefix(target)
                .is_some_and(|rest| rest.starts_with(':') && !rest.starts_with("::="))
        })
    };
    let mut stages = Vec::new();
    if has_target("check") {
        stages.push(VerifyStage::new("check", "make check"));
    }
    if has_target("test") {
        stages.push(VerifyStage::new("test", "make test"));
    }
    (!stages.is_empty()).then_some(stages)
}

fn has_python_tests(package_root: &std::path::Path) -> bool {
    fn is_test_file(name: &str) -> bool {
        (name.starts_with("test_") || name.ends_with("_test.py")) && name.ends_with(".py")
    }
    fn walk(dir: &std::path::Path) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(
                    name,
                    "__pycache__"
                        | ".venv"
                        | ".cargo-home"
                        | "venv"
                        | "node_modules"
                        | "dist"
                        | "build"
                        | ".git"
                        | ".hg"
                        | ".svn"
                        | ".jj"
                        | ".tox"
                        | ".mypy_cache"
                        | ".pytest_cache"
                        | ".ruff_cache"
                ) {
                    continue;
                }
                if walk(&path) {
                    return true;
                }
            } else if file_type.is_file()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                && is_test_file(name)
            {
                return true;
            }
        }
        false
    }
    walk(package_root)
}
