use std::path::{Component, Path};

use anyhow::{Result, bail};
use globset::Glob;
use serde::{Deserialize, Serialize};

pub const TASK_SCHEMA_VERSION: u32 = 2;
pub const DEFAULT_CANDIDATE_TIMEOUT_SECONDS: u64 = 900;
pub const DEFAULT_CHECK_TIMEOUT_SECONDS: u64 = 120;

/// A benchmark task whose final score is decided by an external, immutable
/// oracle. Only `fixture/` is exposed to the candidate; `task.toml` and the
/// optional oracle bundle remain outside its workspace.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub schema_version: u32,
    pub name: Option<String>,
    pub prompt: String,
    /// Workspace-relative globs for files the candidate is permitted to change.
    pub allowed_changes: Vec<String>,
    /// A check that may be shown to the candidate and used for repair feedback.
    /// It is never the source of the final benchmark score.
    #[serde(default)]
    pub visible_feedback: Option<CheckSpec>,
    /// The scorer captured before candidate launch and injected only into a
    /// fresh post-attempt verification copy.
    pub final_oracle: FinalOracle,
    #[serde(default)]
    pub workspace: WorkspaceSpec,
    #[serde(default)]
    pub timeouts: TaskTimeouts,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct CheckSpec {
    pub command: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FinalOracle {
    pub command: String,
    /// Optional task-directory-relative bundle. Its bytes are captured before
    /// the candidate starts and restored as `.hi-eval-oracle/` for scoring.
    #[serde(default)]
    pub bundle: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceSpec {
    #[serde(default)]
    pub kind: WorkspaceKind,
    /// For `dirty_git`, a tracked file rewritten after the fixture commit.
    pub dirty_path: Option<String>,
    pub dirty_contents: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Copy, Default, PartialEq, Eq)]
pub enum WorkspaceKind {
    #[default]
    #[serde(rename = "non_git")]
    Plain,
    #[serde(rename = "clean_git")]
    CleanRepository,
    #[serde(rename = "dirty_git")]
    DirtyRepository,
}

#[derive(Debug, Deserialize, Clone, Copy, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskTimeouts {
    #[serde(default = "default_candidate_timeout")]
    pub candidate_seconds: u64,
    #[serde(default = "default_feedback_timeout")]
    pub visible_feedback_seconds: u64,
    #[serde(default = "default_oracle_timeout")]
    pub oracle_seconds: u64,
}

impl Default for TaskTimeouts {
    fn default() -> Self {
        Self {
            candidate_seconds: DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
            visible_feedback_seconds: DEFAULT_CHECK_TIMEOUT_SECONDS,
            oracle_seconds: DEFAULT_CHECK_TIMEOUT_SECONDS,
        }
    }
}

const fn default_candidate_timeout() -> u64 {
    DEFAULT_CANDIDATE_TIMEOUT_SECONDS
}

const fn default_feedback_timeout() -> u64 {
    DEFAULT_CHECK_TIMEOUT_SECONDS
}

const fn default_oracle_timeout() -> u64 {
    DEFAULT_CHECK_TIMEOUT_SECONDS
}

impl Task {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TASK_SCHEMA_VERSION {
            bail!(
                "unsupported task schema {}; expected {}",
                self.schema_version,
                TASK_SCHEMA_VERSION
            );
        }
        if self.prompt.trim().is_empty() {
            bail!("task prompt must not be empty");
        }
        if self.allowed_changes.is_empty() {
            bail!("allowed_changes must contain at least one glob");
        }
        for pattern in &self.allowed_changes {
            Glob::new(pattern).map_err(|err| {
                anyhow::anyhow!("invalid allowed_changes glob {pattern:?}: {err}")
            })?;
        }
        if self.final_oracle.command.trim().is_empty() {
            bail!("final_oracle.command must not be empty");
        }
        if let Some(feedback) = &self.visible_feedback
            && feedback.command.trim().is_empty()
        {
            bail!("visible_feedback.command must not be empty");
        }
        if let Some(bundle) = &self.final_oracle.bundle {
            let path = Path::new(bundle);
            if path.as_os_str().is_empty()
                || path.is_absolute()
                || path.components().any(|component| {
                    matches!(
                        component,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                bail!("final_oracle.bundle must be a contained relative path");
            }
        }
        match self.workspace.kind {
            WorkspaceKind::DirtyRepository => {
                let path =
                    self.workspace.dirty_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("dirty_git requires workspace.dirty_path")
                    })?;
                validate_relative_path(Path::new(path), "workspace.dirty_path")?;
                if self.workspace.dirty_contents.is_none() {
                    bail!("dirty_git requires workspace.dirty_contents");
                }
            }
            WorkspaceKind::Plain | WorkspaceKind::CleanRepository => {
                if self.workspace.dirty_path.is_some() || self.workspace.dirty_contents.is_some() {
                    bail!("workspace dirty fields require kind = 'dirty_git'");
                }
            }
        }
        if self.timeouts.candidate_seconds == 0
            || self.timeouts.visible_feedback_seconds == 0
            || self.timeouts.oracle_seconds == 0
        {
            bail!("task timeouts must be greater than zero");
        }
        Ok(())
    }
}

fn validate_relative_path(path: &Path, label: &str) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!("{label} must be a contained relative path");
    }
    Ok(())
}

/// One way of running `hi` against the tasks.
pub struct Config {
    pub name: &'static str,
    /// Use explicit visible feedback when present, otherwise keep the CLI's
    /// automatic verification pipeline enabled. False passes are still decided
    /// solely by the external final oracle.
    pub use_verify: bool,
    /// One sampling temperature per candidate. The config solves the task if
    /// ANY candidate passes — execution-grounded best-of-N (the test suite is
    /// the judge). Tokens are summed across all candidates.
    pub temperatures: &'static [f32],
    /// Extra environment variables set on every child `hi` run — how a config
    /// turns on an orchestration knob (e.g. the `/goal team` skeptic gate via
    /// `HI_GOAL_TEAM` + `HI_SKEPTIC_MODEL`) without any other difference. This is
    /// what lets a config pair be an honest A/B of one lever.
    pub env: &'static [(&'static str, &'static str)],
}

pub const CONFIGS: &[Config] = &[
    Config {
        name: "baseline",
        use_verify: false,
        temperatures: &[0.0],
        env: &[],
    },
    Config {
        name: "verify",
        use_verify: true,
        temperatures: &[0.0],
        env: &[],
    },
    Config {
        name: "best-of-3",
        use_verify: true,
        temperatures: &[0.2, 0.7, 1.0],
        env: &[],
    },
    // The skeptic-gate A/B: identical to `baseline` (no --verify, one candidate)
    // except the `/goal team` gate is on. Run BOTH in goal mode
    // (`HI_EVAL_GOAL=1 HI_EVAL_TURNS=N`) and compare `baseline` vs `goal-team` —
    // the only difference is the reviewer, so a pass-rate delta is its value and
    // the token delta is its cost.
    Config {
        name: "goal-team",
        use_verify: false,
        temperatures: &[0.0],
        env: &[
            ("HI_GOAL_TEAM", "1"),
            ("HI_SKEPTIC_MODEL", "pipe/glm-5.2-fast"),
        ],
    },
];

#[cfg(test)]
mod task_tests {
    use super::{DEFAULT_CANDIDATE_TIMEOUT_SECONDS, DEFAULT_CHECK_TIMEOUT_SECONDS, Task};

    #[test]
    fn schema_v2_defaults_timeouts_and_optional_feedback() {
        let task: Task = toml::from_str(
            r#"
schema_version = 2
prompt = "fix it"
allowed_changes = ["src/**"]

[final_oracle]
command = "cargo test"
"#,
        )
        .unwrap();
        task.validate().unwrap();
        assert!(task.visible_feedback.is_none());
        assert_eq!(
            task.timeouts.candidate_seconds,
            DEFAULT_CANDIDATE_TIMEOUT_SECONDS
        );
        assert_eq!(task.timeouts.oracle_seconds, DEFAULT_CHECK_TIMEOUT_SECONDS);
    }

    #[test]
    fn rejects_v1_bad_globs_and_zero_timeouts() {
        let old: Task = toml::from_str(
            r#"
schema_version = 1
prompt = "fix it"
allowed_changes = ["src/**"]
[final_oracle]
command = "true"
"#,
        )
        .unwrap();
        assert!(old.validate().is_err());

        let invalid: Task = toml::from_str(
            r#"
schema_version = 2
prompt = "fix it"
allowed_changes = ["["]
[timeouts]
candidate_seconds = 0
visible_feedback_seconds = 1
oracle_seconds = 1
[final_oracle]
command = "true"
"#,
        )
        .unwrap();
        assert!(invalid.validate().is_err());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalProfile {
    Default,
    Pipenetwork,
    PipenetworkMcp,
}

impl EvalProfile {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("default") {
            "default" => Ok(Self::Default),
            "pipenetwork" => Ok(Self::Pipenetwork),
            "pipenetwork-mcp" => Ok(Self::PipenetworkMcp),
            other => {
                bail!("unknown --profile={other}; known: default, pipenetwork, pipenetwork-mcp")
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Pipenetwork => "pipenetwork",
            Self::PipenetworkMcp => "pipenetwork-mcp",
        }
    }

    pub fn hi_args(self) -> &'static [&'static str] {
        match self {
            Self::Default => &[],
            Self::Pipenetwork | Self::PipenetworkMcp => &[
                "--provider",
                "pipenetwork",
                "--compat",
                "auto",
                "--tool-mode",
                "auto",
            ],
        }
    }

    pub fn validate_env(self) -> Result<()> {
        if matches!(self, Self::Pipenetwork | Self::PipenetworkMcp)
            && std::env::var("PIPENETWORK_API_KEY").is_err()
            && std::env::var("HI_API_KEY").is_err()
        {
            bail!(
                "--profile={} requires PIPENETWORK_API_KEY or HI_API_KEY",
                self.label()
            );
        }
        Ok(())
    }

    pub fn uses_mcp_metadata(self) -> bool {
        matches!(self, Self::PipenetworkMcp)
    }
}
