use anyhow::{Result, bail};
use serde::Deserialize;

/// A benchmark task: a prompt to run and a command that decides success.
#[derive(Deserialize, Clone)]
pub struct Task {
    pub name: Option<String>,
    pub prompt: String,
    /// Shell command run in the work dir; exit 0 == solved.
    pub verify: String,
}

/// One way of running `hi` against the tasks.
pub struct Config {
    pub name: &'static str,
    /// Pass `--verify <task.verify>` so the agent iterates to green.
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
