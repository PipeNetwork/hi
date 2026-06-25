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
    /// the judge). Cost/tokens are summed across all candidates.
    pub temperatures: &'static [f32],
}

pub const CONFIGS: &[Config] = &[
    Config {
        name: "baseline",
        use_verify: false,
        temperatures: &[0.0],
    },
    Config {
        name: "verify",
        use_verify: true,
        temperatures: &[0.0],
    },
    Config {
        name: "best-of-3",
        use_verify: true,
        temperatures: &[0.2, 0.7, 1.0],
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalProfile {
    Default,
    Terminaili,
}

impl EvalProfile {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("default") {
            "default" => Ok(Self::Default),
            "terminaili" => Ok(Self::Terminaili),
            other => bail!("unknown --profile={other}; known: default, terminaili"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Terminaili => "terminaili",
        }
    }

    pub fn hi_args(self) -> &'static [&'static str] {
        match self {
            Self::Default => &[],
            Self::Terminaili => &[
                "--provider",
                "terminaili",
                "--compat",
                "auto",
                "--tool-mode",
                "auto",
            ],
        }
    }

    pub fn validate_env(self) -> Result<()> {
        if matches!(self, Self::Terminaili)
            && std::env::var("TERMINAILI_API_KEY").is_err()
            && std::env::var("HI_API_KEY").is_err()
        {
            bail!("--profile=terminaili requires TERMINAILI_API_KEY or HI_API_KEY");
        }
        Ok(())
    }
}
