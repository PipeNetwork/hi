use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use crate::{discovery, fuzz, live_baseline, runner};

#[derive(Parser, Debug)]
#[command(
    name = "hi-smoke",
    version,
    about = "Drive the real hi TUI through a PTY and assert lifecycle invariants"
)]
pub(crate) struct Cli {
    /// Path to the hi executable. Falls back to HI_BIN, then target/debug/hi.
    #[arg(long, global = true, value_name = "PATH")]
    hi_bin: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Parse and validate every scenario without launching hi.
    Validate {
        #[arg(value_name = "SUITE")]
        suite: PathBuf,
    },
    /// Run curated scenarios.
    Run {
        #[arg(value_name = "SUITE")]
        suite: PathBuf,
        #[arg(long, value_enum, default_value_t = RunMode::Scripted)]
        mode: RunMode,
        #[arg(long, value_name = "TAG")]
        tag: Vec<String>,
        #[arg(long, default_value = "artifacts/tui-smoke")]
        artifacts: PathBuf,
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        #[arg(long)]
        keep: bool,
    },
    /// Run deterministic, state-aware fault combinations.
    Fuzz {
        #[arg(value_name = "SUITE")]
        suite: PathBuf,
        #[arg(long, default_value_t = 0)]
        seed_start: u64,
        #[arg(long, default_value_t = 1)]
        seeds: u64,
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        #[arg(long, default_value = "artifacts/tui-smoke-fuzz")]
        artifacts: PathBuf,
        #[arg(long)]
        keep: bool,
    },
    /// Re-run the exact normalized scenario saved in a failure bundle.
    Replay {
        #[arg(value_name = "REPLAY")]
        replay: PathBuf,
        #[arg(long, default_value = "artifacts/tui-smoke-replay")]
        artifacts: PathBuf,
        #[arg(long)]
        keep: bool,
    },
    /// Evaluate a live-canary summary against the reviewed live-only baseline.
    CheckLiveBaseline {
        #[arg(value_name = "SUMMARY")]
        summary: PathBuf,
        #[arg(value_name = "BASELINE")]
        baseline: PathBuf,
        /// Persist distinct successful nightly run IDs while the baseline is observing.
        #[arg(long, value_name = "PATH", requires = "nightly_run_id")]
        observation_state: Option<PathBuf>,
        /// Stable identifier for this nightly run (for example, GitHub's run ID).
        #[arg(long, value_name = "ID", requires = "observation_state")]
        nightly_run_id: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub(crate) enum RunMode {
    #[default]
    Scripted,
    Live,
}

impl Cli {
    pub(crate) fn execute(self) -> Result<()> {
        match self.command {
            Command::Validate { suite } => {
                let found = discovery::discover(&suite)?;
                println!("validated {} TUI smoke scenario(s)", found.len());
                Ok(())
            }
            Command::Run {
                suite,
                mode,
                tag,
                artifacts,
                jobs,
                keep,
            } => {
                ensure_jobs(jobs)?;
                let hi_bin = resolve_hi_bin(self.hi_bin.as_deref())?;
                runner::run_suite(runner::SuiteOptions {
                    hi_bin,
                    suite,
                    mode,
                    tags: tag,
                    artifacts,
                    jobs,
                    keep,
                })
            }
            Command::Fuzz {
                suite,
                seed_start,
                seeds,
                jobs,
                artifacts,
                keep,
            } => {
                ensure_jobs(jobs)?;
                if seeds == 0 {
                    bail!("--seeds must be greater than zero");
                }
                let hi_bin = resolve_hi_bin(self.hi_bin.as_deref())?;
                fuzz::run(fuzz::FuzzOptions {
                    hi_bin,
                    suite,
                    seed_start,
                    seeds,
                    jobs,
                    artifacts,
                    keep,
                })
            }
            Command::Replay {
                replay,
                artifacts,
                keep,
            } => {
                let hi_bin = resolve_hi_bin(self.hi_bin.as_deref())?;
                runner::replay(&hi_bin, &replay, &artifacts, keep)
            }
            Command::CheckLiveBaseline {
                summary,
                baseline,
                observation_state,
                nightly_run_id,
            } => live_baseline::check(
                &summary,
                &baseline,
                observation_state.as_deref(),
                nightly_run_id.as_deref(),
            ),
        }
    }
}

fn ensure_jobs(jobs: usize) -> Result<()> {
    if jobs == 0 {
        bail!("--jobs must be greater than zero");
    }
    if jobs > 64 {
        bail!("--jobs is capped at 64");
    }
    Ok(())
}

pub(crate) fn resolve_hi_bin(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidate = explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("HI_BIN").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/debug/hi"));
    let candidate = if candidate.is_absolute() {
        candidate
    } else {
        std::env::current_dir()
            .context("determining current directory")?
            .join(candidate)
    };
    if !candidate.is_file() {
        bail!(
            "hi executable not found at {}; build it with `cargo build -p hi` or pass --hi-bin",
            candidate.display()
        );
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_are_bounded() {
        assert!(ensure_jobs(0).is_err());
        assert!(ensure_jobs(1).is_ok());
        assert!(ensure_jobs(64).is_ok());
        assert!(ensure_jobs(65).is_err());
    }

    #[test]
    fn live_observation_state_requires_a_run_id_and_vice_versa() {
        let base = [
            "hi-smoke",
            "check-live-baseline",
            "summary.json",
            "baseline.json",
        ];
        assert!(
            Cli::try_parse_from(
                base.into_iter()
                    .chain(["--observation-state", "state.json"])
            )
            .is_err()
        );
        assert!(Cli::try_parse_from(base.into_iter().chain(["--nightly-run-id", "123"])).is_err());
        assert!(
            Cli::try_parse_from(base.into_iter().chain([
                "--observation-state",
                "state.json",
                "--nightly-run-id",
                "123",
            ]))
            .is_ok()
        );
    }
}
