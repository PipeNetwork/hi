mod artifacts;
mod cli;
mod discovery;
mod fuzz;
mod isolation;
mod live_baseline;
mod live_route;
mod pty;
mod runner;
mod scenario;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    cli::Cli::parse().execute()
}
