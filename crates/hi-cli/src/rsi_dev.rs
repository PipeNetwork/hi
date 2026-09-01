//! Optional `hi rsi up|down` wrapper around the ipop laptop loopback scripts.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, anyhow, bail};

pub(crate) fn run(args: &[String]) -> Result<()> {
    let action = args.first().map(String::as_str).unwrap_or("help");
    match action {
        "up" => exec_script("rsi-dev-up.sh"),
        "down" => exec_script("rsi-dev-down.sh"),
        _ => {
            eprintln!(
                "usage: hi rsi <up|down>\n\nStarts or stops the laptop RSI loopback (Postgres, rsi-api/trust, unsandboxed rsi-hi-worker, public API with IPOP_TASKS_USE_RSI=1).\nSet IPOP_ROOT to the ipop checkout, or run the scripts directly."
            );
            if matches!(action, "help" | "--help" | "-h") {
                Ok(())
            } else {
                bail!("unknown rsi subcommand {action}");
            }
        }
    }
}

fn exec_script(name: &str) -> Result<()> {
    let script = script_path(name)?;
    let status = Command::new("bash")
        .arg(&script)
        .status()
        .map_err(|error| anyhow!("running {}: {error}", script.display()))?;
    if status.success() {
        Ok(())
    } else {
        bail!("{} exited {}", script.display(), status);
    }
}

fn script_path(name: &str) -> Result<PathBuf> {
    if let Ok(root) = std::env::var("IPOP_ROOT") {
        let path = Path::new(&root).join("scripts").join(name);
        if path.is_file() {
            return Ok(path);
        }
        bail!(
            "IPOP_ROOT is set but {} is missing. Expected {}",
            name,
            path.display()
        );
    }
    let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ipop/scripts")
        .join(name);
    if bundled.is_file() {
        return Ok(bundled);
    }
    bail!("set IPOP_ROOT to the ipop checkout, then run:\n  $IPOP_ROOT/scripts/{name}");
}
