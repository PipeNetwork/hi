use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

use crate::config::{EvalProfile, Task};
use crate::results::{RunArtifact, RunResult};

pub fn write_artifact(
    dir: &Path,
    profile: EvalProfile,
    condense: bool,
    recovery: bool,
    result: &RunResult,
) -> Result<()> {
    let artifact = RunArtifact {
        task: result.task.clone(),
        config: result.config.to_string(),
        model: result.model.clone(),
        trial: result.trial,
        profile: profile.label().to_string(),
        condense,
        recovery,
        passed: result.passed,
        failure_bucket: result.fail.map(|f| f.label().to_string()),
        failure_confidence: result.failure_confidence,
        changed_files: result.changed_files.clone(),
        provider_error_kind: result.provider_error_kind.clone(),
        compat_fallbacks_used: result.compat_fallbacks_used.clone(),
        candidates: result.candidates,
        tokens: result.tokens,
        cost_usd: result.cost_usd,
        duration_seconds: result.seconds,
        mcp_model: result.mcp_model.clone(),
        verify_output_summary: result.verify_output_summary.clone(),
        trajectory: result.trajectory.clone(),
    };
    let name = format!(
        "trial-{:03}-{}-{}-{}.json",
        result.trial + 1,
        sanitize_name(&result.config),
        sanitize_name(&result.model),
        sanitize_name(&result.task)
    );
    let json = serde_json::to_string_pretty(&artifact)?;
    std::fs::write(dir.join(name), json)?;

    let mut jsonl = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("runs.jsonl"))?;
    writeln!(jsonl, "{}", serde_json::to_string(&artifact)?)?;
    Ok(())
}

pub fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

pub fn default_artifacts_dir() -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from("target")
        .join("hi-eval")
        .join("runs")
        .join(format!("{stamp}-{}", std::process::id()))
}

/// Validate that every task is well-formed: its verify fails on the raw
/// fixture and passes once the `fixed/` reference is overlaid. Needs no model.
pub fn validate_tasks(tasks: &[(PathBuf, Task)]) -> Result<()> {
    eprintln!("Validating {} task(s)...", tasks.len());
    let mut broken = 0;
    for (dir, task) in tasks {
        let label = task.name.clone().unwrap_or_else(|| dir_name(dir));
        match validate_task(dir, task) {
            Ok(()) => println!("  OK      {label}"),
            Err(reason) => {
                println!("  BROKEN  {label}: {reason}");
                broken += 1;
            }
        }
    }
    if broken > 0 {
        bail!("{broken} task(s) are not well-formed");
    }
    println!(
        "\nAll {} tasks well-formed (fail-before, pass-after).",
        tasks.len()
    );
    Ok(())
}

pub fn validate_task(dir: &Path, task: &Task) -> std::result::Result<(), String> {
    let work = make_workdir().map_err(|e| e.to_string())?;
    let fixture = dir.join("fixture");
    if fixture.is_dir() {
        copy_dir(&fixture, &work).map_err(|e| e.to_string())?;
    }

    let result = (|| {
        if verify_in(&work, &task.verify) {
            return Err("verify already passes on the unmodified fixture".to_string());
        }
        let fixed = dir.join("fixed");
        if !fixed.is_dir() {
            return Err("no fixed/ reference to validate pass-after".to_string());
        }
        copy_dir(&fixed, &work).map_err(|e| e.to_string())?;
        if !verify_in(&work, &task.verify) {
            return Err("verify still fails after applying fixed/".to_string());
        }
        Ok(())
    })();

    let _ = std::fs::remove_dir_all(&work);
    result
}

pub fn verify_in(dir: &Path, cmd: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn discover_tasks(dir: &Path) -> Result<Vec<(PathBuf, Task)>> {
    let mut tasks = Vec::new();
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("reading tasks dir {}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        let toml_path = path.join("task.toml");
        if !toml_path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&toml_path)
            .with_context(|| format!("reading {}", toml_path.display()))?;
        let task: Task =
            toml::from_str(&text).with_context(|| format!("parsing {}", toml_path.display()))?;
        tasks.push((path, task));
    }
    Ok(tasks)
}

pub fn find_hi() -> Result<PathBuf> {
    // Must be absolute: each task runs with a different current_dir, so a
    // relative program path would resolve against the temp work dir.
    let candidate = if let Ok(bin) = std::env::var("HI_BIN") {
        PathBuf::from(bin)
    } else {
        ["target/debug/hi", "target/release/hi"]
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.is_file())
            .context("could not find the hi binary; build it or set HI_BIN")?
    };
    std::fs::canonicalize(&candidate)
        .with_context(|| format!("resolving hi binary path {}", candidate.display()))
}

pub fn make_workdir() -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hi-eval-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

pub fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            std::fs::create_dir_all(&to)?;
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

pub fn dir_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("task")
        .to_string()
}
