//! `hi-eval` — coding-task benchmark runner for `hi`.
//!
//! Runs each task under each config in an isolated copy of its fixture, scores
//! pass/fail by the task's own verify command (ground truth), and reports
//! pass-rate, cost, and time per config. This is how we measure whether a
//! lever (e.g. verification-in-the-loop) actually beats a baseline — including
//! a real backend like `openrouter/fusion`.
//!
//! Model selection flows through to `hi` via the usual env vars
//! (HI_MODEL / HI_BASE_URL / HI_API_KEY), so you compare backends by swapping
//! env, not code:
//!
//!   HI_MODEL=openrouter/fusion HI_API_KEY=… cargo run -p hi-eval -- bench/tasks
//!
//! Usage: hi-eval [TASKS_DIR]   (default: bench/tasks). Set HI_BIN to override
//! the hi binary path.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// A benchmark task: a prompt to run and a command that decides success.
#[derive(Deserialize)]
struct Task {
    name: Option<String>,
    prompt: String,
    /// Shell command run in the work dir; exit 0 == solved.
    verify: String,
}

/// One way of running `hi` against the tasks.
struct Config {
    name: &'static str,
    /// Pass `--verify <task.verify>` so the agent iterates to green.
    use_verify: bool,
    /// One sampling temperature per candidate. The config solves the task if
    /// ANY candidate passes — execution-grounded best-of-N (the test suite is
    /// the judge). Cost/tokens are summed across all candidates.
    temperatures: &'static [f32],
}

const CONFIGS: &[Config] = &[
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

struct RunResult {
    config: &'static str,
    passed: bool,
    candidates: usize,
    cost_usd: f64,
    tokens: u64,
    seconds: f64,
}

struct Candidate {
    passed: bool,
    cost_usd: f64,
    tokens: u64,
    seconds: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let validate = args.iter().any(|a| a == "--validate");
    let tasks_dir = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "bench/tasks".to_string());

    // --configs=baseline,verify selects a subset of configs (default: all).
    let configs_filter: Option<Vec<String>> = args
        .iter()
        .find_map(|a| a.strip_prefix("--configs="))
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let active: Vec<&Config> = CONFIGS
        .iter()
        .filter(|c| {
            configs_filter
                .as_ref()
                .is_none_or(|f| f.iter().any(|n| n == c.name))
        })
        .collect();
    if active.is_empty() {
        bail!("no configs match --configs; known: baseline, verify, best-of-3");
    }

    let tasks = discover_tasks(Path::new(&tasks_dir))?;
    if tasks.is_empty() {
        bail!("no tasks (with task.toml) found under {tasks_dir}");
    }

    if validate {
        return validate_tasks(&tasks);
    }

    let hi = find_hi()?;
    let model = std::env::var("HI_MODEL").unwrap_or_else(|_| "(unset)".into());
    eprintln!(
        "hi-eval: {} task(s) × {} config(s) · model={model} · hi={}",
        tasks.len(),
        active.len(),
        hi.display()
    );

    let mut results = Vec::new();
    for (dir, task) in &tasks {
        let label = task.name.clone().unwrap_or_else(|| dir_name(dir));
        for config in &active {
            let result = run_config(&hi, dir, task, config)
                .with_context(|| format!("running task '{label}' [{}]", config.name))?;
            eprintln!(
                "  {:10} {:4} {label}  ({} cand, {} tok, ${:.4}, {:.1}s)",
                config.name,
                if result.passed { "PASS" } else { "FAIL" },
                result.candidates,
                result.tokens,
                result.cost_usd,
                result.seconds
            );
            results.push(result);
        }
    }

    print_summary(&results, tasks.len(), &active);
    Ok(())
}

/// Validate that every task is well-formed: its verify fails on the raw
/// fixture and passes once the `fixed/` reference is overlaid. Needs no model.
fn validate_tasks(tasks: &[(PathBuf, Task)]) -> Result<()> {
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

fn validate_task(dir: &Path, task: &Task) -> std::result::Result<(), String> {
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

fn verify_in(dir: &Path, cmd: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run all of a config's candidates; the config solves the task if any passes.
/// Cost and tokens are summed (in production the N candidates run in parallel,
/// so wall-clock would be the max, but cost is still the sum).
fn run_config(hi: &Path, task_dir: &Path, task: &Task, config: &Config) -> Result<RunResult> {
    let mut result = RunResult {
        config: config.name,
        passed: false,
        candidates: config.temperatures.len(),
        cost_usd: 0.0,
        tokens: 0,
        seconds: 0.0,
    };
    for &temperature in config.temperatures {
        let candidate = run_candidate(hi, task_dir, task, config.use_verify, temperature)?;
        result.passed |= candidate.passed;
        result.cost_usd += candidate.cost_usd;
        result.tokens += candidate.tokens;
        result.seconds += candidate.seconds;
    }
    Ok(result)
}

/// One independent attempt in an isolated copy of the fixture.
fn run_candidate(
    hi: &Path,
    task_dir: &Path,
    task: &Task,
    use_verify: bool,
    temperature: f32,
) -> Result<Candidate> {
    let work = make_workdir()?;
    let fixture = task_dir.join("fixture");
    if fixture.is_dir() {
        copy_dir(&fixture, &work)?;
    }
    let report = work.join(".hi-eval-report.json");

    let mut cmd = Command::new(hi);
    cmd.current_dir(&work)
        .arg("--no-save")
        .arg("--report")
        .arg(&report)
        .arg("--temperature")
        .arg(temperature.to_string());
    if use_verify {
        cmd.arg("--verify").arg(&task.verify);
    }
    cmd.arg(&task.prompt);

    let started = Instant::now();
    let output = cmd.output().context("failed to launch hi")?;
    let seconds = started.elapsed().as_secs_f64();
    if !output.status.success() {
        eprintln!(
            "    (hi exited {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    // Ground truth: run the verify command ourselves in the work dir.
    let passed = Command::new("sh")
        .arg("-c")
        .arg(&task.verify)
        .current_dir(&work)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let (tokens, cost_usd) = read_report(&report);
    let _ = std::fs::remove_dir_all(&work);

    Ok(Candidate {
        passed,
        cost_usd,
        tokens,
        seconds,
    })
}

fn print_summary(results: &[RunResult], task_count: usize, active: &[&Config]) {
    println!("\n=== Results ({task_count} tasks) ===");
    println!(
        "{:<10} {:>8} {:>6} {:>10} {:>10}",
        "config", "pass", "cand", "cost", "tokens"
    );
    for config in active {
        let rows: Vec<&RunResult> = results.iter().filter(|r| r.config == config.name).collect();
        let passed = rows.iter().filter(|r| r.passed).count();
        let cost: f64 = rows.iter().map(|r| r.cost_usd).sum();
        let tokens: u64 = rows.iter().map(|r| r.tokens).sum();
        println!(
            "{:<10} {:>8} {:>6} {:>10} {:>10}",
            config.name,
            format!("{passed}/{}", rows.len()),
            config.temperatures.len(),
            format!("${cost:.4}"),
            tokens
        );
    }
}

fn read_report(path: &Path) -> (u64, f64) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (0, 0.0);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (0, 0.0);
    };
    let tokens = value["total_tokens"].as_u64().unwrap_or(0);
    let cost = value["cost_usd"].as_f64().unwrap_or(0.0);
    (tokens, cost)
}

fn discover_tasks(dir: &Path) -> Result<Vec<(PathBuf, Task)>> {
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

fn find_hi() -> Result<PathBuf> {
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

fn make_workdir() -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hi-eval-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
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

fn dir_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("task")
        .to_string()
}
