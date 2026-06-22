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

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

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
    task: String,
    trial: usize,
    passed: bool,
    /// Why it failed (None when it passed) — for the failure-mode breakdown.
    fail: Option<FailKind>,
    provider_error_kind: Option<String>,
    compat_fallbacks_used: Vec<String>,
    changed_files: Vec<String>,
    verify_output_summary: String,
    failure_confidence: Option<&'static str>,
    candidates: usize,
    cost_usd: f64,
    tokens: u64,
    seconds: f64,
}

struct Candidate {
    passed: bool,
    fail: Option<FailKind>,
    provider_error_kind: Option<String>,
    compat_fallbacks_used: Vec<String>,
    changed_files: Vec<String>,
    verify_output_summary: String,
    failure_confidence: Option<&'static str>,
    cost_usd: f64,
    tokens: u64,
    seconds: f64,
}

/// Why a candidate failed — so the summary shows *where* hi loses, not just how
/// often. Ordered by how far the attempt got (Error = least, Logic = most).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FailKind {
    /// `hi` itself errored/crashed (provider failure, non-zero exit).
    Error,
    /// The model changed no files — answered, gave up, or never acted.
    NoEdits,
    /// Files changed but the code doesn't build/load (compile/type/import error).
    Compile,
    /// Builds and runs, but behavior is wrong (the model's rule was off).
    Logic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvalProfile {
    Default,
    Terminaili,
}

impl EvalProfile {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("default") {
            "default" => Ok(Self::Default),
            "terminaili" => Ok(Self::Terminaili),
            other => bail!("unknown --profile={other}; known: default, terminaili"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Terminaili => "terminaili",
        }
    }

    fn hi_args(self) -> &'static [&'static str] {
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

    fn validate_env(self) -> Result<()> {
        if matches!(self, Self::Terminaili)
            && std::env::var("TERMINAILI_API_KEY").is_err()
            && std::env::var("HI_API_KEY").is_err()
        {
            bail!("--profile=terminaili requires TERMINAILI_API_KEY or HI_API_KEY");
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct RunArtifact {
    task: String,
    config: String,
    trial: usize,
    profile: String,
    passed: bool,
    failure_bucket: Option<String>,
    failure_confidence: Option<&'static str>,
    changed_files: Vec<String>,
    provider_error_kind: Option<String>,
    compat_fallbacks_used: Vec<String>,
    candidates: usize,
    tokens: u64,
    cost_usd: f64,
    duration_seconds: f64,
    verify_output_summary: String,
}

impl FailKind {
    fn label(self) -> &'static str {
        match self {
            FailKind::Error => "error",
            FailKind::NoEdits => "no-edits",
            FailKind::Compile => "compile",
            FailKind::Logic => "logic",
        }
    }
    /// Progress rank — when candidates fail different ways, the config's
    /// representative failure is the one that got furthest.
    fn rank(self) -> u8 {
        match self {
            FailKind::Error => 0,
            FailKind::NoEdits => 1,
            FailKind::Compile => 2,
            FailKind::Logic => 3,
        }
    }
}

/// Classify a failed candidate from the signals we have.
fn classify(passed: bool, hi_ok: bool, edited: bool, verify_output: &str) -> Option<FailKind> {
    if passed {
        return None;
    }
    if !hi_ok {
        return Some(FailKind::Error);
    }
    if !edited {
        return Some(FailKind::NoEdits);
    }
    if looks_like_build_error(verify_output) {
        Some(FailKind::Compile)
    } else {
        Some(FailKind::Logic)
    }
}

/// Heuristic: does verify output indicate the code didn't build/load (vs. a
/// behavioral test failure)? Strong, language-specific markers only, so test
/// assertions ("expected X got Y", "AssertionError") stay classified as logic.
fn looks_like_build_error(s: &str) -> bool {
    const MARKERS: &[&str] = &[
        "error[E",             // rustc
        "cannot find",         // rustc / go
        "cannot borrow",       // rustc
        "mismatched types",    // rustc
        "unresolved import",   // rustc
        "SyntaxError",         // python / js
        "IndentationError",    // python
        "NameError",           // python
        "ImportError",         // python
        "ModuleNotFoundError", // python
        "Cannot find name",    // ts
        "Cannot find module",  // ts / js
        "is not defined",      // js
        "undefined:",          // go
        "undefined reference", // c/c++ link
        "cannot use",          // go type error
        "compilation failed",
        "build failed",
    ];
    MARKERS.iter().any(|m| s.contains(m))
}

/// A content snapshot of `dir` (relative path → bytes), excluding eval/run and
/// common build artifacts, so we can tell whether the model actually changed
/// task files rather than just triggering a build.
fn dir_snapshot(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(base: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, Vec<u8>>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(
                name.as_ref(),
                ".hi-eval-report.json"
                    | ".hi-debug.log"
                    | ".git"
                    | "target"
                    | "node_modules"
                    | ".next"
                    | "dist"
                    | "build"
                    | "__pycache__"
                    | ".pytest_cache"
            ) {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                walk(base, &path, out);
            } else if let Ok(bytes) = std::fs::read(&path)
                && let Ok(rel) = path.strip_prefix(base)
            {
                out.insert(rel.to_string_lossy().into_owned(), bytes);
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let validate = args.iter().any(|a| a == "--validate");
    let tasks_dir = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "bench/tasks".to_string());
    let profile = EvalProfile::parse(args.iter().find_map(|a| a.strip_prefix("--profile=")))?;
    let artifacts_dir = args
        .iter()
        .find_map(|a| a.strip_prefix("--artifacts="))
        .map(PathBuf::from)
        .unwrap_or_else(default_artifacts_dir);

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

    // --trials=N repeats the whole matrix N times so the summary can report a
    // mean ± spread and pass@k (single runs are too noisy to trust).
    let trials: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--trials="))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    let tasks = discover_tasks(Path::new(&tasks_dir))?;
    if tasks.is_empty() {
        bail!("no tasks (with task.toml) found under {tasks_dir}");
    }

    if validate {
        return validate_tasks(&tasks);
    }
    profile.validate_env()?;
    std::fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("creating artifacts dir {}", artifacts_dir.display()))?;

    let hi = find_hi()?;
    let model = std::env::var("HI_MODEL").unwrap_or_else(|_| "(unset)".into());
    eprintln!(
        "hi-eval: {} task(s) × {} config(s) × {trials} trial(s) · model={model} · profile={} · hi={} · artifacts={}",
        tasks.len(),
        active.len(),
        profile.label(),
        hi.display(),
        artifacts_dir.display()
    );

    let mut results = Vec::new();
    for trial in 0..trials {
        if trials > 1 {
            eprintln!("--- trial {}/{trials} ---", trial + 1);
        }
        for (dir, task) in &tasks {
            let label = task.name.clone().unwrap_or_else(|| dir_name(dir));
            for config in &active {
                let mut result = run_config(&hi, dir, task, config, profile)
                    .with_context(|| format!("running task '{label}' [{}]", config.name))?;
                result.task = label.clone();
                result.trial = trial;
                write_artifact(&artifacts_dir, profile, &result)?;
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
    }

    print_summary(&results, tasks.len(), &active, trials);
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
fn run_config(
    hi: &Path,
    task_dir: &Path,
    task: &Task,
    config: &Config,
    profile: EvalProfile,
) -> Result<RunResult> {
    let mut result = RunResult {
        config: config.name,
        task: String::new(),
        trial: 0,
        passed: false,
        fail: None,
        provider_error_kind: None,
        compat_fallbacks_used: Vec::new(),
        changed_files: Vec::new(),
        verify_output_summary: String::new(),
        failure_confidence: None,
        candidates: config.temperatures.len(),
        cost_usd: 0.0,
        tokens: 0,
        seconds: 0.0,
    };
    let mut fails: Vec<FailKind> = Vec::new();
    let mut summaries = Vec::new();
    for &temperature in config.temperatures {
        let candidate = run_candidate(hi, task_dir, task, config.use_verify, temperature, profile)?;
        result.passed |= candidate.passed;
        if let Some(k) = candidate.fail {
            fails.push(k);
        }
        if result.provider_error_kind.is_none() {
            result.provider_error_kind = candidate.provider_error_kind.clone();
        }
        result
            .compat_fallbacks_used
            .extend(candidate.compat_fallbacks_used.clone());
        result.changed_files.extend(candidate.changed_files.clone());
        if !candidate.verify_output_summary.is_empty() {
            summaries.push(candidate.verify_output_summary.clone());
        }
        result.failure_confidence = candidate.failure_confidence;
        result.cost_usd += candidate.cost_usd;
        result.tokens += candidate.tokens;
        result.seconds += candidate.seconds;
    }
    // When the config failed (no candidate passed), its representative failure is
    // the one that got furthest (e.g. logic over no-edits).
    if !result.passed {
        result.fail = fails.into_iter().max_by_key(|k| k.rank());
    }
    result.changed_files.sort();
    result.changed_files.dedup();
    result.compat_fallbacks_used.sort();
    result.compat_fallbacks_used.dedup();
    result.verify_output_summary = summaries.join("\n--- candidate ---\n");
    Ok(result)
}

/// One independent attempt in an isolated copy of the fixture.
fn run_candidate(
    hi: &Path,
    task_dir: &Path,
    task: &Task,
    use_verify: bool,
    temperature: f32,
    profile: EvalProfile,
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
    for arg in profile.hi_args() {
        cmd.arg(arg);
    }
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

    // Snapshot before verification so verify-side effects (cache files, build
    // outputs, etc.) don't get misclassified as model edits.
    let fallback_edited = dir_snapshot(&work) != dir_snapshot(&fixture);

    // Ground truth: run the verify command ourselves, capturing its output so
    // we can classify *how* a failure failed (compile vs. logic).
    let (passed, verify_output) = match Command::new("sh")
        .arg("-c")
        .arg(&task.verify)
        .current_dir(&work)
        .output()
    {
        Ok(o) => {
            let mut out = String::from_utf8_lossy(&o.stdout).into_owned();
            out.push_str(&String::from_utf8_lossy(&o.stderr));
            (o.status.success(), out)
        }
        Err(_) => (false, String::new()),
    };

    let report = read_report(&report);
    let edited = if report.changed_files.is_empty() {
        fallback_edited
    } else {
        true
    };
    let fail = classify(passed, output.status.success(), edited, &verify_output);
    let failure_confidence = fail.map(|kind| match kind {
        FailKind::Error if report.provider_error_kind.is_some() => "high",
        FailKind::NoEdits if report.changed_files.is_empty() => "high",
        FailKind::Compile if looks_like_build_error(&verify_output) => "high",
        FailKind::Logic => "medium",
        _ => "low",
    });

    let _ = std::fs::remove_dir_all(&work);

    Ok(Candidate {
        passed,
        fail,
        provider_error_kind: report.provider_error_kind,
        compat_fallbacks_used: report.compat_fallbacks_used,
        changed_files: report.changed_files,
        verify_output_summary: summarize_output(&verify_output),
        failure_confidence,
        cost_usd: report.cost_usd,
        tokens: report.tokens,
        seconds,
    })
}

fn write_artifact(dir: &Path, profile: EvalProfile, result: &RunResult) -> Result<()> {
    let artifact = RunArtifact {
        task: result.task.clone(),
        config: result.config.to_string(),
        trial: result.trial,
        profile: profile.label().to_string(),
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
        verify_output_summary: result.verify_output_summary.clone(),
    };
    let name = format!(
        "trial-{:03}-{}-{}.json",
        result.trial + 1,
        sanitize_name(result.config),
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

fn summarize_output(output: &str) -> String {
    const MAX: usize = 4000;
    let trimmed = output.trim();
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(1800).collect();
    let tail: String = trimmed
        .chars()
        .rev()
        .take(1800)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n\n[... verify output truncated ...]\n\n{tail}")
}

fn sanitize_name(value: &str) -> String {
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

fn default_artifacts_dir() -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from("target")
        .join("hi-eval")
        .join("runs")
        .join(format!("{stamp}-{}", std::process::id()))
}

fn print_summary(results: &[RunResult], task_count: usize, active: &[&Config], trials: usize) {
    println!("\n=== Results ({task_count} tasks × {trials} trial(s)) ===");
    println!(
        "{:<10} {:>14} {:>8} {:>6} {:>11} {:>10}",
        "config", "pass@1", "pass@k", "cand", "tok/trial", "$/trial"
    );
    for config in active {
        let rows: Vec<&RunResult> = results.iter().filter(|r| r.config == config.name).collect();

        // Tasks passed per trial → mean ± spread (pass@1, with error bars).
        let mut per_trial = vec![0usize; trials];
        for r in &rows {
            if r.passed {
                per_trial[r.trial] += 1;
            }
        }
        let mean = per_trial.iter().sum::<usize>() as f64 / trials as f64;
        let std = (per_trial
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / trials as f64)
            .sqrt();
        let pass1 = if trials == 1 {
            format!("{}/{task_count}", per_trial[0])
        } else {
            format!("{mean:.1}±{std:.1}/{task_count}")
        };

        // pass@k: tasks solved by at least one trial (the capability ceiling).
        let mut solved: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for r in &rows {
            if r.passed {
                solved.insert(r.task.as_str());
            }
        }

        let tokens: u64 = rows.iter().map(|r| r.tokens).sum::<u64>() / trials as u64;
        let cost: f64 = rows.iter().map(|r| r.cost_usd).sum::<f64>() / trials as f64;
        println!(
            "{:<10} {:>14} {:>8} {:>6} {:>11} {:>10}",
            config.name,
            pass1,
            format!("{}/{task_count}", solved.len()),
            config.temperatures.len(),
            tokens,
            format!("${cost:.4}")
        );

        // Failure-mode breakdown: where this config loses, across all trials.
        let mut hist = [0usize; 4];
        for r in &rows {
            if let Some(k) = r.fail {
                hist[k.rank() as usize] += 1;
            }
        }
        if hist.iter().any(|&n| n > 0) {
            let parts: Vec<String> = [
                FailKind::Logic,
                FailKind::Compile,
                FailKind::NoEdits,
                FailKind::Error,
            ]
            .iter()
            .filter(|k| hist[k.rank() as usize] > 0)
            .map(|k| format!("{} {}", k.label(), hist[k.rank() as usize]))
            .collect();
            println!(
                "           why: {} (of {} failing cells)",
                parts.join(" · "),
                hist.iter().sum::<usize>()
            );
        }
    }
}

struct ReportInfo {
    tokens: u64,
    cost_usd: f64,
    provider_error_kind: Option<String>,
    compat_fallbacks_used: Vec<String>,
    changed_files: Vec<String>,
}

fn read_report(path: &Path) -> ReportInfo {
    let Ok(text) = std::fs::read_to_string(path) else {
        return ReportInfo::default();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return ReportInfo::default();
    };
    ReportInfo {
        tokens: value["total_tokens"].as_u64().unwrap_or(0),
        cost_usd: value["cost_usd"].as_f64().unwrap_or(0.0),
        provider_error_kind: value["provider_error_kind"].as_str().map(str::to_string),
        compat_fallbacks_used: string_array(&value["compat_fallbacks_used"]),
        changed_files: string_array(&value["changed_files"]),
    }
}

impl Default for ReportInfo {
    fn default() -> Self {
        Self {
            tokens: 0,
            cost_usd: 0.0,
            provider_error_kind: None,
            compat_fallbacks_used: Vec::new(),
            changed_files: Vec::new(),
        }
    }
}

fn string_array(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
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

#[cfg(test)]
mod tests {
    use super::{FailKind, classify, dir_snapshot, looks_like_build_error};

    #[test]
    fn classify_covers_each_mode() {
        // Passed → no failure.
        assert_eq!(classify(true, true, true, ""), None);
        // hi crashed → error, regardless of edits.
        assert_eq!(classify(false, false, false, ""), Some(FailKind::Error));
        // Ran fine but changed nothing → no-edits.
        assert_eq!(classify(false, true, false, ""), Some(FailKind::NoEdits));
        // Edited but doesn't compile → compile.
        assert_eq!(
            classify(false, true, true, "error[E0382]: borrow of moved value"),
            Some(FailKind::Compile)
        );
        // Edited, compiles, wrong behavior → logic.
        assert_eq!(
            classify(false, true, true, "assertion failed: expected 4 got 5"),
            Some(FailKind::Logic)
        );
    }

    #[test]
    fn build_errors_vs_assertions() {
        assert!(looks_like_build_error("error[E0599]: no method named foo"));
        assert!(looks_like_build_error(
            "Traceback... ModuleNotFoundError: no module"
        ));
        assert!(looks_like_build_error("x.ts: Cannot find name 'foo'"));
        // Behavioral failures must NOT look like build errors.
        assert!(!looks_like_build_error(
            "test result: FAILED. 1 passed; 1 failed"
        ));
        assert!(!looks_like_build_error("AssertionError: expected 4, got 5"));
    }

    #[test]
    fn dir_snapshot_ignores_build_artifacts() {
        let root = std::env::temp_dir().join(format!("hi-eval-snapshot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
        let before = dir_snapshot(&root);

        std::fs::write(root.join("target").join("build.log"), "compiled\n").unwrap();
        let after_build = dir_snapshot(&root);
        assert_eq!(
            before, after_build,
            "build artifacts must not count as edits"
        );

        std::fs::write(root.join("main.rs"), "fn main(){println!(\"x\");}\n").unwrap();
        let after_source = dir_snapshot(&root);
        assert_ne!(before, after_source, "source edits must still count");

        let _ = std::fs::remove_dir_all(&root);
    }
}
