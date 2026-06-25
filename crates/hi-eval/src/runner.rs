use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::artifacts::{copy_dir, make_workdir};
use crate::config::{EvalProfile, Task};
use crate::results::{
    Candidate, FailKind, RunResult, Trajectory, TrajectoryAttribution, classify,
    looks_like_build_error,
};

/// A content snapshot of `dir` (relative path → bytes), excluding eval/run and
/// common build artifacts, so we can tell whether the model actually changed
/// task files rather than just triggering a build.
pub fn dir_snapshot(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
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

/// Run all of a config's candidates; the config solves the task if any passes.
/// Cost and tokens are summed. Candidates run in parallel since each gets its own
/// isolated workdir — wall-clock is the max, not the sum.
pub fn run_config(
    hi: &Path,
    task_dir: &Path,
    task: &Task,
    config_name: &str,
    use_verify: bool,
    temperatures: &[f32],
    profile: EvalProfile,
) -> impl std::future::Future<Output = Result<RunResult>> {
    async move {
    let mut result = RunResult {
        config: config_name.to_string(),
        task: String::new(),
        trial: 0,
        passed: false,
        fail: None,
        provider_error_kind: None,
        compat_fallbacks_used: Vec::new(),
        changed_files: Vec::new(),
        verify_output_summary: String::new(),
        failure_confidence: None,
        candidates: temperatures.len(),
        cost_usd: 0.0,
        tokens: 0,
        seconds: 0.0,
        trajectory: Trajectory::default(),
    };

    // Run candidates in parallel — each gets its own temp workdir.
    // `run_config` is already executing on the eval runtime (called from
    // `async_main` via `tokio::spawn`), so we must not create a second Tokio
    // runtime here and `block_on` from within it — nesting runtimes panics.
    // Just await the candidate futures directly on this runtime.
    let candidates: Vec<Candidate> = async {
        let mut futs = Vec::new();
        for &temperature in temperatures {
            let hi = hi.to_path_buf();
            let task_dir = task_dir.to_path_buf();
            let prompt = task.prompt.clone();
            let verify = task.verify.clone();
            futs.push(tokio::task::spawn_blocking(move || {
                run_candidate(
                    &hi,
                    &task_dir,
                    &Task {
                        name: None,
                        prompt,
                        verify,
                    },
                    use_verify,
                    temperature,
                    profile,
                )
            }));
        }
        let mut out = Vec::with_capacity(futs.len());
        for fut in futs {
            out.push(fut.await.context("joining candidate task")??);
        }
        Ok::<_, anyhow::Error>(out)
    }
    .await?;

    let mut fails: Vec<FailKind> = Vec::new();
    let mut summaries = Vec::new();
    // Track the representative candidate (furthest-progressing) so its
    // trajectory is surfaced, mirroring how `result.fail` is chosen.
    let mut best_rank: i32 = -1;
    let mut representative_trajectory: Option<Trajectory> = None;
    for candidate in candidates {
        let cand_rank = candidate
            .fail
            .map(|k| k.rank() as i32)
            .unwrap_or(if candidate.passed { 4 } else { -1 });
        if cand_rank > best_rank {
            best_rank = cand_rank;
            representative_trajectory = Some(candidate.trajectory.clone());
        }
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
    result.trajectory = representative_trajectory.unwrap_or_default();
    result.changed_files.sort();
    result.changed_files.dedup();
    result.compat_fallbacks_used.sort();
    result.compat_fallbacks_used.dedup();
    result.verify_output_summary = summaries.join("\n--- candidate ---\n");
    Ok(result)
    }
}

/// One independent attempt in an isolated copy of the fixture.
pub fn run_candidate(
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
        trajectory: report.trajectory,
    })
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

struct ReportInfo {
    tokens: u64,
    cost_usd: f64,
    provider_error_kind: Option<String>,
    compat_fallbacks_used: Vec<String>,
    changed_files: Vec<String>,
    trajectory: Trajectory,
}

fn read_report(path: &Path) -> ReportInfo {
    let Ok(text) = std::fs::read_to_string(path) else {
        return ReportInfo::default();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return ReportInfo::default();
    };
    let tel = &value["telemetry"];
    let trajectory = Trajectory {
        verify_rounds: tel["verify_rounds"].as_u64().unwrap_or(0) as u32,
        recovery_retries: tel["recovery_retries"].as_u64().unwrap_or(0) as u32,
        repeat_nudges: tel["repeat_nudges"].as_u64().unwrap_or(0) as u32,
        continue_nudges: tel["continue_nudges"].as_u64().unwrap_or(0) as u32,
        hit_step_cap: tel["hit_step_cap"].as_bool().unwrap_or(false),
        stalled_unfinished: tel["stalled_unfinished"].as_bool().unwrap_or(false),
        stalled_repeating: tel["stalled_repeating"].as_bool().unwrap_or(false),
        verify_attributions: tel["verify_attributions"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|a| {
                        Some(TrajectoryAttribution {
                            path: a["path"].as_str().unwrap_or("").to_string(),
                            line: a["line"].as_u64().map(|n| n as u32),
                            column: a["column"].as_u64().map(|n| n as u32),
                            message: a["message"].as_str().unwrap_or("").to_string(),
                            kind: a["kind"].as_str().unwrap_or("").to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default(),
        tool_calls: tel["tool_calls"].as_u64().unwrap_or(0) as u32,
        max_concurrent_batch: tel["max_concurrent_batch"].as_u64().unwrap_or(0) as u32,
        serial_runs: tel["serial_runs"].as_u64().unwrap_or(0) as u32,
    };
    ReportInfo {
        tokens: value["total_tokens"].as_u64().unwrap_or(0),
        cost_usd: value["cost_usd"].as_f64().unwrap_or(0.0),
        provider_error_kind: value["provider_error_kind"].as_str().map(str::to_string),
        compat_fallbacks_used: string_array(&value["compat_fallbacks_used"]),
        changed_files: string_array(&value["changed_files"]),
        trajectory,
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
            trajectory: Trajectory::default(),
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

#[cfg(test)]
mod tests {
    use super::dir_snapshot;

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
