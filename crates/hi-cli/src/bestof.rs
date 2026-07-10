//! Interactive best-of-N: run N candidate attempts in isolated git worktrees,
//! keep the first that passes verification (the test suite is the judge), and
//! apply its changes back to the working repo.
//!
//! Candidates run from HEAD, so commit or stash uncommitted work first — we
//! warn if the tree is dirty. Requires `--verify`/`--auto-verify` (it defines
//! what "best" means) and a one-shot prompt.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};

pub struct BestOf<'a> {
    pub exe: &'a Path,
    pub provider: &'a str,
    pub model: &'a str,
    pub base_url: &'a str,
    pub api_key: &'a str,
    pub verify: &'a str,
    pub prompt: &'a str,
    pub candidates: u32,
    pub max_steps: Option<u32>,
    pub max_verify: u32,
}

pub fn run(opts: &BestOf) -> Result<()> {
    if !hi_tools::worktree::in_git_repo() {
        bail!("--best-of requires a git repository (candidates run in worktrees)");
    }
    if working_tree_dirty() {
        eprintln!(
            "\x1b[33mwarning: working tree has uncommitted changes; candidates run from HEAD and won't see them\x1b[0m"
        );
    }

    // Create all worktrees up front so a failure here aborts before spawning.
    let mut worktrees: Vec<(u32, PathBuf, f32)> = Vec::new();
    for i in 0..opts.candidates {
        let temperature = temperature_for(i, opts.candidates);
        let worktree = hi_tools::worktree::worktree_path("bestof", i);
        if let Err(err) = hi_tools::worktree::add_worktree(&worktree, "HEAD") {
            hi_tools::worktree::cleanup(
                &worktrees
                    .iter()
                    .map(|(_, wt, _)| wt.clone())
                    .collect::<Vec<_>>(),
            );
            return Err(err);
        }
        worktrees.push((i, worktree, temperature));
    }

    println!(
        "\x1b[36m── running {} candidates in parallel ──────────────────\x1b[0m",
        opts.candidates
    );
    let cleanup_paths: Vec<PathBuf> = worktrees.iter().map(|(_, wt, _)| wt.clone()).collect();

    // Durable artifacts dir (survives worktree teardown) for per-candidate reports/logs.
    let art_dir = artifacts_dir();
    let _ = std::fs::create_dir_all(&art_dir);

    // Spawn all candidate threads at once so they run concurrently. Each thread
    // owns copies of the scalar settings (the borrowed `BestOf` can't cross the
    // thread boundary) and builds its own `BestOf` view over those locals.
    let handles: Vec<std::thread::JoinHandle<(u32, PathBuf, bool)>> = worktrees
        .into_iter()
        .map(|(i, worktree, temperature)| {
            let exe = opts.exe.to_path_buf();
            let provider = opts.provider.to_string();
            let model = opts.model.to_string();
            let base_url = opts.base_url.to_string();
            let api_key = opts.api_key.to_string();
            let verify = opts.verify.to_string();
            let prompt = opts.prompt.to_string();
            let max_steps = opts.max_steps;
            let max_verify = opts.max_verify;
            let art_dir = art_dir.clone();
            std::thread::spawn(move || {
                let thread_opts = BestOf {
                    exe: &exe,
                    provider: &provider,
                    model: &model,
                    base_url: &base_url,
                    api_key: &api_key,
                    verify: &verify,
                    prompt: &prompt,
                    candidates: 0, // unused by run_candidate
                    max_steps,
                    max_verify,
                };
                let report_path = art_dir.join(format!("candidate-{i}.report.json"));
                let log_path = art_dir.join(format!("candidate-{i}.log"));
                let success =
                    run_candidate(&thread_opts, &worktree, temperature, &report_path, &log_path)
                        .unwrap_or(false);
                println!(
                    "\x1b[36m── candidate {}/{} (temp {temperature:.1}) finished ─────────────────\x1b[0m",
                    i + 1,
                    // count isn't available here; print as marker only
                    i + 1
                );
                (i, worktree, success)
            })
        })
        .collect();

    // Join all threads and collect results.
    let mut results: Vec<(u32, PathBuf, bool)> = Vec::new();
    let mut join_error: Option<anyhow::Error> = None;
    for handle in handles {
        match handle.join() {
            Ok(result) => results.push(result),
            Err(_) => {
                join_error = Some(anyhow!("candidate thread panicked"));
            }
        }
    }

    if let Some(err) = join_error {
        hi_tools::worktree::cleanup(&cleanup_paths);
        return Err(err);
    }

    // Deterministic order: by candidate index.
    results.sort_by_key(|r| r.0);

    // Find the first passing candidate (in index order): ran successfully AND
    // passes the verify command. Worktrees are independent, so checking each is
    // safe regardless of the others.
    let mut winner: Option<(u32, PathBuf)> = None;
    let total = opts.candidates;
    for (i, worktree, success) in &results {
        let idx = *i;
        if !success {
            println!(
                "\x1b[33m✗ candidate {}/{} exited with an error; skipping verification\x1b[0m",
                idx + 1,
                total
            );
            continue;
        }
        if hi_tools::worktree::verify_passes(worktree, opts.verify) {
            println!(
                "\x1b[32m✓ candidate {}/{} passed verification\x1b[0m",
                idx + 1,
                total
            );
            winner = Some((idx, worktree.clone()));
            break;
        }
        println!(
            "\x1b[33m✗ candidate {}/{} failed verification\x1b[0m",
            idx + 1,
            total
        );
    }

    let result = match &winner {
        Some((i, worktree)) => {
            let result = hi_tools::worktree::apply_changes(worktree, "HEAD")
                .map(|_| ())
                .with_context(|| "applying the winning candidate's changes");
            if result.is_ok() {
                println!(
                    "\x1b[32m✓ applied candidate {} to the working tree\x1b[0m",
                    i + 1
                );
            }
            result
        }
        None => {
            println!(
                "\x1b[31m✗ no candidate passed verification (tried {})\x1b[0m",
                total
            );
            Ok(())
        }
    };

    // Clean up every worktree we created (the artifacts dir is separate and kept).
    hi_tools::worktree::cleanup(&cleanup_paths);
    print_candidate_summary(&art_dir, total);
    result
}

/// Run one candidate `hi` in its worktree, with its own verify-loop. Emits a
/// machine-readable `--report` and captures the candidate's console output to a
/// log — both under the durable artifacts dir, so the parallel run stays
/// inspectable after the worktrees are torn down. Returns whether the candidate
/// process itself completed successfully.
fn run_candidate(
    opts: &BestOf,
    worktree: &Path,
    temperature: f32,
    report_path: &Path,
    log_path: &Path,
) -> Result<bool> {
    // `--report` needs an absolute path: the child runs in the worktree (which is
    // deleted at cleanup), and the artifacts dir lives outside it.
    let report = report_path.to_string_lossy().into_owned();
    let output = Command::new(opts.exe)
        .current_dir(worktree)
        // Force the parent's resolved key (not a re-resolved default-profile
        // literal). Env, not argv, so it isn't exposed in `ps`.
        .env("HI_FORCE_API_KEY", opts.api_key)
        .env("HI_API_KEY", opts.api_key)
        .args([
            "--no-save",
            "--provider",
            opts.provider,
            "--model",
            opts.model,
            "--base-url",
            opts.base_url,
            "--verify",
            opts.verify,
            "--temperature",
            &temperature.to_string(),
            "--max-verify",
            &opts.max_verify.to_string(),
            "--report",
            &report,
        ])
        .args(
            opts.max_steps
                .map(|max_steps| vec!["--max-steps".to_string(), max_steps.to_string()])
                .unwrap_or_default(),
        )
        .arg(opts.prompt)
        .output()
        .context("failed to launch candidate hi")?;
    // Persist stdout+stderr next to the report. Capturing (vs. inheriting) also
    // fixes the interleaved-output problem when N candidates run in parallel.
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut log = output.stdout;
    log.extend_from_slice(&output.stderr);
    let _ = std::fs::write(log_path, &log);
    Ok(output.status.success())
}

/// Durable, project-scoped home for best-of candidate artifacts: a `bestof/<pid>`
/// sibling of the session store (so it survives worktree teardown, and is
/// discoverable per project). Falls back to a temp dir if the data root can't be
/// resolved.
fn artifacts_dir() -> PathBuf {
    let pid = std::process::id();
    crate::session::sessions_dir()
        .and_then(|dir| dir.parent().map(Path::to_path_buf))
        .map(|root| root.join("bestof").join(pid.to_string()))
        .unwrap_or_else(|| std::env::temp_dir().join(format!("hi-bestof-{pid}-artifacts")))
}

/// Print a one-line-per-candidate summary read back from the reports, so the
/// parallel run is "explicit and inspectable" without opening each file.
fn print_candidate_summary(art_dir: &Path, count: u32) {
    println!(
        "\x1b[36m── candidate artifacts: {} ──────────────────\x1b[0m",
        art_dir.display()
    );
    for i in 0..count {
        let report = art_dir.join(format!("candidate-{i}.report.json"));
        let line = std::fs::read_to_string(&report)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .map(|json| {
                let verified = json.get("verify_passed").and_then(|v| v.as_bool());
                let tokens = json.get("total_tokens").and_then(|v| v.as_u64());
                let files = json
                    .get("changed_files")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len());
                format!(
                    "verify={} · {} tokens · {} files changed",
                    verified.map(|v| v.to_string()).unwrap_or("?".into()),
                    tokens.map(|t| t.to_string()).unwrap_or("?".into()),
                    files.map(|f| f.to_string()).unwrap_or("?".into()),
                )
            })
            .unwrap_or_else(|| "(no report)".to_string());
        println!("   candidate {}/{}: {line}", i + 1, count);
    }
}

/// Spread candidate temperatures across [0.2, 1.0] for diversity.
fn temperature_for(index: u32, count: u32) -> f32 {
    if count <= 1 {
        return 0.2;
    }
    0.2 + (index as f32) * (0.8 / (count - 1) as f32)
}

fn working_tree_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_candidate_reports_nonzero_exit() {
        let exe = Path::new("/bin/false");
        if !exe.exists() {
            return;
        }
        let opts = BestOf {
            exe,
            provider: "openai",
            model: "test-model",
            base_url: "http://127.0.0.1:9/v1",
            api_key: "test-key",
            verify: "true",
            prompt: "do the thing",
            candidates: 1,
            max_steps: Some(1),
            max_verify: 1,
        };

        let tmp = std::env::temp_dir();
        let report = tmp.join(format!("hi-bestof-test-{}.report.json", std::process::id()));
        let log = tmp.join(format!("hi-bestof-test-{}.log", std::process::id()));
        let _ = std::fs::remove_file(&log);
        assert!(
            !run_candidate(&opts, Path::new("."), 0.2, &report, &log)
                .expect("candidate command launches"),
            "a failing candidate process must not be considered eligible to win"
        );
        // Even a failed candidate leaves a durable (possibly empty) log — the run
        // stays inspectable after the worktrees are gone.
        assert!(log.exists(), "candidate log must be persisted");
        let _ = std::fs::remove_file(&log);
    }
}
