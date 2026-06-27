//! Interactive best-of-N: run N candidate attempts in isolated git worktrees,
//! keep the first that passes verification (the test suite is the judge), and
//! apply its changes back to the working repo.
//!
//! Candidates run from HEAD, so commit or stash uncommitted work first — we
//! warn if the tree is dirty. Requires `--verify`/`--auto-verify` (it defines
//! what "best" means) and a one-shot prompt.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

pub struct BestOf<'a> {
    pub exe: &'a Path,
    pub provider: &'a str,
    pub model: &'a str,
    pub base_url: &'a str,
    pub api_key: &'a str,
    pub verify: &'a str,
    pub prompt: &'a str,
    pub candidates: u32,
    pub max_steps: u32,
    pub max_verify: u32,
}

pub fn run(opts: &BestOf) -> Result<()> {
    if !in_git_repo() {
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
        let worktree = worktree_path(i);
        if let Err(err) = add_worktree(&worktree) {
            cleanup(
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
                let success =
                    run_candidate(&thread_opts, &worktree, temperature).unwrap_or(false);
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
    for handle in handles {
        results.push(handle.join().expect("candidate thread panicked"));
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
        if verify_passes(worktree, opts.verify) {
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
            apply_changes(worktree).with_context(|| "applying the winning candidate's changes")?;
            println!(
                "\x1b[32m✓ applied candidate {} to the working tree\x1b[0m",
                i + 1
            );
            Ok(())
        }
        None => {
            println!(
                "\x1b[31m✗ no candidate passed verification (tried {})\x1b[0m",
                total
            );
            Ok(())
        }
    };

    // Clean up every worktree we created.
    let all_paths: Vec<PathBuf> = results.iter().map(|(_, wt, _)| wt.clone()).collect();
    cleanup(&all_paths);
    result
}

/// Run one candidate `hi` in its worktree, with its own verify-loop. Returns
/// whether the candidate process itself completed successfully.
fn run_candidate(opts: &BestOf, worktree: &Path, temperature: f32) -> Result<bool> {
    let status = Command::new(opts.exe)
        .current_dir(worktree)
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
            "--max-steps",
            &opts.max_steps.to_string(),
            "--max-verify",
            &opts.max_verify.to_string(),
            opts.prompt,
        ])
        .status()
        .context("failed to launch candidate hi")?;
    Ok(status.success())
}

/// Spread candidate temperatures across [0.2, 1.0] for diversity.
fn temperature_for(index: u32, count: u32) -> f32 {
    if count <= 1 {
        return 0.2;
    }
    0.2 + (index as f32) * (0.8 / (count - 1) as f32)
}

fn in_git_repo() -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn working_tree_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false)
}

fn worktree_path(index: u32) -> PathBuf {
    std::env::temp_dir().join(format!("hi-bestof-{}-{index}", std::process::id()))
}

fn add_worktree(path: &Path) -> Result<()> {
    let output = Command::new("git")
        .args(["worktree", "add", "--detach"])
        .arg(path)
        .arg("HEAD")
        .output()
        .context("running git worktree add")?;
    if !output.status.success() {
        bail!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Ground-truth check: run the verify command in the worktree ourselves.
fn verify_passes(worktree: &Path, verify: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(verify)
        .current_dir(worktree)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Apply the worktree's changes (including new files) to the main working tree.
fn apply_changes(worktree: &Path) -> Result<()> {
    // Stage everything so the diff captures new/deleted files too.
    let add = Command::new("git")
        .current_dir(worktree)
        .args(["add", "-A"])
        .output()
        .context("git add in worktree")?;
    if !add.status.success() {
        bail!(
            "git add failed: {}",
            String::from_utf8_lossy(&add.stderr).trim()
        );
    }

    let diff = Command::new("git")
        .current_dir(worktree)
        .args(["diff", "--cached", "HEAD"])
        .output()
        .context("git diff in worktree")?;
    if !diff.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&diff.stderr).trim()
        );
    }
    if diff.stdout.is_empty() {
        return Ok(()); // nothing changed
    }

    // Apply the patch in the main repo via stdin.
    use std::io::Write;
    let mut child = Command::new("git")
        .args(["apply", "--whitespace=nowarn"])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("spawning git apply")?;
    child
        .stdin
        .take()
        .context("git apply stdin")?
        .write_all(&diff.stdout)
        .context("writing patch to git apply")?;
    let status = child.wait().context("waiting for git apply")?;
    if !status.success() {
        bail!("git apply failed (working tree may conflict with the patch)");
    }
    Ok(())
}

fn cleanup(worktrees: &[PathBuf]) {
    for path in worktrees {
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(path)
            .output();
    }
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
            max_steps: 1,
            max_verify: 1,
        };

        assert!(
            !run_candidate(&opts, Path::new("."), 0.2).expect("candidate command launches"),
            "a failing candidate process must not be considered eligible to win"
        );
    }
}
