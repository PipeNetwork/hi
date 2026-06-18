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

    let mut created: Vec<PathBuf> = Vec::new();
    let mut winner: Option<(u32, PathBuf)> = None;

    for i in 0..opts.candidates {
        let temperature = temperature_for(i, opts.candidates);
        let worktree = worktree_path(i);

        if let Err(err) = add_worktree(&worktree) {
            cleanup(&created);
            return Err(err);
        }
        created.push(worktree.clone());

        println!(
            "\x1b[36m── candidate {}/{} (temp {temperature:.1}) ─────────────────\x1b[0m",
            i + 1,
            opts.candidates
        );
        run_candidate(opts, &worktree, temperature)?;

        if verify_passes(&worktree, opts.verify) {
            println!("\x1b[32m✓ candidate {} passed verification\x1b[0m", i + 1);
            winner = Some((i, worktree));
            break;
        }
        println!("\x1b[33m✗ candidate {} failed verification\x1b[0m", i + 1);
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
                opts.candidates
            );
            Ok(())
        }
    };

    cleanup(&created);
    result
}

/// Run one candidate `hi` in its worktree, with its own verify-loop.
fn run_candidate(opts: &BestOf, worktree: &Path, temperature: f32) -> Result<()> {
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
    let _ = status;
    Ok(())
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
        bail!("git add failed: {}", String::from_utf8_lossy(&add.stderr).trim());
    }

    let diff = Command::new("git")
        .current_dir(worktree)
        .args(["diff", "--cached", "HEAD"])
        .output()
        .context("git diff in worktree")?;
    if !diff.status.success() {
        bail!("git diff failed: {}", String::from_utf8_lossy(&diff.stderr).trim());
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
