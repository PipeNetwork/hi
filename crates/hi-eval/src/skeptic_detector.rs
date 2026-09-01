//! Detector eval for the `/goal team` skeptic reviewer.
//!
//! The end-to-end "does the skeptic pay" eval is expensive and noisy (long
//! episodes × trials × configs). This measures the reviewer **directly and
//! cheaply**: mine bug-fix commits and present each two ways —
//!
//! - the **forward** diff (the fix itself: a correct change → the reviewer should
//!   APPROVE), and
//! - the **reversed** diff (undoing the fix, re-introducing the bug → the reviewer
//!   should OBJECT).
//!
//! Run the real `skeptic_review` over both via `hi --skeptic-review` and report
//! recall (of flawed diffs, how many it caught), specificity (of correct diffs,
//! how many it left alone), and precision. A reviewer that objects to everything
//! has no precision; one that approves everything has no recall — both useless.
//! This quantifies which, before spending on full episodes.
//!
//! Caveat: labels are commit-message-derived, so the corpus is noisy (a "fix"
//! subject isn't always a crisp acceptance criterion). Averaged over N commits
//! the forward/reversed signal still separates a discriminating reviewer from a
//! rubber-stamp; treat absolute numbers as directional, not exact.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

pub struct Options {
    pub repo: PathBuf,
    pub hi_bin: PathBuf,
    pub reviewer: String,
    pub provider_args: Vec<String>,
    pub n: usize,
    pub max_diff_lines: usize,
    pub concurrency: usize,
}

struct Item {
    commit: String,
    intent: String,
    diff: String,
    /// A reversed (bug-reintroducing) diff — the reviewer should OBJECT.
    flawed: bool,
}

pub async fn run(opts: Options) -> Result<()> {
    let items = mine(&opts.repo, opts.n, opts.max_diff_lines)?;
    if items.is_empty() {
        bail!(
            "no usable fix commits mined from {} (need git 'fix' commits with small diffs)",
            opts.repo.display()
        );
    }
    eprintln!(
        "mined {} labeled diffs ({} fix commits × forward+reversed); reviewing with {} …",
        items.len(),
        items.len() / 2,
        opts.reviewer
    );

    let sem = Arc::new(Semaphore::new(opts.concurrency.max(1)));
    let opts = Arc::new(opts);
    let mut handles = Vec::new();
    for item in items {
        let sem = sem.clone();
        let opts = opts.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore");
            let verdict = review_one(&opts, &item).await;
            (item, verdict)
        }));
    }

    // Confusion matrix over (flawed?, objected?).
    let (mut tp, mut miss, mut tn, mut fp, mut errs) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for h in handles {
        let (item, verdict) = h.await.expect("join");
        match verdict {
            Err(e) => {
                eprintln!(
                    "  · {} ({}): error — {e:#}",
                    short(&item.commit),
                    label(&item)
                );
                errs += 1;
            }
            Ok(objected) => {
                match (item.flawed, objected) {
                    (true, true) => tp += 1,    // caught a flaw
                    (true, false) => miss += 1, // missed a flaw
                    (false, false) => tn += 1,  // correctly approved
                    (false, true) => fp += 1,   // false alarm
                }
                eprintln!(
                    "  · {} ({}): {}",
                    short(&item.commit),
                    label(&item),
                    if objected { "OBJECT" } else { "approve" }
                );
            }
        }
    }
    report(tp, miss, tn, fp, errs, &opts.reviewer);
    Ok(())
}

/// Mine up to `n` *corrective/additive* commits (fix/add/implement/handle/…),
/// each → a forward (correct — achieves the subject) and reversed (flawed — undoes
/// it) item. Filters to focused diffs (≤ `max_diff_lines`) and dedups identical
/// diffs (cherry-picks). Additive subjects make the reversed diff *clearly*
/// wrong, which is the label we rely on.
fn mine(repo: &Path, n: usize, max_diff_lines: usize) -> Result<Vec<Item>> {
    let out = std::process::Command::new("git")
        .current_dir(repo)
        .args(["log", "--no-merges", "--format=%H%x09%s", "-n", "500"])
        .output()
        .context("git log")?;
    if !out.status.success() {
        bail!("git log failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let mut items = Vec::new();
    let mut seen_diffs = std::collections::HashSet::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if items.len() >= n * 2 {
            break;
        }
        let Some((hash, subject)) = line.split_once('\t') else {
            continue;
        };
        if !is_corrective(subject) {
            continue;
        }
        let fwd = git_diff(repo, hash, false)?;
        let n_lines = fwd.lines().count();
        if fwd.trim().is_empty() || n_lines < 3 || n_lines > max_diff_lines {
            continue;
        }
        if !seen_diffs.insert(fwd.clone()) {
            continue; // identical diff already used (cherry-pick / duplicate)
        }
        let rev = git_diff(repo, hash, true)?;
        if rev.trim().is_empty() {
            continue;
        }
        let intent = clean_intent(subject);
        items.push(Item {
            commit: hash.to_string(),
            intent: intent.clone(),
            diff: fwd,
            flawed: false,
        });
        items.push(Item {
            commit: hash.to_string(),
            intent,
            diff: rev,
            flawed: true,
        });
    }
    Ok(items)
}

/// Whether a commit subject describes adding or fixing behaviour — so undoing it
/// (the reversed diff) is clearly wrong. Skips pure refactor/rename/docs/format
/// commits, whose reversal isn't obviously a defect.
fn is_corrective(subject: &str) -> bool {
    let s = subject.trim().to_ascii_lowercase();
    const VERBS: &[&str] = &[
        "fix",
        "add",
        "implement",
        "handle",
        "support",
        "ensure",
        "guard",
        "prevent",
        "restore",
        "correct",
        "exclude",
        "harden",
        "raise",
        "avoid",
        "validate",
        "enable",
    ];
    VERBS
        .iter()
        .any(|v| s.starts_with(v) || s.contains(&format!(" {v}")) || s.contains(&format!(":{v}")))
}

/// The commit's diff (`--format=` suppresses the message header so the intent
/// only reaches the reviewer via the sub-goal). `-R` reverses it.
fn git_diff(repo: &Path, hash: &str, reverse: bool) -> Result<String> {
    let mut args = vec!["show", "--no-color", "--format="];
    if reverse {
        args.push("-R");
    }
    args.push(hash);
    let out = std::process::Command::new("git")
        .current_dir(repo)
        .args(&args)
        .output()
        .context("git show")?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Strip a leading conventional-commit `fix:` / `fix(scope):` / `fix ` prefix so
/// the sub-goal reads as the intent, not "this is a fix" (which would leak the
/// label to the reviewer).
fn clean_intent(subject: &str) -> String {
    let s = subject.trim();
    if let Some(idx) = s.find(':')
        && idx < 20
        && s[..idx].to_ascii_lowercase().starts_with("fix")
    {
        return s[idx + 1..].trim().to_string();
    }
    if s.len() > 4 && s[..3].eq_ignore_ascii_case("fix") && s.as_bytes()[3] == b' ' {
        return s[4..].trim().to_string();
    }
    s.to_string()
}

/// Review one item via `hi --skeptic-review`; returns whether the reviewer
/// objected.
async fn review_one(opts: &Options, item: &Item) -> Result<bool> {
    let payload = serde_json::json!({
        "objective": item.intent,
        "sub_goal": item.intent,
        "diff": item.diff,
    })
    .to_string();
    let mut child = tokio::process::Command::new(&opts.hi_bin)
        .args(&opts.provider_args)
        .arg("--model")
        .arg(&opts.reviewer)
        .arg("--skeptic-review")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("spawn hi --skeptic-review")?;
    {
        let mut stdin = child.stdin.take().context("child stdin")?;
        stdin.write_all(payload.as_bytes()).await?;
        // Drop closes stdin → EOF, so `hi` stops reading and reviews.
    }
    let out = child.wait_with_output().await.context("waiting for hi")?;
    if !out.status.success() {
        bail!("hi --skeptic-review exited {}", out.status);
    }
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parsing verdict JSON from hi")?;
    Ok(v["objected"].as_bool().unwrap_or(false))
}

fn report(tp: u32, miss: u32, tn: u32, fp: u32, errs: u32, reviewer: &str) {
    let recall = pct(tp, tp + miss);
    let specificity = pct(tn, tn + fp);
    let precision = pct(tp, tp + fp);
    println!("\n=== skeptic detector eval — reviewer: {reviewer} ===");
    println!(
        "flawed (reversed) diffs → should OBJECT:   caught {tp:>3}   missed {miss:>3}   recall      {recall:>3.0}%"
    );
    println!(
        "correct (forward) diffs → should APPROVE: approved {tn:>3}   false-alarm {fp:>3}   specificity {specificity:>3.0}%"
    );
    println!(
        "precision (of its objections, share real):                          {precision:>3.0}%"
    );
    if errs > 0 {
        println!("errors (skipped):                                                   {errs}");
    }
    println!(
        "\nreading: recall = catches real misses; specificity = doesn't cry wolf. A useful\n\
         reviewer needs both — rubber-stamp scores recall 0, cry-wolf scores specificity 0."
    );
}

fn pct(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

fn short(commit: &str) -> &str {
    &commit[..commit.len().min(8)]
}

fn label(item: &Item) -> &'static str {
    if item.flawed { "flawed" } else { "correct" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrective_subjects_selected_refactors_skipped() {
        assert!(is_corrective("fix: handle zero division"));
        assert!(is_corrective("Add cross-request prefix-cache reuse"));
        assert!(is_corrective("hi-tui: harden the loop manager"));
        assert!(is_corrective("worktree: exclude pycache from merges"));
        // Non-corrective subjects (reversing them isn't clearly a defect) skipped.
        assert!(!is_corrective("Rename Foo to Bar"));
        assert!(!is_corrective("docs: tweak README wording"));
        assert!(!is_corrective("bump version to 0.2"));
    }

    #[test]
    fn intent_strips_fix_prefix() {
        assert_eq!(
            clean_intent("fix: handle zero division"),
            "handle zero division"
        );
        assert_eq!(
            clean_intent("fix(hi-gguf): restore match arms"),
            "restore match arms"
        );
        assert_eq!(
            clean_intent("Fix pipenetwork default config resolution"),
            "pipenetwork default config resolution"
        );
        // Non-fix subjects pass through unchanged (used verbatim as the intent).
        assert_eq!(
            clean_intent("Add cross-request cache"),
            "Add cross-request cache"
        );
    }
}
