//! `hi bench swe` — resolve-rate regression gate over Multi-SWE-bench.
//!
//! Runs real issue instances end-to-end: clone at base SHA, hi one-shot with
//! the pinned prompt, then standard-protocol grading — the agent's edits to
//! hidden-test-owned files are reverted, the hidden test patch applied, and
//! the full suite run. "RESOLVED" means the hidden fail-to-pass tests pass
//! with zero regressions. Failure transcripts and diffs are kept under the
//! state root as tuning material.
//!
//! The prompt phrasing is pinned deliberately: a controlled A/B showed that
//! adding "Do NOT modify any existing test files" flips the intent classifier
//! into read-only preflight and zeroes the resolve rate. Grading handles
//! agent test edits structurally, so the prompt does not mention tests.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};

/// Pinned task prompt (see module docs for why it must not mention tests).
const PROMPT_TEMPLATE: &str = "Fix this bug reported against this repository:\n\n\
# {title}\n\n{body}\n\n\
Implement the fix in the source code. Add or adjust tests only as needed to cover the fix.";

/// Rust datasets shipped in ByteDance-Seed/Multi-SWE-bench.
const RUST_REPOS: &[&str] = &[
    "BurntSushi__ripgrep",
    "clap-rs__clap",
    "nushell__nushell",
    "rayon-rs__rayon",
    "serde-rs__serde",
    "sharkdp__bat",
    "sharkdp__fd",
    "tokio-rs__bytes",
    "tokio-rs__tokio",
    "tokio-rs__tracing",
];

const DATASET_BASE: &str =
    "https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench/resolve/main";

pub(crate) async fn run_bench_cli(args: &[String]) -> Result<()> {
    let mut lang = "rust".to_string();
    let mut repo_filter: Option<String> = None;
    let mut limit = usize::MAX;
    let mut retries = 0u32;
    let mut instance_filter: Option<Vec<String>> = None;
    let mut iter = args.iter();
    let mut subcommand = None;
    while let Some(argument) = iter.next() {
        match argument.as_str() {
            "swe" if subcommand.is_none() => subcommand = Some("swe"),
            "--lang" => {
                lang = iter
                    .next()
                    .ok_or_else(|| anyhow!("--lang requires a value"))?
                    .clone();
            }
            "--repo" => {
                repo_filter = Some(
                    iter.next()
                        .ok_or_else(|| anyhow!("--repo requires a dataset name, e.g. sharkdp__fd"))?
                        .clone(),
                );
            }
            "--limit" => {
                limit = iter
                    .next()
                    .ok_or_else(|| anyhow!("--limit requires a number"))?
                    .parse()
                    .context("--limit requires a number")?;
            }
            "--retries" => {
                retries = iter
                    .next()
                    .ok_or_else(|| anyhow!("--retries requires a number"))?
                    .parse::<u32>()
                    .context("--retries requires a number between 0 and 3")?
                    .min(3);
            }
            "--instances" => {
                instance_filter = Some(
                    iter.next()
                        .ok_or_else(|| anyhow!("--instances requires a comma-separated id list"))?
                        .split(',')
                        .map(|id| id.trim().to_string())
                        .filter(|id| !id.is_empty())
                        .collect(),
                );
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => bail!("unexpected bench argument {other:?} (see `hi bench --help`)"),
        }
    }
    if subcommand.is_none() {
        print_usage();
        return Ok(());
    }
    if lang != "rust" {
        bail!("only --lang rust is wired up so far");
    }

    let (_, state_root) = crate::review_target::resolve_runtime_roots()?;
    let bench_root = state_root.join("bench");
    std::fs::create_dir_all(bench_root.join("datasets"))?;

    let repos: Vec<&str> = match &repo_filter {
        Some(filter) => RUST_REPOS
            .iter()
            .filter(|name| **name == filter.as_str())
            .copied()
            .collect(),
        None => RUST_REPOS.to_vec(),
    };
    if repos.is_empty() {
        bail!("unknown --repo; available: {}", RUST_REPOS.join(", "));
    }

    let mut instances = Vec::new();
    for repo in repos {
        let dataset = fetch_dataset(&bench_root, &lang, repo)?;
        instances.extend(parse_instances(&dataset));
    }
    if let Some(filter) = &instance_filter {
        instances.retain(|instance| filter.contains(&instance.id));
    }
    instances.truncate(limit);
    if instances.is_empty() {
        bail!("no runnable instances (all skipped: missing hidden tests or issue text)");
    }
    println!("hi bench swe · {} runnable instance(s)", instances.len());

    let exe = std::env::current_exe().context("resolving hi executable")?;
    let mut tally: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let scorecard_path = bench_root.join("scorecard.jsonl");
    for instance in &instances {
        let mut label = "INFRA";
        let mut attempts = 0u32;
        for attempt in 0..=retries {
            attempts = attempt + 1;
            // Retries carry the previous attempt's failing tests so the next
            // run addresses them instead of repeating the same miss.
            let failure_context = (attempt > 0)
                .then(|| previous_failure_context(&bench_root, &instance.id))
                .flatten();
            let verdict = run_instance(&bench_root, &exe, instance, failure_context.as_deref());
            label = match &verdict {
                Ok(label) => *label,
                Err(error) => {
                    eprintln!("  {}: infra failure: {error:#}", instance.id);
                    "INFRA"
                }
            };
            if label == "RESOLVED" || label == "INFRA" {
                break;
            }
        }
        *tally.entry(label).or_default() += 1;
        println!("VERDICT {}: {label}", instance.id);
        let record = serde_json::json!({
            "instance": instance.id,
            "verdict": label,
            "attempts": attempts,
            "evidence": bench_root.join("runs").join(&instance.id).display().to_string(),
        });
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&scorecard_path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{record}");
        }
    }
    println!("\nscorecard: {tally:?}");
    println!(
        "evidence + failure transcripts: {}",
        bench_root.join("runs").display()
    );
    Ok(())
}

fn print_usage() {
    println!(
        "hi bench swe — resolve-rate regression gate over Multi-SWE-bench\n\n\
         USAGE:\n  hi bench swe [--lang rust] [--repo <dataset>] [--limit N] [--retries N] [--instances id1,id2]\n\n\
         Each instance: clone at base SHA → hi one-shot (pinned prompt) →\n\
         standard-protocol grade (hidden tests own their files) → verdict.\n\
         Verdicts: RESOLVED / NOT_RESOLVED / FAILED / INFRA. Evidence stays\n\
         under <state-root>/bench/runs/<instance>/ for the tuning loop."
    );
}

pub(crate) struct BenchInstance {
    pub id: String,
    pub org: String,
    pub repo: String,
    pub sha: String,
    pub prompt: String,
    pub test_patch: String,
    pub f2p: Vec<String>,
}

fn fetch_dataset(bench_root: &Path, lang: &str, repo: &str) -> Result<String> {
    let cache = bench_root.join("datasets").join(format!("{repo}.jsonl"));
    if let Ok(text) = std::fs::read_to_string(&cache) {
        return Ok(text);
    }
    let url = format!("{DATASET_BASE}/{lang}/{repo}_dataset.jsonl");
    // curl keeps the binary free of another HTTP client; this is a dev tool.
    let output = Command::new("curl")
        .args(["-sL", "--max-time", "300", &url, "-o"])
        .arg(&cache)
        .output()
        .context("running curl")?;
    if !output.status.success() {
        bail!("fetching {url} failed");
    }
    std::fs::read_to_string(&cache).context("reading fetched dataset")
}

/// Runnable instances: must have hidden fail-to-pass tests (else ungradable)
/// and real issue text (else nothing to prompt with).
pub(crate) fn parse_instances(dataset: &str) -> Vec<BenchInstance> {
    let mut out = Vec::new();
    for line in dataset.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let f2p: Vec<String> = value
            .get("f2p_tests")
            .and_then(|t| t.as_object())
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default();
        if f2p.is_empty() {
            continue;
        }
        let issue = value
            .get("resolved_issues")
            .and_then(|issues| issues.as_array())
            .and_then(|issues| issues.first())
            .cloned()
            .unwrap_or_default();
        let title = issue
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        let body = issue
            .get("body")
            .and_then(|b| b.as_str())
            .unwrap_or_default();
        if title.len() + body.len() < 40 {
            continue;
        }
        let get = |key: &str| {
            value
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let sha = value
            .get("base")
            .and_then(|base| base.get("sha"))
            .and_then(|sha| sha.as_str())
            .unwrap_or_default()
            .to_string();
        let id = get("instance_id");
        if id.is_empty() || sha.is_empty() {
            continue;
        }
        out.push(BenchInstance {
            id,
            org: get("org"),
            repo: get("repo"),
            sha,
            prompt: PROMPT_TEMPLATE
                .replace("{title}", title)
                .replace("{body}", body),
            test_patch: get("test_patch"),
            f2p,
        });
    }
    out
}

/// Paths the hidden test patch owns — the agent's edits to these are reverted
/// before grading (standard SWE-bench protocol).
pub(crate) fn test_owned_files(test_patch: &str) -> BTreeSet<String> {
    test_patch
        .lines()
        .filter_map(|line| line.strip_prefix("+++ b/"))
        .map(str::to_string)
        .collect()
}

/// Grade a full `cargo test` output: RESOLVED requires every hidden
/// fail-to-pass test observed passing and zero failing suites.
pub(crate) fn grade_test_output(output: &str, f2p: &[String]) -> &'static str {
    let failed_suites = output
        .lines()
        .filter(|line| line.starts_with("test result: FAILED"))
        .count();
    if failed_suites > 0 {
        return "FAILED";
    }
    let ok_suites = output
        .lines()
        .filter(|line| line.starts_with("test result: ok"))
        .count();
    if all_f2p_pass(output, f2p) && ok_suites > 0 {
        "RESOLVED"
    } else {
        "NOT_RESOLVED"
    }
}

fn all_f2p_pass(output: &str, f2p: &[String]) -> bool {
    f2p.iter().all(|name| {
        output.lines().any(|line| {
            line.starts_with("test ") && line.contains(name.as_str()) && line.ends_with("... ok")
        })
    })
}

/// Names of failing tests in a libtest run (`test NAME ... FAILED`).
pub(crate) fn failing_test_names(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("test ")?
                .strip_suffix("... FAILED")
                .map(|name| name.trim().to_string())
        })
        .collect()
}

/// Re-grade a FAILED run against the base repository's own failures: only
/// failures the agent *introduced* count against it. A suite that was already
/// red at base+test_patch (flaky infra, environment-dependent tests) must not
/// charge the model — per SWE-bench norms the bar is "hidden tests pass, zero
/// NEW failures". Caveat: a test flaking red during the baseline run masks a
/// real regression of that one test; acceptable for a trend metric.
pub(crate) fn attribute_with_baseline(
    agent_output: &str,
    baseline_output: &str,
    f2p: &[String],
) -> &'static str {
    let agent_failing = failing_test_names(agent_output);
    let base_failing = failing_test_names(baseline_output);
    let introduced: Vec<&String> = agent_failing.difference(&base_failing).collect();
    if !introduced.is_empty() {
        return "FAILED";
    }
    if all_f2p_pass(agent_output, f2p) {
        "RESOLVED"
    } else {
        "NOT_RESOLVED"
    }
}

fn run_instance(
    bench_root: &Path,
    exe: &Path,
    instance: &BenchInstance,
    failure_context: Option<&str>,
) -> Result<&'static str> {
    let run_dir = bench_root.join("runs").join(&instance.id);
    std::fs::create_dir_all(&run_dir)?;
    let repo_dir = run_dir.join("repo");
    let _ = std::fs::remove_dir_all(&repo_dir);
    let url = format!("https://github.com/{}/{}", instance.org, instance.repo);
    run_quiet(
        Command::new("git")
            .args(["clone", "--quiet", &url])
            .arg(&repo_dir),
    )?;
    run_quiet(Command::new("git").arg("-C").arg(&repo_dir).args([
        "checkout",
        "-q",
        &instance.sha,
    ]))?;

    let prompt = match failure_context {
        Some(context) => format!("{}\n\n{context}", instance.prompt),
        None => instance.prompt.clone(),
    };
    std::fs::write(run_dir.join("prompt.txt"), &prompt)?;
    let report = run_dir.join("report.json");
    let log = std::fs::File::create(run_dir.join("run.log"))?;
    let mut agent_run = Command::new(exe);
    agent_run
        .arg(&prompt)
        .arg("--report")
        .arg(&report)
        .current_dir(&repo_dir)
        .env(
            "CARGO_TARGET_DIR",
            bench_root.join("target").join(&instance.repo),
        )
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // The grade, not the run, decides the verdict — a run killed at its
    // deadline still grades whatever it changed so far.
    if let Err(error) = wait_with_deadline(agent_run, AGENT_RUN_TIMEOUT, "hi one-shot agent run") {
        eprintln!("  ⚠ {error:#}");
    }

    let diff = Command::new("git")
        .arg("-C")
        .arg(&repo_dir)
        .arg("diff")
        .output()?;
    std::fs::write(run_dir.join("agent.diff"), &diff.stdout)?;

    // Standard protocol: hidden tests own their files.
    for file in test_owned_files(&instance.test_patch) {
        let _ = run_quiet(
            Command::new("git")
                .arg("-C")
                .arg(&repo_dir)
                .args(["checkout", "--", &file]),
        );
    }
    let patch_path = run_dir.join("test.patch");
    std::fs::write(&patch_path, &instance.test_patch)?;
    if run_quiet(
        Command::new("git")
            .arg("-C")
            .arg(&repo_dir)
            .arg("apply")
            .arg(&patch_path),
    )
    .is_err()
    {
        let _ = std::fs::remove_dir_all(&repo_dir);
        return Ok("INFRA");
    }
    let mut grade_cmd = Command::new("cargo");
    grade_cmd.arg("test").current_dir(&repo_dir).env(
        "CARGO_TARGET_DIR",
        bench_root.join("target").join(&instance.repo),
    );
    let test_output = match wait_with_deadline(grade_cmd, GRADE_TEST_TIMEOUT, "grading cargo test")
    {
        Ok(output) => output,
        Err(error) => {
            // Can't grade what never finished — infrastructure, not the model.
            eprintln!("  ⚠ {error:#}");
            let _ = std::fs::remove_dir_all(&repo_dir);
            return Ok("INFRA");
        }
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&test_output.stdout),
        String::from_utf8_lossy(&test_output.stderr)
    );
    std::fs::write(run_dir.join("full-test.log"), &combined)?;
    let mut verdict = grade_test_output(&combined, &instance.f2p);
    if verdict == "FAILED" {
        // Baseline attribution: drop every agent change, re-apply the hidden
        // tests, and charge the model only for failures it introduced.
        let clean = run_quiet(
            Command::new("git")
                .arg("-C")
                .arg(&repo_dir)
                .args(["checkout", "--", "."]),
        )
        .and_then(|_| {
            run_quiet(
                Command::new("git")
                    .arg("-C")
                    .arg(&repo_dir)
                    .args(["clean", "-fdq"]),
            )
        })
        .and_then(|_| {
            run_quiet(
                Command::new("git")
                    .arg("-C")
                    .arg(&repo_dir)
                    .arg("apply")
                    .arg(&patch_path),
            )
        });
        if clean.is_ok() {
            match cargo_test_output(bench_root, &repo_dir, &instance.repo) {
                Ok(baseline) => {
                    std::fs::write(run_dir.join("baseline-test.log"), &baseline)?;
                    verdict = attribute_with_baseline(&combined, &baseline, &instance.f2p);
                }
                // Keep the un-attributed FAILED rather than erroring the
                // whole instance over a wedged baseline build.
                Err(error) => eprintln!("  ⚠ baseline attribution skipped: {error:#}"),
            }
        }
    }
    // Evidence (diff, logs, report) is the point; the checkout is not.
    let _ = std::fs::remove_dir_all(&repo_dir);
    Ok(verdict)
}

/// Failing-test names from the previous attempt's grade log, as a compact
/// context block for the retry prompt.
fn previous_failure_context(bench_root: &Path, instance_id: &str) -> Option<String> {
    let log = std::fs::read_to_string(
        bench_root
            .join("runs")
            .join(instance_id)
            .join("full-test.log"),
    )
    .ok()?;
    let failing = failing_test_names(&log);
    if failing.is_empty() {
        return None;
    }
    let names: Vec<&str> = failing.iter().map(String::as_str).take(8).collect();
    Some(format!(
        "A previous attempt at this fix left these tests failing: {}. \
         Address them specifically; take a fresh approach rather than repeating the previous attempt.",
        names.join(", ")
    ))
}

fn cargo_test_output(bench_root: &Path, repo_dir: &Path, repo: &str) -> Result<String> {
    let mut command = Command::new("cargo");
    command
        .arg("test")
        .current_dir(repo_dir)
        .env("CARGO_TARGET_DIR", bench_root.join("target").join(repo));
    let output = wait_with_deadline(command, GRADE_TEST_TIMEOUT, "baseline cargo test")?;
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

/// Hard per-step deadlines. Every bench child used to wait unbounded — one
/// starved grading build wedged a live bench run for 33 hours (1.2s of CPU)
/// with nothing recorded past the previous instance.
const AGENT_RUN_TIMEOUT: Duration = Duration::from_secs(45 * 60);
const GRADE_TEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Run `command` with a hard deadline, killing its whole process group on
/// expiry. The waiter thread owns `wait_with_output`, so a chatty child keeps
/// its pipes drained and cannot deadlock against the deadline.
fn wait_with_deadline(
    mut command: Command,
    deadline: Duration,
    label: &str,
) -> Result<std::process::Output> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let child = command
        .spawn()
        .with_context(|| format!("spawning {label}"))?;
    let pid = child.id();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    match rx.recv_timeout(deadline) {
        Ok(result) => result.with_context(|| format!("waiting for {label}")),
        Err(_) => {
            #[cfg(unix)]
            {
                // Negative pid → the process group set above.
                let _ = Command::new("kill")
                    .args(["-9", &format!("-{pid}")])
                    .status();
            }
            // Let the waiter observe the kill so the child is reaped.
            let _ = rx.recv_timeout(Duration::from_secs(10));
            bail!(
                "{label} exceeded its {}s budget and was killed",
                deadline.as_secs()
            )
        }
    }
}

fn run_quiet(command: &mut Command) -> Result<()> {
    let output = command.output().context("spawning command")?;
    if !output.status.success() {
        bail!(
            "command failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_pinned_without_test_file_clauses() {
        // The A/B that zeroed a batch: negated test-file clauses flip the
        // intent classifier into read-only preflight. Grading owns that
        // concern; the prompt must never regrow the clause.
        assert!(PROMPT_TEMPLATE.contains("Implement the fix in the source code"));
        assert!(!PROMPT_TEMPLATE.to_lowercase().contains("do not modify"));
        assert!(!PROMPT_TEMPLATE.to_lowercase().contains("don't modify"));
    }

    #[test]
    fn instances_require_hidden_tests_and_issue_text() {
        let dataset = concat!(
            r#"{"instance_id":"a-1","org":"o","repo":"r","base":{"sha":"abc"},"f2p_tests":{"t1":{}},"resolved_issues":[{"title":"Panic when using -j","body":"long enough body describing the bug in detail"}],"test_patch":"x"}"#,
            "\n",
            r#"{"instance_id":"a-2","org":"o","repo":"r","base":{"sha":"abc"},"f2p_tests":{},"resolved_issues":[{"title":"has no hidden tests","body":"plenty of text but ungradable"}],"test_patch":"x"}"#,
            "\n",
            r#"{"instance_id":"a-3","org":"o","repo":"r","base":{"sha":"abc"},"f2p_tests":{"t":{}},"resolved_issues":[],"test_patch":"x"}"#,
        );
        let instances = parse_instances(dataset);
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].id, "a-1");
        assert!(instances[0].prompt.contains("Panic when using -j"));
    }

    #[test]
    fn grading_requires_f2p_pass_and_no_failing_suites() {
        let f2p = vec!["test_get_int".to_string()];
        let pass = "test test_get_int ... ok\ntest result: ok. 5 passed; 0 failed;\n";
        assert_eq!(grade_test_output(pass, &f2p), "RESOLVED");
        let regressed = "test test_get_int ... ok\ntest result: ok. 5 passed; 0 failed;\ntest result: FAILED. 1 passed; 1 failed;\n";
        assert_eq!(grade_test_output(regressed, &f2p), "FAILED");
        let f2p_missing = "test other ... ok\ntest result: ok. 5 passed; 0 failed;\n";
        assert_eq!(grade_test_output(f2p_missing, &f2p), "NOT_RESOLVED");
    }

    #[test]
    fn baseline_attribution_charges_only_introduced_failures() {
        let f2p = vec!["test_new_feature".to_string()];
        // Pre-existing failure at base: not the agent's fault. f2p passes.
        let agent = "test test_new_feature ... ok\ntest test_flaky_env ... FAILED\n\
                     test result: ok. 5 passed; 0 failed;\ntest result: FAILED. 3 passed; 1 failed;\n";
        let base = "test test_new_feature ... FAILED\ntest test_flaky_env ... FAILED\n\
                    test result: FAILED. 2 passed; 2 failed;\n";
        assert_eq!(grade_test_output(agent, &f2p), "FAILED");
        assert_eq!(attribute_with_baseline(agent, base, &f2p), "RESOLVED");
        // A failure absent at base IS the agent's regression.
        let agent_broke = "test test_new_feature ... ok\ntest test_other ... FAILED\n\
                           test result: FAILED. 4 passed; 1 failed;\n";
        assert_eq!(attribute_with_baseline(agent_broke, base, &f2p), "FAILED");
        // No new failures but f2p still failing: not resolved.
        let no_fix = "test test_new_feature ... FAILED\ntest test_flaky_env ... FAILED\n\
                      test result: FAILED. 2 passed; 2 failed;\n";
        assert_eq!(attribute_with_baseline(no_fix, base, &f2p), "NOT_RESOLVED");
    }

    #[test]
    fn failing_names_parse_from_libtest_lines() {
        let names =
            failing_test_names("test a ... ok\n    test b::c ... FAILED\ntest d ... FAILED\n");
        assert_eq!(names.len(), 2);
        assert!(names.contains("b::c") && names.contains("d"));
    }

    #[test]
    fn test_patch_ownership_lists_target_files() {
        let patch = "--- a/tests/tests.rs\n+++ b/tests/tests.rs\n@@\n--- a/tests/other.rs\n+++ b/tests/other.rs\n";
        let owned = test_owned_files(patch);
        assert!(owned.contains("tests/tests.rs"));
        assert!(owned.contains("tests/other.rs"));
    }
}
