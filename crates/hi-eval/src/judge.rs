//! Deterministic process/budget judge over a hi `--report` tape.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JudgeRules {
    pub process: ProcessRules,
    pub budget: BudgetRules,
    pub run: RunRules,
}

/// Live-only runner hints. Replay (`hi-eval judge --suite`) ignores these.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunRules {
    /// Sequential `--max-steps` values on the same `--session-file`.
    /// `[3, 8]` means first invocation stops at 3 model calls, then resume
    /// with 8 more. Empty keeps the default single-shot path.
    pub steps: Vec<u32>,
    /// Seed the session with a user-turn image of this many chars (capped at
    /// 2_000_000) plus filler turns so resume elides it.
    pub seed_image_chars: Option<u64>,
    /// Seed an old tool result of this many chars so resume bounding is visible.
    pub seed_tool_result_chars: Option<u64>,
    /// Workspace-relative prefixes restored from fixture/ before the oracle
    /// and dropped from `allowed_changes` (inner-task side effects, e.g. `bug/`).
    pub ignore_change_prefixes: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProcessRules {
    pub forbid_tools: Vec<String>,
    pub require_tools: Vec<String>,
    pub require_verify: bool,
    pub mutate_after_read: bool,
    pub no_root_cargo_test: bool,
    /// `driver.py` must slice/cap tool output (self-similar-driver).
    pub require_output_slice: bool,
    /// Mutate tools (`edit`/`write`/`apply_patch`/…) must not target these
    /// prefixes. Reads and inner-host edits are allowed.
    pub forbid_path_prefixes: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetRules {
    pub max_request_tokens_est: Option<u64>,
    pub max_tool_result_chars: Option<u64>,
    pub max_image_chars_after_elide: Option<u64>,
    pub max_same_path_rereads: Option<u32>,
    pub max_write_existing_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JudgeVerdict {
    Pass,
    Fail,
    Skip,
}

#[derive(Clone, Debug, Serialize)]
pub struct JudgeViolation {
    pub rule: String,
    pub got: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct JudgeReport {
    pub outcome: JudgeVerdict,
    pub process: JudgeVerdict,
    pub budget: JudgeVerdict,
    pub violations: Vec<JudgeViolation>,
}

impl JudgeReport {
    pub fn ok(&self) -> bool {
        !matches!(self.process, JudgeVerdict::Fail) && !matches!(self.budget, JudgeVerdict::Fail)
    }
}

pub fn load_rules(path: &Path) -> Result<JudgeRules> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading judge rules {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

pub fn judge_report(report: &Value, rules: &JudgeRules) -> JudgeReport {
    let mut violations = Vec::new();
    judge_process(report, &rules.process, &mut violations);
    judge_budget(report, &rules.budget, &mut violations);
    let process_fail = violations.iter().any(|v| v.rule.starts_with("process."));
    let budget_fail = violations.iter().any(|v| v.rule.starts_with("budget."));
    JudgeReport {
        // Outcome is the hidden oracle, scored elsewhere. Replay tapes skip it.
        outcome: JudgeVerdict::Skip,
        process: if process_fail {
            JudgeVerdict::Fail
        } else {
            JudgeVerdict::Pass
        },
        budget: if budget_fail {
            JudgeVerdict::Fail
        } else {
            JudgeVerdict::Pass
        },
        violations,
    }
}

fn tools(report: &Value) -> impl Iterator<Item = &Value> {
    report
        .get("telemetry")
        .and_then(|t| t.get("tool_timeline"))
        .and_then(|t| t.as_array())
        .into_iter()
        .flatten()
}

fn tool_name(entry: &Value) -> &str {
    entry.get("tool").and_then(|v| v.as_str()).unwrap_or("")
}

/// Paths this call touched: the single `path` field, plus every
/// `effects.file_changes[].path` so multi-file `apply_patch` / `multi_edit`
/// still participate in process rules.
fn tool_paths(entry: &Value) -> Vec<String> {
    let mut paths = Vec::new();
    if let Some(path) = entry.get("path").and_then(|v| v.as_str()) {
        if !path.is_empty() {
            paths.push(path.to_string());
        }
    }
    if let Some(changes) = entry
        .pointer("/effects/file_changes")
        .and_then(|v| v.as_array())
    {
        for change in changes {
            if let Some(path) = change.get("path").and_then(|v| v.as_str()) {
                if !path.is_empty() && !paths.iter().any(|existing| existing == path) {
                    paths.push(path.to_string());
                }
            }
        }
    }
    paths
}

fn judge_process(report: &Value, rules: &ProcessRules, violations: &mut Vec<JudgeViolation>) {
    let names: Vec<&str> = tools(report).map(tool_name).collect();
    for forbidden in &rules.forbid_tools {
        if names.iter().any(|n| *n == forbidden) {
            violations.push(JudgeViolation {
                rule: format!("process.forbid_tools.{forbidden}"),
                got: forbidden.clone(),
            });
        }
    }
    for required in &rules.require_tools {
        if !names.iter().any(|n| *n == required) {
            violations.push(JudgeViolation {
                rule: format!("process.require_tools.{required}"),
                got: "missing".into(),
            });
        }
    }
    if rules.require_verify {
        let rounds = report
            .pointer("/telemetry/verify_rounds")
            .or_else(|| report.pointer("/verification/rounds"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if rounds == 0 {
            violations.push(JudgeViolation {
                rule: "process.require_verify".into(),
                got: "0".into(),
            });
        }
    }
    if rules.mutate_after_read {
        let mut read_paths = std::collections::BTreeSet::new();
        for entry in tools(report) {
            let name = tool_name(entry);
            let paths = tool_paths(entry);
            if name == "read" {
                for path in &paths {
                    read_paths.insert(path.clone());
                }
            }
            if matches!(name, "edit" | "multi_edit" | "apply_patch" | "write") {
                for path in &paths {
                    if !read_paths.contains(path) {
                        violations.push(JudgeViolation {
                            rule: "process.mutate_after_read".into(),
                            got: path.clone(),
                        });
                    }
                }
            }
        }
    }
    if rules.no_root_cargo_test {
        for entry in tools(report) {
            if tool_name(entry) != "bash" {
                continue;
            }
            let cmd = entry.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if is_root_cargo_test(cmd) {
                violations.push(JudgeViolation {
                    rule: "process.no_root_cargo_test".into(),
                    got: cmd.to_string(),
                });
            }
        }
    }
    if !rules.forbid_path_prefixes.is_empty() {
        for entry in tools(report) {
            let name = tool_name(entry);
            if !matches!(
                name,
                "write" | "edit" | "multi_edit" | "apply_patch" | "delete" | "move"
            ) {
                continue;
            }
            for path in tool_paths(entry) {
                if let Some(prefix) = rules
                    .forbid_path_prefixes
                    .iter()
                    .find(|prefix| path_under_prefix(&path, prefix))
                {
                    violations.push(JudgeViolation {
                        rule: format!("process.forbid_path_prefixes.{prefix}"),
                        got: path,
                    });
                }
            }
        }
    }
    if rules.require_output_slice {
        let bounded = report
            .pointer("/harness/driver_bounds_output")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !bounded {
            violations.push(JudgeViolation {
                rule: "process.require_output_slice".into(),
                got: "driver.py does not slice tool output".into(),
            });
        }
    }
}

/// Syntactic check used by the live runner: the written driver must cap or
/// slice tool results instead of concatenating them unbounded.
pub fn driver_bounds_output(src: &str) -> bool {
    src.contains("[:") || src.contains("[-") || src.contains("slice(")
}

pub fn path_under_prefix(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

fn is_root_cargo_test(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    let has_cargo_test = lower.contains("cargo test") || lower.contains("cargo t ");
    if !has_cargo_test {
        return false;
    }
    !(lower.contains(" -p ")
        || lower.contains(" --package ")
        || lower.contains("--manifest-path")
        || lower.contains("-p="))
}

fn judge_budget(report: &Value, rules: &BudgetRules, violations: &mut Vec<JudgeViolation>) {
    let requests = report
        .pointer("/telemetry/requests")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(max) = rules.max_request_tokens_est {
        let got = requests
            .iter()
            .filter_map(|r| r.get("input_tokens_est").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0);
        if got > max {
            violations.push(JudgeViolation {
                rule: "budget.max_request_tokens_est".into(),
                got: got.to_string(),
            });
        }
    }
    if let Some(max) = rules.max_tool_result_chars {
        let from_requests = requests
            .iter()
            .filter_map(|r| r.get("max_tool_result_chars").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0);
        if from_requests > max {
            violations.push(JudgeViolation {
                rule: "budget.max_tool_result_chars".into(),
                got: from_requests.to_string(),
            });
        }
    }
    if let Some(max) = rules.max_image_chars_after_elide {
        let got = requests
            .iter()
            .filter_map(|r| r.get("max_image_chars").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0);
        let elided = requests
            .iter()
            .filter_map(|r| r.get("elided_images").and_then(|v| v.as_u64()))
            .sum::<u64>();
        if got > max {
            violations.push(JudgeViolation {
                rule: "budget.max_image_chars_after_elide".into(),
                got: got.to_string(),
            });
        }
        if got > 800 && elided == 0 {
            violations.push(JudgeViolation {
                rule: "budget.image_elision_misses".into(),
                got: got.to_string(),
            });
        }
    }
    if let Some(max) = rules.max_same_path_rereads {
        let mut counts = std::collections::BTreeMap::<String, u32>::new();
        for entry in tools(report) {
            if tool_name(entry) != "read" {
                continue;
            }
            for path in tool_paths(entry) {
                *counts.entry(path).or_default() += 1;
            }
        }
        for (path, n) in counts {
            if n > max {
                violations.push(JudgeViolation {
                    rule: "budget.max_same_path_rereads".into(),
                    got: format!("{path}:{n}"),
                });
            }
        }
    }
    if let Some(max) = rules.max_write_existing_bytes {
        for entry in tools(report) {
            if tool_name(entry) != "write" {
                continue;
            }
            let after = entry
                .pointer("/effects/file_changes/0/after_len")
                .and_then(|v| v.as_u64())
                .or_else(|| entry.get("result_chars").and_then(|v| v.as_u64()))
                .unwrap_or(0);
            let before = entry
                .pointer("/effects/file_changes/0/before_len")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if before > 0 && after > max {
                violations.push(JudgeViolation {
                    rule: "budget.max_write_existing_bytes".into(),
                    got: after.to_string(),
                });
            }
        }
    }
}

pub fn judge_suite(suite: &Path) -> Result<Vec<(PathBuf, JudgeReport)>> {
    let mut out = Vec::new();
    if !suite.is_dir() {
        bail!("{} is not a directory", suite.display());
    }
    let mut tasks: Vec<PathBuf> = std::fs::read_dir(suite)
        .with_context(|| format!("reading {}", suite.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    tasks.sort();
    for task in tasks {
        let rules_path = task.join("judge.toml");
        if !rules_path.is_file() {
            continue;
        }
        let rules = load_rules(&rules_path)?;
        let tape_dir = task.join("tape");
        if !tape_dir.is_dir() {
            continue;
        }
        let mut tapes: Vec<PathBuf> = std::fs::read_dir(&tape_dir)
            .with_context(|| format!("reading {}", tape_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        tapes.sort();
        for tape in tapes {
            let raw = std::fs::read_to_string(&tape)
                .with_context(|| format!("reading {}", tape.display()))?;
            let report: Value = serde_json::from_str(&raw)
                .with_context(|| format!("parsing {}", tape.display()))?;
            let judged = judge_report(&report, &rules);
            let expect_fail = tape
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.contains("fail"));
            if expect_fail && judged.ok() {
                bail!(
                    "{}: expected fail tape to violate rules, got pass",
                    tape.display()
                );
            }
            if !expect_fail && !judged.ok() {
                bail!(
                    "{}: pass tape failed judge: {}",
                    tape.display(),
                    serde_json::to_string(&judged.violations).unwrap_or_default()
                );
            }
            out.push((tape, judged));
        }
    }
    Ok(out)
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if let Some(suite) = flag(args, "--suite") {
        let results = judge_suite(Path::new(&suite))?;
        if results.is_empty() {
            bail!("no judge.toml + tape/*.json pairs under {suite}");
        }
        for (path, report) in &results {
            println!(
                "{}  process={:?} budget={:?} violations={}",
                path.display(),
                report.process,
                report.budget,
                report.violations.len()
            );
        }
        println!("judged {} tape(s)", results.len());
        return Ok(());
    }
    let report_path = flag(args, "--report").context("judge requires --report <file>")?;
    let rules_path = flag(args, "--rules").context("judge requires --rules <judge.toml>")?;
    let raw =
        std::fs::read_to_string(&report_path).with_context(|| format!("reading {report_path}"))?;
    let report: Value = serde_json::from_str(&raw)?;
    let rules = load_rules(Path::new(&rules_path))?;
    let judged = judge_report(&report, &rules);
    println!("{}", serde_json::to_string_pretty(&judged)?);
    if !judged.ok() {
        std::process::exit(1);
    }
    Ok(())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rules_write_edit() -> JudgeRules {
        JudgeRules {
            process: ProcessRules {
                forbid_tools: vec!["write".into()],
                require_tools: vec!["edit".into()],
                ..ProcessRules::default()
            },
            budget: BudgetRules {
                max_image_chars_after_elide: Some(800),
                max_tool_result_chars: Some(50_000),
                ..BudgetRules::default()
            },
            ..JudgeRules::default()
        }
    }

    #[test]
    fn pass_tape_allows_edit_and_elided_image() {
        let report = json!({
            "outcome": {},
            "telemetry": {
                "verify_rounds": 1,
                "tool_timeline": [
                    {"tool": "read", "path": "a.rs"},
                    {"tool": "edit", "path": "a.rs"}
                ],
                "requests": [{
                    "input_tokens_est": 1200,
                    "max_tool_result_chars": 400,
                    "max_image_chars": 40,
                    "elided_images": 1
                }]
            }
        });
        let judged = judge_report(&report, &rules_write_edit());
        assert!(judged.ok(), "{:?}", judged.violations);
        assert_eq!(judged.process, JudgeVerdict::Pass);
        assert_eq!(judged.budget, JudgeVerdict::Pass);
    }

    #[test]
    fn image_bomb_tape_fails_budget() {
        let report = json!({
            "telemetry": {
                "tool_timeline": [],
                "requests": [{
                    "input_tokens_est": 100,
                    "max_image_chars": 2_000_000,
                    "elided_images": 0
                }]
            }
        });
        let judged = judge_report(&report, &rules_write_edit());
        assert!(!judged.ok());
        assert!(
            judged.violations.iter().any(|v| v.rule.contains("image")),
            "{:?}",
            judged.violations
        );
    }

    #[test]
    fn write_overwrite_fails_process() {
        let report = json!({
            "telemetry": {
                "tool_timeline": [{"tool": "write", "path": "a.rs"}]
            }
        });
        let judged = judge_report(&report, &rules_write_edit());
        assert_eq!(judged.process, JudgeVerdict::Fail);
        assert!(
            judged
                .violations
                .iter()
                .any(|v| v.rule.contains("forbid_tools.write")),
            "{:?}",
            judged.violations
        );
    }

    #[test]
    fn root_cargo_test_fails_process() {
        let report = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "edit", "path": "solution.py"},
                    {"tool": "bash", "command": "cargo test --quiet"}
                ]
            }
        });
        let rules = JudgeRules {
            process: ProcessRules {
                no_root_cargo_test: true,
                require_tools: vec!["edit".into()],
                ..ProcessRules::default()
            },
            ..JudgeRules::default()
        };
        let judged = judge_report(&report, &rules);
        assert_eq!(judged.process, JudgeVerdict::Fail);
        assert!(
            judged
                .violations
                .iter()
                .any(|v| v.rule.contains("no_root_cargo_test")),
            "{:?}",
            judged.violations
        );
    }

    #[test]
    fn cargo_t_at_root_fails_process() {
        let report = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "edit", "path": "solution.py"},
                    {"tool": "bash", "command": "CARGO t --quiet"}
                ]
            }
        });
        let rules = JudgeRules {
            process: ProcessRules {
                no_root_cargo_test: true,
                require_tools: vec!["edit".into()],
                ..ProcessRules::default()
            },
            ..JudgeRules::default()
        };
        assert_eq!(judge_report(&report, &rules).process, JudgeVerdict::Fail);
    }

    #[test]
    fn package_local_cargo_test_is_allowed() {
        let report = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "edit", "path": "solution.py"},
                    {"tool": "bash", "command": "cargo test -p pkg --quiet"}
                ]
            }
        });
        let rules = JudgeRules {
            process: ProcessRules {
                no_root_cargo_test: true,
                require_tools: vec!["edit".into()],
                ..ProcessRules::default()
            },
            ..JudgeRules::default()
        };
        assert!(judge_report(&report, &rules).ok());
    }

    #[test]
    fn reread_budget_fails_when_same_path_is_read_too_often() {
        let report = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "read", "path": "solution.py"},
                    {"tool": "read", "path": "solution.py"},
                    {"tool": "read", "path": "solution.py"},
                    {"tool": "edit", "path": "solution.py"}
                ]
            }
        });
        let rules = JudgeRules {
            budget: BudgetRules {
                max_same_path_rereads: Some(2),
                ..BudgetRules::default()
            },
            ..JudgeRules::default()
        };
        let judged = judge_report(&report, &rules);
        assert_eq!(judged.budget, JudgeVerdict::Fail);
    }

    #[test]
    fn mutate_under_forbidden_prefix_fails_process() {
        let rules = JudgeRules {
            process: ProcessRules {
                require_tools: vec!["write".into()],
                forbid_path_prefixes: vec!["bug/".into()],
                ..ProcessRules::default()
            },
            ..JudgeRules::default()
        };
        let cheat = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "write", "path": "driver.py"},
                    {"tool": "edit", "path": "bug/solution.py"}
                ]
            }
        });
        let judged = judge_report(&cheat, &rules);
        assert_eq!(judged.process, JudgeVerdict::Fail);
        assert!(
            judged
                .violations
                .iter()
                .any(|v| v.rule.contains("forbid_path_prefixes")),
            "{:?}",
            judged.violations
        );
        let via_host = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "write", "path": "driver.py"},
                    {"tool": "read", "path": "bug/solution.py"},
                    {"tool": "bash", "command": "python3 host.py"}
                ]
            }
        });
        assert!(judge_report(&via_host, &rules).ok());
        let via_patch = json!({
            "telemetry": {
                "tool_timeline": [
                    {"tool": "write", "path": "driver.py"},
                    {
                        "tool": "apply_patch",
                        "path": "",
                        "effects": {"file_changes": [{"path": "bug/solution.py"}]}
                    }
                ]
            }
        });
        assert_eq!(judge_report(&via_patch, &rules).process, JudgeVerdict::Fail);
    }

    #[test]
    fn output_slice_rule_reads_harness_flag() {
        let rules = JudgeRules {
            process: ProcessRules {
                require_output_slice: true,
                ..ProcessRules::default()
            },
            ..JudgeRules::default()
        };
        let missing = json!({"telemetry": {"tool_timeline": []}});
        assert!(!judge_report(&missing, &rules).ok());
        let ok = json!({
            "telemetry": {"tool_timeline": []},
            "harness": {"driver_bounds_output": true}
        });
        assert!(judge_report(&ok, &rules).ok());
        assert!(driver_bounds_output(
            "content = read.get('content','')[:4000]"
        ));
        assert!(!driver_bounds_output("prompt += result"));
    }

    #[test]
    fn run_section_parses_and_is_ignored_by_replay() {
        let rules: JudgeRules = toml::from_str(
            r#"
[process]
require_tools = ["edit"]
[run]
steps = [3, 8]
seed_image_chars = 2000000
"#,
        )
        .unwrap();
        assert_eq!(rules.run.steps, vec![3, 8]);
        assert_eq!(rules.run.seed_image_chars, Some(2_000_000));
    }
}
