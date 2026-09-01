//! Two-binary A/B: compare absolute `hi` binaries on the same eval matrix
//! with swapped trial order, SHA-256 identity, and process-pass deltas.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::artifacts::command_output_with_timeout;
use crate::results::RunResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AbSide {
    Baseline,
    Candidate,
}

impl AbSide {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AbBinaryMeta {
    pub path: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AbTaskDelta {
    pub task: String,
    pub baseline_process_pass_rate: Option<f64>,
    pub candidate_process_pass_rate: Option<f64>,
    pub baseline_solve_at_n: f64,
    pub candidate_solve_at_n: f64,
    pub classification: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct AbOverall {
    pub baseline_process_pass_rate: Option<f64>,
    pub candidate_process_pass_rate: Option<f64>,
    pub process_pass_rate_delta: Option<f64>,
    pub baseline_solve_at_n: f64,
    pub candidate_solve_at_n: f64,
    pub solve_at_n_delta: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AbReport {
    pub schema_version: u32,
    pub baseline: AbBinaryMeta,
    pub candidate: AbBinaryMeta,
    pub trials: usize,
    pub overall: AbOverall,
    pub tasks: Vec<AbTaskDelta>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AbMeta {
    pub schema_version: u32,
    pub baseline: AbBinaryMeta,
    pub candidate: AbBinaryMeta,
    pub trials: usize,
    pub configs: Vec<String>,
    pub models: Vec<String>,
    pub tasks: Vec<String>,
    pub trial_order_rule: &'static str,
    pub env: BTreeMap<String, String>,
}

pub fn create_trial_order(trial_index: usize) -> [AbSide; 2] {
    if trial_index.is_multiple_of(2) {
        [AbSide::Baseline, AbSide::Candidate]
    } else {
        [AbSide::Candidate, AbSide::Baseline]
    }
}

pub fn require_absolute_executable(path: &str, label: &str) -> Result<PathBuf> {
    let path = PathBuf::from(path);
    if !path.is_absolute() {
        bail!(
            "{label} binary must be an absolute path, got {}",
            path.display()
        );
    }
    if !path.exists() {
        bail!("{label} binary does not exist: {}", path.display());
    }
    let meta = std::fs::metadata(&path)
        .with_context(|| format!("stat {} binary {}", label, path.display()))?;
    if !meta.is_file() {
        bail!("{label} binary is not a file: {}", path.display());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            bail!("{label} binary is not executable: {}", path.display());
        }
    }
    std::fs::canonicalize(&path)
        .with_context(|| format!("resolving {label} binary path {}", path.display()))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn capture_version(bin: &Path) -> Result<String> {
    let mut cmd = Command::new(bin);
    cmd.arg("--version");
    let output = command_output_with_timeout(&mut cmd, Duration::from_secs(5))
        .with_context(|| format!("running {} --version", bin.display()))?;
    if !output.success() {
        bail!(
            "{} --version failed: {}",
            bin.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        bail!("{} --version produced no stdout", bin.display());
    }
    Ok(stdout)
}

pub fn redact_sensitive_value(key: &str, value: &str) -> String {
    let upper = key.to_ascii_uppercase();
    if ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
        .iter()
        .any(|needle| upper.contains(needle))
    {
        "[redacted]".into()
    } else {
        value.to_string()
    }
}

pub fn snapshot_eval_env() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (key, value) in std::env::vars() {
        if !(key.starts_with("HI_")
            || key.starts_with("PIPENETWORK_")
            || key.starts_with("OPENAI_"))
        {
            continue;
        }
        out.insert(key.clone(), redact_sensitive_value(&key, &value));
    }
    out
}

pub fn inspect_binary(path: &Path) -> Result<AbBinaryMeta> {
    Ok(AbBinaryMeta {
        path: path.display().to_string(),
        sha256: sha256_file(path)?,
        version: capture_version(path)?,
    })
}

pub fn classify_delta(baseline: Option<f64>, candidate: Option<f64>) -> &'static str {
    match (baseline, candidate) {
        (Some(b), Some(c)) if (c - b).abs() < 1e-12 => {
            if c > 0.0 {
                "unchanged-pass"
            } else {
                "unchanged-fail"
            }
        }
        (Some(b), Some(c)) if c > b => "improved",
        (Some(b), Some(c)) if c < b => "regressed",
        _ => "inconclusive",
    }
}

pub fn build_ab_report(
    baseline: AbBinaryMeta,
    candidate: AbBinaryMeta,
    trials: usize,
    baseline_results: &[RunResult],
    candidate_results: &[RunResult],
) -> AbReport {
    let mut task_names: Vec<String> = baseline_results
        .iter()
        .chain(candidate_results.iter())
        .map(|row| row.task.clone())
        .collect();
    task_names.sort();
    task_names.dedup();

    let tasks = task_names
        .into_iter()
        .map(|task| {
            let base: Vec<&RunResult> = baseline_results
                .iter()
                .filter(|row| row.task == task)
                .collect();
            let cand: Vec<&RunResult> = candidate_results
                .iter()
                .filter(|row| row.task == task)
                .collect();
            let baseline_process = process_pass_rate(base.iter().copied());
            let candidate_process = process_pass_rate(cand.iter().copied());
            AbTaskDelta {
                classification: classify_delta(baseline_process, candidate_process),
                baseline_process_pass_rate: baseline_process,
                candidate_process_pass_rate: candidate_process,
                baseline_solve_at_n: solve_at_n(base.iter().copied()),
                candidate_solve_at_n: solve_at_n(cand.iter().copied()),
                task,
            }
        })
        .collect();

    let baseline_process = process_pass_rate(baseline_results.iter());
    let candidate_process = process_pass_rate(candidate_results.iter());
    let baseline_solve = solve_at_n(baseline_results.iter());
    let candidate_solve = solve_at_n(candidate_results.iter());
    AbReport {
        schema_version: 1,
        baseline,
        candidate,
        trials,
        overall: AbOverall {
            process_pass_rate_delta: match (baseline_process, candidate_process) {
                (Some(b), Some(c)) => Some(c - b),
                _ => None,
            },
            baseline_process_pass_rate: baseline_process,
            candidate_process_pass_rate: candidate_process,
            baseline_solve_at_n: baseline_solve,
            candidate_solve_at_n: candidate_solve,
            solve_at_n_delta: candidate_solve - baseline_solve,
        },
        tasks,
    }
}

fn process_pass_rate<'a>(rows: impl Iterator<Item = &'a RunResult>) -> Option<f64> {
    let judged: Vec<&serde_json::Value> = rows
        .flat_map(|row| &row.candidates)
        .filter_map(|candidate| candidate.judge.as_ref())
        .collect();
    if judged.is_empty() {
        return None;
    }
    let passes = judged
        .iter()
        .filter(|report| report.get("process").and_then(|v| v.as_str()) == Some("pass"))
        .count();
    Some(passes as f64 / judged.len() as f64)
}

fn solve_at_n<'a>(rows: impl Iterator<Item = &'a RunResult>) -> f64 {
    let rows: Vec<&RunResult> = rows.collect();
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().filter(|row| row.passed).count() as f64 / rows.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{Candidate, RunResult};

    fn empty_result(task: &str, passed: bool, process: Option<&str>) -> RunResult {
        RunResult {
            config: "baseline".into(),
            model: "test".into(),
            task: task.into(),
            trial: 0,
            passed,
            fail: None,
            provider_error_kind: None,
            compat_fallbacks_used: Vec::new(),
            changed_files: Vec::new(),
            verify_output_summary: String::new(),
            failure_confidence: None,
            candidates: vec![Candidate {
                index: 0,
                temperature: 0.0,
                seed: None,
                passed,
                fail: None,
                agent_process: crate::results::AgentProcessOutcome::ExitedSuccessfully,
                agent_exit_code: Some(0),
                agent_output_summary: String::new(),
                agent_output_truncated: false,
                reported_success: passed,
                false_verified: false,
                actual_model_route: None,
                turn_outcome: None,
                provider_error_kind: None,
                failure_mode: None,
                model_outcome: None,
                partial_artifact: None,
                trace: None,
                compat_fallbacks_used: Vec::new(),
                changed_files: Vec::new(),
                verify_output_summary: String::new(),
                failure_confidence: None,
                tokens: 0,
                input_tokens: 0,
                session_tokens: 0,
                session_input_tokens: 0,
                cost: None,
                seconds: 0.0,
                patch: String::new(),
                patch_truncated: false,
                checks: Vec::new(),
                trajectory: Default::default(),
                growth: Vec::new(),
                judge: process.map(|verdict| {
                    serde_json::json!({"process": verdict, "budget": "pass", "violations": []})
                }),
                max_request_tokens: None,
                tape: None,
            }],
            tokens: 0,
            input_tokens: 0,
            seconds: 0.0,
            mcp_model: None,
            trajectory: Default::default(),
            growth: Vec::new(),
        }
    }

    #[test]
    fn trial_order_swaps_on_odd_index() {
        assert_eq!(create_trial_order(0), [AbSide::Baseline, AbSide::Candidate]);
        assert_eq!(create_trial_order(1), [AbSide::Candidate, AbSide::Baseline]);
        assert_eq!(create_trial_order(2), [AbSide::Baseline, AbSide::Candidate]);
    }

    #[test]
    fn relative_binary_is_rejected() {
        let err = require_absolute_executable("target/debug/hi", "baseline")
            .unwrap_err()
            .to_string();
        assert!(err.contains("absolute"), "{err}");
    }

    #[test]
    fn sha256_file_hashes_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        std::fs::write(&path, b"hello-ab").unwrap();
        let digest = sha256_file(&path).unwrap();
        assert_eq!(digest, format!("{:x}", Sha256::digest(b"hello-ab")));
    }

    #[test]
    fn redact_helper_masks_secret_keys() {
        assert_eq!(
            redact_sensitive_value("HI_API_KEY", "sk-live"),
            "[redacted]"
        );
        assert_eq!(
            redact_sensitive_value("HI_MODEL", "pipe/coder"),
            "pipe/coder"
        );
        assert_eq!(
            redact_sensitive_value("PIPENETWORK_TOKEN", "abc"),
            "[redacted]"
        );
    }

    #[test]
    fn classify_and_report_process_delta() {
        let meta = AbBinaryMeta {
            path: "/tmp/hi".into(),
            sha256: "abc".into(),
            version: "hi 0.0.0".into(),
        };
        let baseline = vec![empty_result("local-discovery", true, Some("fail"))];
        let candidate = vec![empty_result("local-discovery", true, Some("pass"))];
        let report = build_ab_report(meta.clone(), meta, 1, &baseline, &candidate);
        assert_eq!(report.tasks[0].classification, "improved");
        assert_eq!(report.overall.process_pass_rate_delta, Some(1.0));
        assert_eq!(classify_delta(Some(1.0), Some(1.0)), "unchanged-pass");
        assert_eq!(classify_delta(Some(0.0), Some(0.0)), "unchanged-fail");
        assert_eq!(classify_delta(Some(1.0), Some(0.0)), "regressed");
        assert_eq!(classify_delta(None, Some(1.0)), "inconclusive");
    }
}
