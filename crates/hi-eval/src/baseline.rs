//! Eval north-star baseline capture and regression compare.
//!
//! `eval-baseline/core-0.2.json` holds the locked coding metrics for the 0.2
//! matrix. Capture after a full provider-backed run; compare subsequent runs
//! against it so solve-rate / false-verified / cost regressions are explicit.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::reporting::{EvaluationSummary, FailureBucketCounts};

/// On-disk baseline schema (v2). Null metric fields mean "not yet captured".
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CodingBaseline {
    pub schema_version: u32,
    /// RFC3339 timestamp when metrics were written, or null if placeholder.
    pub captured_at: Option<String>,
    /// Model route string from the capture run (e.g. `openrouter/fusion`).
    pub model_route: Option<String>,
    pub trials: u32,
    /// Task roots included in the capture (relative paths).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suites: Vec<String>,
    /// Config names included (e.g. baseline, verify).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub configs: Vec<String>,
    pub task_count: Option<usize>,
    pub cell_count: Option<usize>,
    pub solve_rate: Option<f64>,
    pub candidate_pass_rate: Option<f64>,
    pub false_verified_count: Option<usize>,
    pub false_verified_rate: Option<f64>,
    pub infrastructure_error_rate: Option<f64>,
    pub cost_per_solved: Option<f64>,
    pub tokens_per_solved: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_buckets: Option<FailureBucketCounts>,
    /// Free-form note (placeholder explanation or capture provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl CodingBaseline {
    pub fn placeholder() -> Self {
        Self {
            schema_version: 2,
            captured_at: None,
            model_route: None,
            trials: 3,
            suites: default_north_star_suites(),
            configs: vec!["baseline".into(), "verify".into()],
            task_count: None,
            cell_count: None,
            solve_rate: None,
            candidate_pass_rate: None,
            false_verified_count: None,
            false_verified_rate: None,
            infrastructure_error_rate: None,
            cost_per_solved: None,
            tokens_per_solved: None,
            failure_buckets: None,
            note: Some(
                "Populate from the first scheduled provider-backed 0.2 evaluation \
                 over the north-star ladder (tasks + vloop-dense + hidden); until \
                 then solve-rate and cost fields remain null. Integrity and \
                 infrastructure thresholds are enforced immediately."
                    .into(),
            ),
        }
    }

    pub fn is_captured(&self) -> bool {
        self.captured_at.is_some() && self.solve_rate.is_some()
    }

    pub fn from_summary(
        summary: &EvaluationSummary,
        model_route: Option<String>,
        trials: u32,
        suites: Vec<String>,
        configs: Vec<String>,
    ) -> Self {
        Self {
            schema_version: 2,
            captured_at: Some(utc_now_rfc3339()),
            model_route,
            trials,
            suites,
            configs,
            task_count: Some(summary.task_count),
            cell_count: Some(summary.cell_count),
            solve_rate: Some(summary.solve_rate),
            candidate_pass_rate: Some(summary.candidate_pass_rate),
            false_verified_count: Some(summary.false_verified_count),
            false_verified_rate: Some(summary.false_verified_rate),
            infrastructure_error_rate: Some(summary.infrastructure_error_rate),
            cost_per_solved: summary.cost_per_solved,
            tokens_per_solved: summary.tokens_per_solved,
            failure_buckets: Some(summary.failure_buckets.clone()),
            note: Some("Captured from hi-eval summary.json via --write-baseline.".into()),
        }
    }
}

/// Default ladder for the coding north star (ordered by cost/signal).
pub fn default_north_star_suites() -> Vec<String> {
    vec![
        "bench/tasks".into(),
        "bench/spec".into(),
        "bench/vloop-dense".into(),
        "bench/hidden".into(),
    ]
}

pub fn default_baseline_path() -> std::path::PathBuf {
    std::path::PathBuf::from("eval-baseline/core-0.2.json")
}

pub fn load_baseline(path: &Path) -> Result<CodingBaseline> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading baseline {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing baseline {}", path.display()))
}

pub fn write_baseline(path: &Path, baseline: &CodingBaseline) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(baseline)? + "\n";
    std::fs::write(path, body).with_context(|| format!("writing baseline {}", path.display()))?;
    Ok(())
}

/// Capture a baseline from a completed run's `summary.json`.
pub fn capture_from_summary_file(
    summary_path: &Path,
    baseline_path: &Path,
    model_route: Option<String>,
    trials: u32,
    suites: Vec<String>,
    configs: Vec<String>,
) -> Result<CodingBaseline> {
    let text = std::fs::read_to_string(summary_path)
        .with_context(|| format!("reading summary {}", summary_path.display()))?;
    let summary: EvaluationSummary = serde_json::from_str(&text)
        .with_context(|| format!("parsing summary {}", summary_path.display()))?;
    let baseline = CodingBaseline::from_summary(&summary, model_route, trials, suites, configs);
    write_baseline(baseline_path, &baseline)?;
    Ok(baseline)
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct BaselineDelta {
    pub metric: String,
    pub baseline: Option<f64>,
    pub current: Option<f64>,
    pub delta: Option<f64>,
    /// Higher is better for this metric (solve_rate); false for rates we want down.
    pub higher_is_better: bool,
    pub regressed: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct BaselineCompareReport {
    pub baseline_captured: bool,
    pub deltas: Vec<BaselineDelta>,
    pub regressions: usize,
    pub note: String,
}

/// Compare a live summary against a locked baseline.
///
/// When the baseline is still a placeholder (`solve_rate` null), comparison is
/// informational only — no regressions are flagged (nothing to regress against).
///
/// `solve_rate_tolerance` / `false_verified_tolerance` are absolute deltas
/// (e.g. `0.05` = five percentage points) that count as a regression.
pub fn compare_to_baseline(
    baseline: &CodingBaseline,
    summary: &EvaluationSummary,
    solve_rate_tolerance: f64,
    false_verified_tolerance: f64,
) -> BaselineCompareReport {
    if !baseline.is_captured() {
        return BaselineCompareReport {
            baseline_captured: false,
            deltas: Vec::new(),
            regressions: 0,
            note: "baseline not yet captured — run a full matrix then \
                   `hi-eval --write-baseline=<summary.json>`"
                .into(),
        };
    }

    let mut deltas = Vec::new();
    deltas.push(metric_delta(
        "solve_rate",
        baseline.solve_rate,
        Some(summary.solve_rate),
        true,
        solve_rate_tolerance,
    ));
    deltas.push(metric_delta(
        "candidate_pass_rate",
        baseline.candidate_pass_rate,
        Some(summary.candidate_pass_rate),
        true,
        solve_rate_tolerance,
    ));
    deltas.push(metric_delta(
        "false_verified_rate",
        baseline.false_verified_rate,
        Some(summary.false_verified_rate),
        false,
        false_verified_tolerance,
    ));
    deltas.push(metric_delta(
        "infrastructure_error_rate",
        baseline.infrastructure_error_rate,
        Some(summary.infrastructure_error_rate),
        false,
        0.05,
    ));
    if baseline.cost_per_solved.is_some() || summary.cost_per_solved.is_some() {
        // Cost: allow 25% relative rise before flagging.
        let regressed = match (baseline.cost_per_solved, summary.cost_per_solved) {
            (Some(b), Some(c)) if b > 0.0 => c > b * 1.25,
            _ => false,
        };
        deltas.push(BaselineDelta {
            metric: "cost_per_solved".into(),
            baseline: baseline.cost_per_solved,
            current: summary.cost_per_solved,
            delta: match (baseline.cost_per_solved, summary.cost_per_solved) {
                (Some(b), Some(c)) => Some(c - b),
                _ => None,
            },
            higher_is_better: false,
            regressed,
        });
    }
    if baseline.tokens_per_solved.is_some() || summary.tokens_per_solved.is_some() {
        let regressed = match (baseline.tokens_per_solved, summary.tokens_per_solved) {
            (Some(b), Some(c)) if b > 0.0 => c > b * 1.25,
            _ => false,
        };
        deltas.push(BaselineDelta {
            metric: "tokens_per_solved".into(),
            baseline: baseline.tokens_per_solved,
            current: summary.tokens_per_solved,
            delta: match (baseline.tokens_per_solved, summary.tokens_per_solved) {
                (Some(b), Some(c)) => Some(c - b),
                _ => None,
            },
            higher_is_better: false,
            regressed,
        });
    }

    let regressions = deltas.iter().filter(|d| d.regressed).count();
    let note = if regressions == 0 {
        "no regressions vs locked baseline".into()
    } else {
        format!("{regressions} metric(s) regressed vs locked baseline")
    };
    BaselineCompareReport {
        baseline_captured: true,
        deltas,
        regressions,
        note,
    }
}

fn metric_delta(
    name: &str,
    baseline: Option<f64>,
    current: Option<f64>,
    higher_is_better: bool,
    tolerance: f64,
) -> BaselineDelta {
    let delta = match (baseline, current) {
        (Some(b), Some(c)) => Some(c - b),
        _ => None,
    };
    let regressed = match (baseline, current, delta) {
        (Some(_), Some(_), Some(d)) if higher_is_better => d < -tolerance,
        (Some(_), Some(_), Some(d)) if !higher_is_better => d > tolerance,
        _ => false,
    };
    BaselineDelta {
        metric: name.into(),
        baseline,
        current,
        delta,
        higher_is_better,
        regressed,
    }
}

pub fn print_compare_report(report: &BaselineCompareReport) {
    println!("\n=== Baseline compare ===");
    if !report.baseline_captured {
        println!("{}", report.note);
        return;
    }
    println!(
        "{:<28} {:>10} {:>10} {:>10} {}",
        "metric", "baseline", "current", "delta", "status"
    );
    for d in &report.deltas {
        let status = if d.regressed { "REGRESS" } else { "ok" };
        println!(
            "{:<28} {:>10} {:>10} {:>10} {status}",
            d.metric,
            fmt_opt(d.baseline),
            fmt_opt(d.current),
            fmt_opt(d.delta),
        );
    }
    println!("{}", report.note);
}

fn fmt_opt(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.4}"),
        None => "—".into(),
    }
}

/// Minimal RFC3339 UTC timestamp without pulling a time crate.
fn utc_now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Good enough for capture provenance; not leap-second precise.
    let days = secs / 86_400;
    let time = secs % 86_400;
    let hour = time / 3600;
    let min = (time % 3600) / 60;
    let sec = time % 60;
    // Civil date from Unix day count (Howard Hinnant algorithm).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Ensure a baseline file exists (write placeholder if missing).
pub fn ensure_placeholder(path: &Path) -> Result<CodingBaseline> {
    if path.is_file() {
        return load_baseline(path);
    }
    let baseline = CodingBaseline::placeholder();
    write_baseline(path, &baseline)?;
    Ok(baseline)
}

/// Exit code helper: 0 ok, 2 when regressions found against a captured baseline.
pub fn compare_exit_code(report: &BaselineCompareReport) -> i32 {
    if report.baseline_captured && report.regressions > 0 {
        2
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::{EvaluationSummary, FailureBucketCounts};

    fn sample_summary(solve: f64, false_v: f64) -> EvaluationSummary {
        EvaluationSummary {
            schema_version: 2,
            task_count: 10,
            cell_count: 30,
            candidate_count: 30,
            solved_cell_count: (solve * 30.0).round() as usize,
            candidate_pass_rate: solve,
            solve_at_n: solve,
            pass_at_k: Some(solve),
            pass_at_k_k: Some(3),
            false_verified_count: (false_v * 30.0).round() as usize,
            false_verified_rate: false_v,
            infrastructure_error_rate: 0.0,
            solve_rate: solve,
            cost_per_solved: Some(0.05),
            tokens_per_solved: Some(12_000.0),
            failure_buckets: FailureBucketCounts {
                no_edits: 2,
                compile: 1,
                logic: 3,
                error: 0,
                unknown: 0,
            },
            groups: Vec::new(),
        }
    }

    #[test]
    fn placeholder_is_not_captured() {
        assert!(!CodingBaseline::placeholder().is_captured());
    }

    #[test]
    fn capture_marks_baseline_captured() {
        let summary = sample_summary(0.7, 0.05);
        let baseline = CodingBaseline::from_summary(
            &summary,
            Some("openrouter/fusion".into()),
            3,
            default_north_star_suites(),
            vec!["verify".into()],
        );
        assert!(baseline.is_captured());
        assert_eq!(baseline.solve_rate, Some(0.7));
        assert_eq!(baseline.false_verified_rate, Some(0.05));
        assert!(baseline.captured_at.is_some());
    }

    #[test]
    fn compare_flags_solve_rate_regression() {
        let baseline = CodingBaseline::from_summary(
            &sample_summary(0.8, 0.02),
            Some("m".into()),
            3,
            vec![],
            vec![],
        );
        let worse = sample_summary(0.6, 0.02);
        let report = compare_to_baseline(&baseline, &worse, 0.05, 0.05);
        assert!(report.baseline_captured);
        assert!(report.regressions >= 1);
        assert!(
            report
                .deltas
                .iter()
                .any(|d| d.metric == "solve_rate" && d.regressed)
        );
    }

    #[test]
    fn compare_flags_false_verified_rise() {
        let baseline = CodingBaseline::from_summary(
            &sample_summary(0.7, 0.02),
            Some("m".into()),
            3,
            vec![],
            vec![],
        );
        let worse = sample_summary(0.7, 0.15);
        let report = compare_to_baseline(&baseline, &worse, 0.05, 0.05);
        assert!(
            report
                .deltas
                .iter()
                .any(|d| d.metric == "false_verified_rate" && d.regressed)
        );
    }

    #[test]
    fn placeholder_compare_is_informational() {
        let report = compare_to_baseline(
            &CodingBaseline::placeholder(),
            &sample_summary(0.5, 0.1),
            0.05,
            0.05,
        );
        assert!(!report.baseline_captured);
        assert_eq!(report.regressions, 0);
        assert_eq!(compare_exit_code(&report), 0);
    }

    #[test]
    fn round_trip_baseline_file() {
        let dir = std::env::temp_dir().join(format!(
            "hi-eval-baseline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("core-0.2.json");
        let baseline = CodingBaseline::from_summary(
            &sample_summary(0.75, 0.03),
            Some("test/model".into()),
            3,
            default_north_star_suites(),
            vec!["baseline".into(), "verify".into()],
        );
        write_baseline(&path, &baseline).unwrap();
        let loaded = load_baseline(&path).unwrap();
        assert_eq!(loaded.solve_rate, baseline.solve_rate);
        assert_eq!(loaded.model_route.as_deref(), Some("test/model"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ensure_placeholder_writes_missing_file() {
        let dir = std::env::temp_dir().join(format!(
            "hi-eval-ph-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("core-0.2.json");
        let b = ensure_placeholder(&path).unwrap();
        assert!(!b.is_captured());
        assert!(path.is_file());
        let _ = std::fs::remove_dir_all(dir);
    }
}
