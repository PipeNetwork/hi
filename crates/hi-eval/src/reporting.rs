use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::results::{FailKind, RunResult};

pub fn print_summary(results: &[RunResult], task_count: usize, active: &[&Config], trials: usize) {
    println!("\n=== Results ({task_count} tasks × {trials} trial(s)) ===");
    println!(
        "{:<32} {:>11} {:>11} {:>11} {:>11} {:>10} {:>10}",
        "config", "cand pass", "solve@N", "pass@k", "tok/trial", "tok/task", "ctx/task"
    );
    let mut models: Vec<&str> = results.iter().map(|r| r.model.as_str()).collect();
    models.sort();
    models.dedup();
    let multi_model = models.len() > 1;
    for config in active {
        for model in &models {
            let rows: Vec<&RunResult> = results
                .iter()
                .filter(|r| r.config == config.name && r.model == *model)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let label = if multi_model {
                format!("{}@{}", config.name, short_model(model))
            } else {
                config.name.to_string()
            };

            // Cells solved by at least one candidate per trial → mean ± spread.
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
            let solve_at_n = if trials == 1 {
                format!("{}/{task_count}", per_trial[0])
            } else {
                format!("{mean:.1}±{std:.1}/{task_count}")
            };

            let candidate_total = rows.iter().map(|row| row.candidates.len()).sum::<usize>();
            let candidate_passes = rows
                .iter()
                .flat_map(|row| &row.candidates)
                .filter(|candidate| candidate.passed)
                .count();
            let candidate_rate = ratio(candidate_passes, candidate_total);
            let pass_k = group_pass_at_k(&rows, trials)
                .map_or_else(|| "n/a".to_string(), |rate| format!("{:.1}%", rate * 100.0));

            let tokens: u64 = rows.iter().map(|r| r.tokens).sum::<u64>() / trials as u64;
            let input_tokens: u64 =
                rows.iter().map(|r| r.input_tokens).sum::<u64>() / trials as u64;
            // The headline A/B number: mean tokens to attempt one task (summed across
            // a config's candidates). Compare it across condense on/off runs.
            let tok_per_task = tokens / task_count.max(1) as u64;
            // Context (input) tokens per task — the "how much context does hi send"
            // axis. Tune system-prompt/tools/compaction to drop this without
            // dropping solve@N (the Pi-efficiency loop).
            let ctx_per_task = input_tokens / task_count.max(1) as u64;
            println!(
                "{:<32} {:>10.1}% {:>11} {:>11} {:>11} {:>10} {:>10}",
                label,
                candidate_rate * 100.0,
                solve_at_n,
                pass_k,
                tokens,
                tok_per_task,
                ctx_per_task,
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

            // Trajectory diagnostic: how much steering did the representative
            // candidate need, on average? Lower is better — a clean turn needs 0
            // extra rounds. Verify rounds + recovery retries + repeat/continue
            // nudges, averaged across the config's cells. Plus the stall rate, so a
            // config that passes but only by repeatedly nudging a stuck model reads
            // as noisier than one that solves cleanly.
            let avg_extra: f64 = rows
                .iter()
                .map(|r| r.trajectory.extra_rounds() as f64)
                .sum::<f64>()
                / rows.len() as f64;
            let stalls = rows
                .iter()
                .filter(|r| r.trajectory.stalled_unfinished || r.trajectory.stalled_repeating)
                .count();
            if avg_extra > 0.0 || stalls > 0 {
                println!(
                    "           steer: {avg_extra:.1} extra rnd/cell · {stalls} stall(s) of {} cells",
                    rows.len()
                );
            }
            // Verify-repair diagnostic: how often the loop entered repair
            // (verify_rounds >= 2 means at least one round failed), how often
            // a repair episode ended in an oracle pass, and how many rounds
            // re-failed with an unchanged failure signature (repairs that
            // changed nothing — the costliest failure behavior).
            let repair_rows: Vec<&RunResult> = rows
                .iter()
                .filter(|r| r.trajectory.verify_rounds >= 2)
                .copied()
                .collect();
            let repeated: u32 = rows
                .iter()
                .map(|r| r.trajectory.repeated_verify_failures)
                .sum();
            if !repair_rows.is_empty() || repeated > 0 {
                let recovered = repair_rows.iter().filter(|r| r.passed).count();
                println!(
                    "           repair: entered {} of {} cells · recovered {recovered}/{} · {repeated} same-failure round(s)",
                    repair_rows.len(),
                    rows.len(),
                    repair_rows.len(),
                );
            }
            // Scheduler parallelism: average over cells with tool calls — the max
            // concurrent batch size and the share of calls that ran serially. A
            // config whose batches are mostly serial (max_concurrent ≈ 1, most runs
            // serial) isn't benefiting from the dep-aware scheduler; one with high
            // max_concurrent and a low serial share is. Skipped when no cell used
            // tools.
            let tool_rows: Vec<&RunResult> = rows
                .iter()
                .filter(|r| r.trajectory.tool_calls > 0)
                .copied()
                .collect();
            if !tool_rows.is_empty() {
                let avg_max: f64 = tool_rows
                    .iter()
                    .map(|r| r.trajectory.max_concurrent_batch as f64)
                    .sum::<f64>()
                    / tool_rows.len() as f64;
                let total_calls: u64 = tool_rows
                    .iter()
                    .map(|r| r.trajectory.tool_calls as u64)
                    .sum();
                let total_serial: u64 = tool_rows
                    .iter()
                    .map(|r| r.trajectory.serial_runs as u64)
                    .sum();
                let serial_pct = 100_u64
                    .checked_mul(total_serial)
                    .and_then(|p| p.checked_div(total_calls))
                    .unwrap_or(0);
                println!(
                    "           parallel: {avg_max:.1} max concurrent batch · {serial_pct}% of {total_calls} calls serial",
                );
            }

            // Context-growth curve (multi-turn drives only): the representative
            // trial's per-turn input-token trajectory — "how fast does context
            // accumulate vs. how far does the goal get". The exemplar is the row
            // that drove the most turns. Watch for ctx climbing while goal/done
            // stalls: that's the bloat the compaction/curation levers should flatten.
            if let Some(row) = rows
                .iter()
                .filter(|r| !r.growth.is_empty())
                .max_by_key(|r| r.growth.len())
            {
                println!("           growth (ctx tok · tools · sub-goals done):");
                for m in &row.growth {
                    let prog = if m.goal_total > 0 {
                        format!("{}/{}", m.goal_done, m.goal_total)
                    } else {
                        "-".to_string()
                    };
                    println!(
                        "             t{:<2} ctx={:>8} tools={:>3} goal={}",
                        m.turn, m.input_tokens, m.tool_calls, prog
                    );
                }
            }
        }
    }
}

/// Unbiased pass@k estimator for `c` successes among `n` exchangeable samples.
/// Returns `None` for an invalid population/sample size.
pub fn pass_at_k(n: usize, c: usize, k: usize) -> Option<f64> {
    if n == 0 || k == 0 || k > n || c > n {
        return None;
    }
    if n - c < k {
        return Some(1.0);
    }
    let all_fail = (0..k).fold(1.0, |probability, i| {
        probability * (n - c - i) as f64 / (n - i) as f64
    });
    Some(1.0 - all_fail)
}

fn group_pass_at_k(rows: &[&RunResult], k: usize) -> Option<f64> {
    let sampling_signatures = rows
        .iter()
        .flat_map(|row| &row.candidates)
        .map(|candidate| {
            (
                candidate.temperature.to_bits(),
                candidate.actual_model_route.as_deref(),
                candidate.seed.is_some(),
            )
        })
        .collect::<Vec<_>>();
    if !exchangeable_sampling_signatures(&sampling_signatures) {
        // A heterogeneous best-of ensemble is solve@N, never pass@k.
        return None;
    }
    let mut by_task: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for row in rows {
        let entry = by_task.entry(&row.task).or_default();
        entry.0 += row.candidates.len();
        entry.1 += row
            .candidates
            .iter()
            .filter(|candidate| candidate.passed)
            .count();
    }
    let estimates: Option<Vec<_>> = by_task
        .values()
        .map(|&(n, c)| pass_at_k(n, c, k.min(n)))
        .collect();
    let estimates = estimates?;
    Some(estimates.iter().sum::<f64>() / estimates.len() as f64)
}

fn exchangeable_sampling_signatures(signatures: &[(u32, Option<&str>, bool)]) -> bool {
    signatures
        .first()
        .is_some_and(|first| signatures.iter().all(|signature| signature == first))
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub schema_version: u32,
    pub task_count: usize,
    pub cell_count: usize,
    pub candidate_count: usize,
    pub solved_cell_count: usize,
    pub candidate_pass_rate: f64,
    /// Fraction of task/config/trial cells solved by at least one candidate.
    pub solve_at_n: f64,
    /// Standard estimator only when every reported candidate group is
    /// exchangeable; `null` for heterogeneous best-of ensembles.
    pub pass_at_k: Option<f64>,
    pub pass_at_k_k: Option<usize>,
    pub false_verified_count: usize,
    /// Fraction of candidates that were false-verified (visible checks green,
    /// final oracle red). Primary vloop-dense quality signal.
    pub false_verified_rate: f64,
    pub infrastructure_error_rate: f64,
    pub solve_rate: f64,
    pub cost_per_solved: Option<f64>,
    /// Mean total tokens across solved cells only (None if none solved).
    pub tokens_per_solved: Option<f64>,
    /// Candidate failure buckets (no-edits / compile / logic / error).
    pub failure_buckets: FailureBucketCounts,
    pub groups: Vec<GroupSummary>,
}

/// Where candidates lose — counts across all candidates in the run.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureBucketCounts {
    pub no_edits: usize,
    pub compile: usize,
    pub logic: usize,
    pub error: usize,
    /// Failed candidates with no classified bucket.
    pub unknown: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GroupSummary {
    pub config: String,
    pub model: String,
    pub cell_count: usize,
    pub candidate_count: usize,
    pub candidate_pass_rate: f64,
    pub solve_at_n: f64,
    pub pass_at_k: Option<f64>,
    pub pass_at_k_k: Option<usize>,
    /// Cells whose verify loop entered repair (>= 2 verify rounds).
    pub repair_entered_count: usize,
    /// Of those, cells that still ended in an oracle pass.
    pub repair_recovered_count: usize,
    /// Total verify rounds that re-failed with an unchanged failure signature.
    pub repeated_verify_failures: u32,
}

pub fn evaluation_summary(
    results: &[RunResult],
    task_count: usize,
    trials: usize,
) -> EvaluationSummary {
    let candidate_count = results
        .iter()
        .map(|row| row.candidates.len())
        .sum::<usize>();
    let candidate_passes = results
        .iter()
        .flat_map(|row| &row.candidates)
        .filter(|candidate| candidate.passed)
        .count();
    let solved_cell_count = results.iter().filter(|row| row.passed).count();
    let false_verified_count = results.iter().map(RunResult::false_verified_count).sum();
    let infrastructure_errors = results
        .iter()
        .map(RunResult::infrastructure_error_count)
        .sum();
    let known_total_cost = results
        .iter()
        .map(RunResult::known_cost)
        .try_fold(0.0, |sum, cost| cost.map(|cost| sum + cost));

    let mut grouped: BTreeMap<(&str, &str), Vec<&RunResult>> = BTreeMap::new();
    for result in results {
        grouped
            .entry((&result.config, &result.model))
            .or_default()
            .push(result);
    }
    let k = trials.max(1);
    let groups: Vec<GroupSummary> = grouped
        .into_iter()
        .map(|((config, model), rows)| {
            let candidates = rows.iter().map(|row| row.candidates.len()).sum::<usize>();
            let passes = rows
                .iter()
                .flat_map(|row| &row.candidates)
                .filter(|candidate| candidate.passed)
                .count();
            let pass_k = group_pass_at_k(&rows, k);
            let repair_rows: Vec<&&RunResult> = rows
                .iter()
                .filter(|row| row.trajectory.verify_rounds >= 2)
                .collect();
            GroupSummary {
                config: config.to_string(),
                model: model.to_string(),
                cell_count: rows.len(),
                candidate_count: candidates,
                candidate_pass_rate: ratio(passes, candidates),
                solve_at_n: ratio(rows.iter().filter(|row| row.passed).count(), rows.len()),
                pass_at_k: pass_k,
                pass_at_k_k: pass_k.map(|_| k),
                repair_entered_count: repair_rows.len(),
                repair_recovered_count: repair_rows.iter().filter(|row| row.passed).count(),
                repeated_verify_failures: rows
                    .iter()
                    .map(|row| row.trajectory.repeated_verify_failures)
                    .sum(),
            }
        })
        .collect();
    let global_pass_at_k = if groups.is_empty() {
        None
    } else {
        groups
            .iter()
            .map(|group| group.pass_at_k)
            .collect::<Option<Vec<_>>>()
            .map(|values| values.iter().sum::<f64>() / values.len() as f64)
    };
    let solve_rate = ratio(solved_cell_count, results.len());
    let failure_buckets = failure_bucket_counts(results);
    let tokens_per_solved = {
        let solved: Vec<&RunResult> = results.iter().filter(|row| row.passed).collect();
        if solved.is_empty() {
            None
        } else {
            let total: u64 = solved.iter().map(|row| row.tokens).sum();
            Some(total as f64 / solved.len() as f64)
        }
    };
    EvaluationSummary {
        schema_version: 2,
        task_count,
        cell_count: results.len(),
        candidate_count,
        solved_cell_count,
        candidate_pass_rate: ratio(candidate_passes, candidate_count),
        solve_at_n: solve_rate,
        pass_at_k: global_pass_at_k,
        pass_at_k_k: global_pass_at_k.map(|_| k),
        false_verified_count,
        false_verified_rate: ratio(false_verified_count, candidate_count),
        infrastructure_error_rate: ratio(infrastructure_errors, candidate_count),
        solve_rate,
        cost_per_solved: known_total_cost
            .filter(|_| solved_cell_count > 0)
            .map(|cost| cost / solved_cell_count as f64),
        tokens_per_solved,
        failure_buckets,
        groups,
    }
}

fn failure_bucket_counts(results: &[RunResult]) -> FailureBucketCounts {
    let mut counts = FailureBucketCounts::default();
    for candidate in results.iter().flat_map(|row| &row.candidates) {
        if candidate.passed {
            continue;
        }
        match candidate.fail {
            Some(FailKind::NoEdits) => counts.no_edits += 1,
            Some(FailKind::Compile) => counts.compile += 1,
            Some(FailKind::Logic) => counts.logic += 1,
            Some(FailKind::Error) => counts.error += 1,
            None => counts.unknown += 1,
        }
    }
    counts
}

pub fn write_summary(
    artifacts_dir: &Path,
    results: &[RunResult],
    task_count: usize,
    trials: usize,
) -> Result<()> {
    let summary = evaluation_summary(results, task_count, trials);
    std::fs::write(
        artifacts_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(())
}

fn short_model(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

#[cfg(test)]
mod tests {
    use super::{evaluation_summary, exchangeable_sampling_signatures, pass_at_k};

    #[test]
    fn pass_at_k_matches_fixed_synthetic_fixture() {
        assert_eq!(pass_at_k(10, 0, 3), Some(0.0));
        assert_eq!(pass_at_k(10, 10, 3), Some(1.0));
        assert!((pass_at_k(10, 2, 1).unwrap() - 0.2).abs() < 1e-12);
        let expected = 1.0 - (8.0 / 10.0) * (7.0 / 9.0);
        assert!((pass_at_k(10, 2, 2).unwrap() - expected).abs() < 1e-12);
        assert_eq!(pass_at_k(2, 1, 3), None);
        assert_eq!(pass_at_k(0, 0, 1), None);
    }

    #[test]
    fn heterogeneous_ensemble_is_not_exchangeable_pass_at_k() {
        assert!(exchangeable_sampling_signatures(&[
            (0.2f32.to_bits(), Some("provider/model"), true),
            (0.2f32.to_bits(), Some("provider/model"), true),
        ]));
        assert!(!exchangeable_sampling_signatures(&[
            (0.2f32.to_bits(), Some("provider/model"), true),
            (0.7f32.to_bits(), Some("provider/model"), true),
        ]));
        assert!(!exchangeable_sampling_signatures(&[
            (0.2f32.to_bits(), Some("primary/model"), true),
            (0.2f32.to_bits(), Some("fallback/model"), true),
        ]));
        assert!(!exchangeable_sampling_signatures(&[
            (0.2f32.to_bits(), Some("provider/model"), true),
            (0.2f32.to_bits(), Some("provider/model"), false),
        ]));
    }

    #[test]
    fn machine_summary_has_stable_v2_null_metrics_without_samples() {
        let summary = evaluation_summary(&[], 0, 3);
        let value = serde_json::to_value(&summary).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(value["candidate_pass_rate"], 0.0);
        assert_eq!(value["solve_at_n"], 0.0);
        assert!(value["pass_at_k"].is_null());
        assert!(value["cost_per_solved"].is_null());
        assert_eq!(value["false_verified_rate"], 0.0);
        assert!(value["tokens_per_solved"].is_null());
        assert_eq!(value["failure_buckets"]["no_edits"], 0);
        // Round-trip through deserialize (baseline capture path).
        let back: super::EvaluationSummary = serde_json::from_value(value).unwrap();
        assert_eq!(back.solve_rate, 0.0);
    }
}
