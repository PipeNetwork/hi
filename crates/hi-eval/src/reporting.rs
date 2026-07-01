use crate::config::Config;
use crate::results::{FailKind, RunResult};

pub fn print_summary(results: &[RunResult], task_count: usize, active: &[&Config], trials: usize) {
    println!("\n=== Results ({task_count} tasks × {trials} trial(s)) ===");
    println!(
        "{:<32} {:>14} {:>8} {:>6} {:>11} {:>10}",
        "config", "pass@1", "pass@k", "cand", "tok/trial", "tok/task"
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

            // Tasks passed per trial → mean ± spread (pass@1, with error bars).
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
            let pass1 = if trials == 1 {
                format!("{}/{task_count}", per_trial[0])
            } else {
                format!("{mean:.1}±{std:.1}/{task_count}")
            };

            // pass@k: tasks solved by at least one trial (the capability ceiling).
            let mut solved: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for r in &rows {
                if r.passed {
                    solved.insert(r.task.as_str());
                }
            }

            let tokens: u64 = rows.iter().map(|r| r.tokens).sum::<u64>() / trials as u64;
            // The headline A/B number: mean tokens to attempt one task (summed across
            // a config's candidates). Compare it across condense on/off runs.
            let tok_per_task = tokens / task_count.max(1) as u64;
            println!(
                "{:<32} {:>14} {:>8} {:>6} {:>11} {:>10}",
                label,
                pass1,
                format!("{}/{task_count}", solved.len()),
                config.temperatures.len(),
                tokens,
                tok_per_task,
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
        }
    }
}

fn short_model(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}
