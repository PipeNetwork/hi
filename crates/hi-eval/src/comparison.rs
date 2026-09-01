//! Cross-run comparison of denominator-aware evaluation artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct ComparisonReport {
    pub schema_version: u32,
    pub left: String,
    pub right: String,
    pub left_only_tasks: Vec<String>,
    pub right_only_tasks: Vec<String>,
    pub common_tasks: usize,
    pub common_valid_tasks: usize,
    pub left_metrics: ComparisonMetrics,
    pub right_metrics: ComparisonMetrics,
    pub common_valid_metrics: CommonValidMetrics,
}

#[derive(Debug, Default, Serialize)]
pub struct ComparisonMetrics {
    pub task_count: usize,
    pub solved_tasks: usize,
    pub solve_rate: f64,
    pub candidate_count: usize,
    pub attempted_count: usize,
    pub provider_accepted_count: usize,
    pub model_valid_count: usize,
    pub tool_started_count: usize,
    pub solved_count: usize,
    pub provider_accepted_rate: f64,
    pub model_valid_rate: f64,
    pub tool_started_rate: f64,
    pub refusal_rate: f64,
    pub provider_error_rate: f64,
    pub provider_error_count: usize,
    pub policy_blocked_count: usize,
    pub refusal_count: usize,
    pub total_tokens: u64,
    pub known_cost: Option<f64>,
    pub cost_complete: bool,
    pub cost_per_solved: Option<f64>,
}

#[derive(Debug, Default, Serialize)]
pub struct CommonValidMetrics {
    pub task_count: usize,
    pub left_solved_tasks: usize,
    pub right_solved_tasks: usize,
    pub left_solve_rate: f64,
    pub right_solve_rate: f64,
}

#[derive(Default)]
struct TaskRow {
    passed: bool,
    valid: bool,
    candidates: usize,
    provider_errors: usize,
    policy_blocks: usize,
    refusals: usize,
    tools_started: usize,
    attempted: usize,
    provider_accepted: usize,
    model_valid: usize,
    solved: usize,
    total_tokens: u64,
    cost: Option<f64>,
}

pub fn compare_dirs(left: &Path, right: &Path) -> Result<ComparisonReport> {
    let left_rows = load_rows(left)?;
    let right_rows = load_rows(right)?;
    let left_keys: BTreeSet<String> = left_rows.keys().cloned().collect();
    let right_keys: BTreeSet<String> = right_rows.keys().cloned().collect();
    let left_only_tasks = left_keys.difference(&right_keys).cloned().collect();
    let right_only_tasks = right_keys.difference(&left_keys).cloned().collect();
    let common: Vec<String> = left_keys.intersection(&right_keys).cloned().collect();
    let common_valid: Vec<&String> = common
        .iter()
        .filter(|task| left_rows[*task].valid && right_rows[*task].valid)
        .collect();

    let left_metrics = metrics(&left_rows);
    let right_metrics = metrics(&right_rows);
    let left_solved = common_valid
        .iter()
        .filter(|task| left_rows[**task].passed)
        .count();
    let right_solved = common_valid
        .iter()
        .filter(|task| right_rows[**task].passed)
        .count();
    let common_count = common_valid.len();
    Ok(ComparisonReport {
        schema_version: 1,
        left: left.display().to_string(),
        right: right.display().to_string(),
        left_only_tasks,
        right_only_tasks,
        common_tasks: common.len(),
        common_valid_tasks: common_count,
        left_metrics,
        right_metrics,
        common_valid_metrics: CommonValidMetrics {
            task_count: common_count,
            left_solved_tasks: left_solved,
            right_solved_tasks: right_solved,
            left_solve_rate: ratio(left_solved, common_count),
            right_solve_rate: ratio(right_solved, common_count),
        },
    })
}

fn load_rows(root: &Path) -> Result<BTreeMap<String, TaskRow>> {
    let mut rows = BTreeMap::new();
    let mut files = Vec::new();
    collect_json_files(root, &mut files)?;
    for path in files {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading comparison artifact {}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing comparison artifact {}", path.display()))?;
        let Some(task) = value.get("task").and_then(Value::as_str) else {
            continue;
        };
        if value.get("candidate_results").is_none() {
            continue;
        }
        let candidates = value["candidate_results"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut row = TaskRow {
            passed: value["passed"].as_bool().unwrap_or(false),
            valid: false,
            candidates: candidates.len(),
            ..TaskRow::default()
        };
        let mut any_cost_unknown = false;
        let mut total_cost = 0.0;
        for candidate in candidates {
            row.attempted += 1;
            let mode = candidate["failure_mode"].as_str().unwrap_or("");
            let accepted = candidate["model_outcome"]["accepted_completions"]
                .as_u64()
                .unwrap_or(0)
                > 0;
            let model_valid = accepted || mode == "tool_protocol_error";
            row.valid |= model_valid;
            row.provider_accepted += usize::from(accepted);
            row.model_valid += usize::from(model_valid);
            row.solved += usize::from(candidate["passed"].as_bool().unwrap_or(false));
            row.provider_errors += usize::from(candidate["provider_error_kind"].is_string());
            row.policy_blocks += usize::from(mode == "api_policy_blocked");
            row.refusals += usize::from(
                mode == "model_refusal_before_tools" || mode == "model_refusal_after_tools",
            );
            row.tools_started += usize::from(
                candidate["model_outcome"]["tool_calls_before_stop"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0,
            );
            row.total_tokens = row
                .total_tokens
                .saturating_add(candidate["tokens"].as_u64().unwrap_or(0));
            if let Some(cost) = candidate["cost"].as_f64() {
                total_cost += cost;
            } else {
                any_cost_unknown = true;
            }
        }
        row.cost = (!any_cost_unknown).then_some(total_cost);
        rows.entry(task.to_string())
            .and_modify(|existing| merge_task_row(existing, &row))
            .or_insert(row);
    }
    if rows.is_empty() {
        bail!("no evaluation artifacts found under {}", root.display());
    }
    Ok(rows)
}

fn merge_task_row(existing: &mut TaskRow, incoming: &TaskRow) {
    existing.passed |= incoming.passed;
    existing.valid |= incoming.valid;
    existing.candidates += incoming.candidates;
    existing.provider_errors += incoming.provider_errors;
    existing.policy_blocks += incoming.policy_blocks;
    existing.refusals += incoming.refusals;
    existing.tools_started += incoming.tools_started;
    existing.attempted += incoming.attempted;
    existing.provider_accepted += incoming.provider_accepted;
    existing.model_valid += incoming.model_valid;
    existing.solved += incoming.solved;
    existing.total_tokens = existing.total_tokens.saturating_add(incoming.total_tokens);
    existing.cost = match (existing.cost, incoming.cost) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    };
}

fn collect_json_files(root: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("reading comparison artifact directory {}", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("json")
            && path.file_name().and_then(|name| name.to_str()) != Some("summary.json")
        {
            files.push(path);
        }
    }
    Ok(())
}

fn metrics(rows: &BTreeMap<String, TaskRow>) -> ComparisonMetrics {
    let task_count = rows.len();
    let solved_tasks = rows.values().filter(|row| row.passed).count();
    let mut cost_complete = true;
    let mut known_cost = 0.0;
    let mut candidate_count = 0;
    let mut provider_error_count = 0;
    let mut policy_blocked_count = 0;
    let mut refusal_count = 0;
    let mut tool_started_count = 0;
    let mut attempted_count = 0;
    let mut provider_accepted_count = 0;
    let mut model_valid_count = 0;
    let mut solved_count = 0;
    let mut total_tokens = 0u64;
    for row in rows.values() {
        candidate_count += row.candidates;
        provider_error_count += row.provider_errors;
        policy_blocked_count += row.policy_blocks;
        refusal_count += row.refusals;
        tool_started_count += row.tools_started;
        attempted_count += row.attempted;
        provider_accepted_count += row.provider_accepted;
        model_valid_count += row.model_valid;
        solved_count += row.solved;
        total_tokens = total_tokens.saturating_add(row.total_tokens);
        if let Some(cost) = row.cost {
            known_cost += cost;
        } else {
            cost_complete = false;
        }
    }
    ComparisonMetrics {
        task_count,
        solved_tasks,
        solve_rate: ratio(solved_tasks, task_count),
        candidate_count,
        attempted_count,
        provider_accepted_count,
        model_valid_count,
        tool_started_count,
        solved_count,
        provider_accepted_rate: ratio(provider_accepted_count, attempted_count),
        model_valid_rate: ratio(model_valid_count, attempted_count),
        tool_started_rate: ratio(tool_started_count, attempted_count),
        refusal_rate: ratio(refusal_count, attempted_count),
        provider_error_rate: ratio(provider_error_count, attempted_count),
        provider_error_count,
        policy_blocked_count,
        refusal_count,
        total_tokens,
        known_cost: cost_complete.then_some(known_cost),
        cost_complete,
        cost_per_solved: cost_complete
            .then_some(known_cost)
            .filter(|_| solved_count > 0)
            .map(|cost| cost / solved_count as f64),
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{TaskRow, merge_task_row, metrics};

    #[test]
    fn duplicate_trials_use_task_solve_semantics_and_preserve_unknown_cost() {
        let mut first = TaskRow {
            passed: false,
            valid: true,
            candidates: 1,
            attempted: 1,
            provider_accepted: 1,
            model_valid: 1,
            solved: 0,
            total_tokens: 10,
            cost: Some(1.5),
            ..TaskRow::default()
        };
        let second = TaskRow {
            passed: true,
            valid: false,
            candidates: 1,
            attempted: 1,
            model_valid: 0,
            solved: 1,
            total_tokens: 20,
            cost: None,
            ..TaskRow::default()
        };
        merge_task_row(&mut first, &second);
        assert!(first.passed);
        assert!(first.valid);
        assert_eq!(first.candidates, 2);
        assert_eq!(first.total_tokens, 30);
        assert_eq!(first.cost, None);
    }

    #[test]
    fn comparison_rates_keep_attempt_counts_as_denominators() {
        let rows = [
            (
                "solved",
                TaskRow {
                    passed: true,
                    valid: true,
                    candidates: 1,
                    attempted: 1,
                    provider_accepted: 1,
                    model_valid: 1,
                    solved: 1,
                    tools_started: 1,
                    total_tokens: 42,
                    cost: Some(2.0),
                    ..TaskRow::default()
                },
            ),
            (
                "blocked",
                TaskRow {
                    candidates: 1,
                    attempted: 1,
                    ..TaskRow::default()
                },
            ),
        ]
        .into_iter()
        .map(|(task, row)| (task.to_string(), row))
        .collect();
        let report = metrics(&rows);
        assert_eq!(report.task_count, 2);
        assert_eq!(report.attempted_count, 2);
        assert_eq!(report.provider_accepted_count, 1);
        assert_eq!(report.model_valid_count, 1);
        assert_eq!(report.solved_tasks, 1);
        assert!((report.provider_accepted_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(report.total_tokens, 42);
        assert!(!report.cost_complete);
        assert_eq!(report.known_cost, None);
    }
}
