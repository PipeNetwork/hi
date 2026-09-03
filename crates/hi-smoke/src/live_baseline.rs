use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::live_route::LiveRoute;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    schema_version: u16,
    status: BaselineStatus,
    successful_nightly_runs: u64,
    capture_after_successful_runs: u64,
    captured_at: Option<String>,
    model_route: LiveRoute,
    scenario_count: u64,
    scenario_pass_rate: Option<f64>,
    crash_count: Option<u64>,
    infrastructure_loop_count: Option<u64>,
    gates: Gates,
    note: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum BaselineStatus {
    Observing,
    Captured,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gates {
    maximum_scenario_pass_regression_points: f64,
    maximum_crashes: u64,
    maximum_infrastructure_loops: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteSummary {
    schema_version: u16,
    mode: String,
    live_route: Option<LiveRoute>,
    total: u64,
    passed: u64,
    failed: u64,
    scenario_pass_rate: f64,
    crash_count: u64,
    infrastructure_loop_count: u64,
    #[serde(default)]
    infrastructure_failure_count: u64,
    provider_request_count: u64,
    provider_chat_request_count: u64,
    provider_accepted_request_count: u64,
    provider_response_status_counts: BTreeMap<u16, u64>,
    cases: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ObservationState {
    schema_version: u16,
    model_route: LiveRoute,
    scenario_count: u64,
    capture_after_successful_runs: u64,
    successful_nightly_run_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Evaluation<'a> {
    schema_version: u16,
    baseline_status: BaselineStatus,
    result: &'a str,
    scenario_count: u64,
    scenario_pass_rate: f64,
    crash_count: u64,
    infrastructure_loop_count: u64,
    infrastructure_failure_count: u64,
    provider_request_count: u64,
    provider_chat_request_count: u64,
    provider_accepted_request_count: u64,
    provider_response_status_counts: &'a BTreeMap<u16, u64>,
    observed_successful_runs: u64,
    capture_after_successful_runs: u64,
}

pub(crate) fn check(
    summary_path: &Path,
    baseline_path: &Path,
    observation_state_path: Option<&Path>,
    nightly_run_id: Option<&str>,
) -> Result<()> {
    let summary: SuiteSummary = read_json(summary_path)?;
    let baseline: Baseline = read_json(baseline_path)?;
    ensure!(
        observation_state_path.is_some() == nightly_run_id.is_some(),
        "--observation-state and --nightly-run-id must be provided together"
    );
    ensure!(
        summary.schema_version == 1,
        "unsupported suite summary schema"
    );
    ensure!(
        baseline.schema_version == 1,
        "unsupported live baseline schema"
    );
    ensure!(summary.mode == "live", "baseline input is not a live run");
    let summary_route = summary
        .live_route
        .as_ref()
        .context("live summary did not record its provider route")?;
    let summary_route = LiveRoute::new(
        &summary_route.provider,
        &summary_route.model,
        &summary_route.base_url,
    )?;
    let baseline_route = LiveRoute::new(
        &baseline.model_route.provider,
        &baseline.model_route.model,
        &baseline.model_route.base_url,
    )?;
    ensure!(
        summary_route == baseline_route,
        "live provider route changed from {:?} to {:?}; review the baseline explicitly",
        baseline_route,
        summary_route
    );
    ensure!(summary.total > 0, "live summary contains no scenarios");
    ensure!(
        summary.passed + summary.failed == summary.total,
        "live summary totals are inconsistent"
    );
    ensure!(
        summary.cases.len() as u64 == summary.total,
        "live summary case count is inconsistent"
    );
    ensure!(
        summary.total == baseline.scenario_count,
        "live scenario count changed from {} to {}; review the baseline explicitly",
        baseline.scenario_count,
        summary.total
    );
    ensure!(
        baseline.capture_after_successful_runs > 0,
        "capture_after_successful_runs must be positive"
    );
    ensure!(
        baseline.gates.maximum_crashes == 0,
        "live crash gate is a hard invariant and must remain zero"
    );
    ensure!(
        baseline.gates.maximum_infrastructure_loops == 0,
        "live infrastructure-loop gate is a hard invariant and must remain zero"
    );
    let _ = &baseline.note;
    let pass_rate = summary.passed as f64 * 100.0 / summary.total as f64;
    ensure!(
        (summary.scenario_pass_rate - pass_rate).abs() < 0.000_001,
        "live summary pass rate is inconsistent"
    );
    ensure!(
        summary.crash_count <= summary.failed,
        "live crash count exceeds the failed scenario count"
    );
    ensure!(
        summary.infrastructure_loop_count <= summary.failed,
        "live infrastructure-loop count exceeds the failed scenario count"
    );
    ensure!(
        summary.infrastructure_failure_count <= summary.failed,
        "live infrastructure-failure count exceeds the failed scenario count"
    );
    ensure!(
        summary.provider_chat_request_count <= summary.provider_request_count,
        "live provider chat-request count exceeds total request count"
    );
    ensure!(
        summary.provider_accepted_request_count <= summary.provider_request_count,
        "live provider accepted-request count exceeds total request count"
    );
    ensure!(
        summary
            .provider_response_status_counts
            .values()
            .sum::<u64>()
            == summary.provider_request_count,
        "live provider HTTP-status counts do not match total request count"
    );
    ensure!(
        summary
            .provider_response_status_counts
            .iter()
            .filter(|(status, _)| **status >= 200 && **status < 300)
            .map(|(_, count)| count)
            .sum::<u64>()
            == summary.provider_accepted_request_count,
        "live provider accepted-request count does not match successful HTTP statuses"
    );
    let mut case_request_count = 0_u64;
    let mut case_chat_request_count = 0_u64;
    let mut case_accepted_request_count = 0_u64;
    let mut case_passed = 0_u64;
    let mut case_failed = 0_u64;
    for (index, case) in summary.cases.iter().enumerate() {
        let requests = case_metric(case, "provider_request_count", index)?;
        let chats = case_metric(case, "provider_chat_request_count", index)?;
        let accepted = case_metric(case, "provider_accepted_request_count", index)?;
        ensure!(
            chats <= requests && accepted <= requests,
            "live summary case {index} has inconsistent provider request counts"
        );
        match case.get("status").and_then(serde_json::Value::as_str) {
            Some("passed") => {
                case_passed += 1;
                ensure!(
                    accepted > 0,
                    "live summary case {index} passed without an accepted HTTP request"
                );
            }
            Some("failed") => case_failed += 1,
            Some(status) => bail!("live summary case {index} has unsupported status {status:?}"),
            None => bail!("live summary case {index} omitted status"),
        }
        case_request_count = case_request_count.saturating_add(requests);
        case_chat_request_count = case_chat_request_count.saturating_add(chats);
        case_accepted_request_count = case_accepted_request_count.saturating_add(accepted);
    }
    ensure!(
        case_request_count == summary.provider_request_count
            && case_chat_request_count == summary.provider_chat_request_count
            && case_accepted_request_count == summary.provider_accepted_request_count,
        "live provider request aggregates do not match their case summaries"
    );
    ensure!(
        case_passed == summary.passed && case_failed == summary.failed,
        "live summary pass/fail aggregates do not match their case summaries"
    );

    // These are infrastructure invariants, not quality metrics. They gate even
    // during the observation window and cannot be relaxed by a captured rate.
    ensure!(
        summary.crash_count == 0,
        "live crash gate failed: expected zero, found {}",
        summary.crash_count
    );
    ensure!(
        summary.infrastructure_loop_count == 0,
        "live infrastructure-loop gate failed: expected zero, found {}",
        summary.infrastructure_loop_count
    );
    ensure!(
        summary.infrastructure_failure_count == 0,
        "live infrastructure-failure gate failed: expected zero, found {}",
        summary.infrastructure_failure_count
    );

    let mut observed_successful_runs = baseline.successful_nightly_runs;
    let mut capture_required = false;
    let result = match baseline.status {
        BaselineStatus::Observing => {
            ensure!(
                baseline.scenario_pass_rate.is_none()
                    && baseline.crash_count.is_none()
                    && baseline.infrastructure_loop_count.is_none()
                    && baseline.captured_at.is_none(),
                "observing baseline must not contain captured metrics"
            );
            if let (Some(state_path), Some(run_id)) = (observation_state_path, nightly_run_id) {
                let run_id = validate_run_id(run_id)?;
                let mut state = load_observation_state(state_path, &baseline)?;
                observed_successful_runs =
                    observed_successful_runs.max(state.successful_nightly_run_ids.len() as u64);

                if observed_successful_runs >= baseline.capture_after_successful_runs {
                    capture_required = true;
                } else if summary.failed == 0
                    && !state
                        .successful_nightly_run_ids
                        .iter()
                        .any(|known| known == run_id)
                {
                    state.successful_nightly_run_ids.push(run_id.to_owned());
                    write_observation_state(state_path, &state)?;
                    observed_successful_runs =
                        observed_successful_runs.max(state.successful_nightly_run_ids.len() as u64);
                    capture_required =
                        observed_successful_runs >= baseline.capture_after_successful_runs;
                }
            } else {
                capture_required =
                    observed_successful_runs >= baseline.capture_after_successful_runs;
            }

            if capture_required {
                "capture_required"
            } else if summary.failed == 0 {
                "observing"
            } else {
                "observation_not_counted"
            }
        }
        BaselineStatus::Captured => {
            ensure!(
                baseline.successful_nightly_runs >= baseline.capture_after_successful_runs,
                "captured baseline has only {} successful nightly runs; at least {} reviewed runs are required",
                baseline.successful_nightly_runs,
                baseline.capture_after_successful_runs
            );
            ensure!(
                baseline
                    .captured_at
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "captured baseline needs captured_at"
            );
            let baseline_rate = baseline
                .scenario_pass_rate
                .context("captured baseline needs scenario_pass_rate")?;
            let baseline_crashes = baseline
                .crash_count
                .context("captured baseline needs crash_count")?;
            let baseline_infrastructure_loops = baseline
                .infrastructure_loop_count
                .context("captured baseline needs infrastructure_loop_count")?;
            ensure!(
                baseline_crashes == 0 && baseline_infrastructure_loops == 0,
                "captured baseline cannot record crashes or infrastructure loops"
            );
            let regression = (baseline_rate - pass_rate).max(0.0);
            ensure!(
                regression <= baseline.gates.maximum_scenario_pass_regression_points,
                "live scenario pass rate regressed by {regression:.2} points (maximum {:.2})",
                baseline.gates.maximum_scenario_pass_regression_points
            );
            "passed"
        }
    };

    let evaluation = Evaluation {
        schema_version: 1,
        baseline_status: baseline.status,
        result,
        scenario_count: summary.total,
        scenario_pass_rate: pass_rate,
        crash_count: summary.crash_count,
        infrastructure_loop_count: summary.infrastructure_loop_count,
        infrastructure_failure_count: summary.infrastructure_failure_count,
        provider_request_count: summary.provider_request_count,
        provider_chat_request_count: summary.provider_chat_request_count,
        provider_accepted_request_count: summary.provider_accepted_request_count,
        provider_response_status_counts: &summary.provider_response_status_counts,
        observed_successful_runs,
        capture_after_successful_runs: baseline.capture_after_successful_runs,
    };
    let report = summary_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("live-baseline-evaluation.json");
    fs::write(&report, serde_json::to_vec_pretty(&evaluation)?)
        .with_context(|| format!("writing live baseline evaluation {}", report.display()))?;
    println!(
        "live baseline {result}: {:.2}% pass, {} crash(es), {} infrastructure loop(s), {} successful observation(s)",
        pass_rate, summary.crash_count, summary.infrastructure_loop_count, observed_successful_runs
    );
    if capture_required {
        bail!(
            "the {}-run observation window is complete; explicitly review and capture eval-baseline/tui-live.json",
            baseline.capture_after_successful_runs
        );
    }
    Ok(())
}

fn case_metric(case: &serde_json::Value, name: &str, index: usize) -> Result<u64> {
    case.get(name)
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("live summary case {index} omitted {name}"))
}

fn validate_run_id(run_id: &str) -> Result<&str> {
    let run_id = run_id.trim();
    ensure!(!run_id.is_empty(), "nightly run ID must not be empty");
    ensure!(run_id.len() <= 256, "nightly run ID is too long");
    ensure!(
        run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')),
        "nightly run ID contains unsupported characters"
    );
    Ok(run_id)
}

fn load_observation_state(path: &Path, baseline: &Baseline) -> Result<ObservationState> {
    let state = if path.exists() {
        read_json(path)?
    } else {
        ObservationState {
            schema_version: 1,
            model_route: baseline.model_route.clone(),
            scenario_count: baseline.scenario_count,
            capture_after_successful_runs: baseline.capture_after_successful_runs,
            successful_nightly_run_ids: Vec::new(),
        }
    };
    ensure!(
        state.schema_version == 1,
        "unsupported live observation-state schema"
    );
    ensure!(
        state.model_route == baseline.model_route
            && state.scenario_count == baseline.scenario_count
            && state.capture_after_successful_runs == baseline.capture_after_successful_runs,
        "live observation state does not match the current baseline configuration; review or clear the stale state explicitly"
    );
    let unique = state
        .successful_nightly_run_ids
        .iter()
        .collect::<HashSet<_>>();
    ensure!(
        unique.len() == state.successful_nightly_run_ids.len(),
        "live observation state contains duplicate nightly run IDs"
    );
    for run_id in &state.successful_nightly_run_ids {
        validate_run_id(run_id)?;
    }
    Ok(state)
}

fn write_observation_state(path: &Path, state: &ObservationState) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| {
            format!("creating observation state directory {}", parent.display())
        })?;
    }
    let temporary = temporary_state_path(path);
    let bytes = serde_json::to_vec_pretty(state)?;
    fs::write(&temporary, bytes)
        .with_context(|| format!("writing observation state {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("committing observation state {}", path.display()))?;
    Ok(())
}

fn temporary_state_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("observation-state.json");
    path.with_file_name(format!(".{name}.tmp-{}", std::process::id()))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_inputs(
        root: &Path,
        status: &str,
        rate: Option<f64>,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let summary = root.join("summary.json");
        fs::write(
            &summary,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "mode": "live",
                "live_route": {
                    "provider": "pipenetwork",
                    "model": "pipe/deepseek-v4-flash-0731",
                    "base_url": "https://api.pipenetwork.ai/v1"
                },
                "total": 3,
                "passed": 3,
                "failed": 0,
                "scenario_pass_rate": 100.0,
                "crash_count": 0,
                "infrastructure_loop_count": 0,
                "infrastructure_failure_count": 0,
                "provider_request_count": 3,
                "provider_chat_request_count": 3,
                "provider_accepted_request_count": 3,
                "provider_response_status_counts": {"200": 3},
                "cases": [
                    {
                        "status": "passed",
                        "provider_request_count": 1,
                        "provider_chat_request_count": 1,
                        "provider_accepted_request_count": 1
                    },
                    {
                        "status": "passed",
                        "provider_request_count": 1,
                        "provider_chat_request_count": 1,
                        "provider_accepted_request_count": 1
                    },
                    {
                        "status": "passed",
                        "provider_request_count": 1,
                        "provider_chat_request_count": 1,
                        "provider_accepted_request_count": 1
                    }
                ],
            }))
            .unwrap(),
        )
        .unwrap();
        let baseline = root.join("baseline.json");
        fs::write(
            &baseline,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "status": status,
                "successful_nightly_runs": if status == "captured" { 7 } else { 0 },
                "capture_after_successful_runs": 7,
                "captured_at": if status == "captured" { Some("2026-09-02") } else { None },
                "model_route": {
                    "provider": "pipenetwork",
                    "model": "pipe/deepseek-v4-flash-0731",
                    "base_url": "https://api.pipenetwork.ai/v1"
                },
                "scenario_count": 3,
                "scenario_pass_rate": rate,
                "crash_count": if status == "captured" { Some(0) } else { None },
                "infrastructure_loop_count": if status == "captured" { Some(0) } else { None },
                "gates": {
                    "maximum_scenario_pass_regression_points": 5,
                    "maximum_crashes": 0,
                    "maximum_infrastructure_loops": 0
                },
                "note": "test"
            }))
            .unwrap(),
        )
        .unwrap();
        (summary, baseline)
    }

    #[test]
    fn observing_and_captured_baselines_are_evaluated_separately() {
        let observing = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(observing.path(), "observing", None);
        check(&summary, &baseline, None, None).unwrap();

        let captured = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(captured.path(), "captured", Some(100.0));
        check(&summary, &baseline, None, None).unwrap();
    }

    fn update_json(path: &Path, update: impl FnOnce(&mut serde_json::Value)) {
        let mut value: serde_json::Value = read_json(path).unwrap();
        update(&mut value);
        fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    }

    #[test]
    fn captured_baseline_requires_the_full_review_window() {
        let root = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(root.path(), "captured", Some(100.0));
        update_json(&baseline, |value| {
            value["successful_nightly_runs"] = 6.into()
        });

        let error = check(&summary, &baseline, None, None).unwrap_err();
        assert!(error.to_string().contains("at least 7 reviewed runs"));
    }

    #[test]
    fn current_suite_summary_route_is_required_and_must_match_the_baseline() {
        let root = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(root.path(), "observing", None);

        // `runner::suite_summary` writes this route object. Keep this fixture
        // real-shaped so deny_unknown_fields cannot silently break nightly CI.
        check(&summary, &baseline, None, None).unwrap();

        update_json(&summary, |value| {
            value["live_route"]["model"] = "pipe/different-model".into();
        });
        let error = check(&summary, &baseline, None, None).unwrap_err();
        assert!(
            error.to_string().contains("provider route changed"),
            "{error:#}"
        );

        update_json(&summary, |value| {
            value["live_route"] = serde_json::Value::Null
        });
        let error = check(&summary, &baseline, None, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("did not record its provider route"),
            "{error:#}"
        );
    }

    #[test]
    fn live_summary_must_prove_accepted_http_traffic_for_every_passing_case() {
        let root = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(root.path(), "observing", None);
        update_json(&summary, |value| {
            value["provider_accepted_request_count"] = 2.into();
            value["provider_response_status_counts"]["200"] = 2.into();
            value["provider_response_status_counts"]["503"] = 1.into();
            value["cases"][0]["provider_accepted_request_count"] = 0.into();
        });

        let error = check(&summary, &baseline, None, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("passed without an accepted HTTP request"),
            "{error:#}"
        );
    }

    #[test]
    fn live_summary_http_status_counts_must_cover_every_request() {
        let root = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(root.path(), "observing", None);
        update_json(&summary, |value| {
            value["provider_response_status_counts"]["200"] = 2.into();
        });

        let error = check(&summary, &baseline, None, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("HTTP-status counts do not match"),
            "{error:#}"
        );
    }

    #[test]
    fn hard_infrastructure_invariants_gate_during_observation() {
        for (field, message) in [
            ("crash_count", "crash gate failed"),
            (
                "infrastructure_loop_count",
                "infrastructure-loop gate failed",
            ),
            (
                "infrastructure_failure_count",
                "infrastructure-failure gate failed",
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let (summary, baseline) = write_inputs(root.path(), "observing", None);
            update_json(&summary, |value| {
                value["passed"] = 2.into();
                value["failed"] = 1.into();
                value["scenario_pass_rate"] = (200.0 / 3.0).into();
                value[field] = 1.into();
                value["cases"][0]["status"] = "failed".into();
            });

            let error = check(&summary, &baseline, None, None).unwrap_err();
            assert!(error.to_string().contains(message), "{error:#}");
        }
    }

    #[test]
    fn observation_state_counts_distinct_successful_runs_idempotently() {
        let root = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(root.path(), "observing", None);
        let state = root.path().join("cache/observation.json");

        for run in 1..7 {
            let run_id = format!("nightly-{run}");
            check(&summary, &baseline, Some(&state), Some(&run_id)).unwrap();
            check(&summary, &baseline, Some(&state), Some(&run_id)).unwrap();
        }

        let error = check(&summary, &baseline, Some(&state), Some("nightly-7")).unwrap_err();
        assert!(error.to_string().contains("observation window is complete"));

        let persisted: ObservationState = read_json(&state).unwrap();
        assert_eq!(persisted.successful_nightly_run_ids.len(), 7);

        let error = check(&summary, &baseline, Some(&state), Some("nightly-8")).unwrap_err();
        assert!(error.to_string().contains("observation window is complete"));
        let persisted: ObservationState = read_json(&state).unwrap();
        assert_eq!(persisted.successful_nightly_run_ids.len(), 7);
    }

    #[test]
    fn failed_scenarios_do_not_advance_observation_state() {
        let root = tempfile::tempdir().unwrap();
        let (summary, baseline) = write_inputs(root.path(), "observing", None);
        let state = root.path().join("observation.json");
        update_json(&summary, |value| {
            value["passed"] = 2.into();
            value["failed"] = 1.into();
            value["scenario_pass_rate"] = (200.0 / 3.0).into();
            value["cases"][0]["status"] = "failed".into();
        });

        check(&summary, &baseline, Some(&state), Some("failed-night")).unwrap();
        assert!(!state.exists());

        update_json(&summary, |value| {
            value["passed"] = 3.into();
            value["failed"] = 0.into();
            value["scenario_pass_rate"] = 100.0.into();
            value["cases"][0]["status"] = "passed".into();
        });
        check(&summary, &baseline, Some(&state), Some("successful-night")).unwrap();
        let persisted: ObservationState = read_json(&state).unwrap();
        assert_eq!(persisted.successful_nightly_run_ids, ["successful-night"]);
    }
}
