//! Identity-safe manifest evaluation reports.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hi_eval::EvalStateStore;
use serde::Serialize;
use serde_json::Value;

use crate::review_target::resolve_runtime_roots;

#[derive(Debug, Serialize)]
struct IncomparableRecord {
    observed_identity_digest: String,
    reason: &'static str,
    record: Value,
}

#[derive(Debug, Serialize)]
struct ReportComparability {
    status: &'static str,
    expected_identity_digest: Option<String>,
    comparable_records: usize,
    incomparable_records: usize,
    unscoped_records: usize,
}

struct PartitionedRecords {
    comparable: Vec<Value>,
    incomparable: Vec<IncomparableRecord>,
    unscoped: Vec<Value>,
}

pub(crate) fn report(args: &[String]) -> Result<()> {
    let profile = required_profile(args)?;
    let (_, default_state_root) = resolve_runtime_roots()?;
    let state = EvalStateStore::new(
        flag_path(args, "--state").unwrap_or_else(|| default_state_root.join("evals")),
    );
    let root = state.profile_root(&profile)?;
    let preparation = state.read_preparation(&profile).ok();
    let run = state.read_run(&profile)?;
    let expected = run
        .as_ref()
        .map(|record| record.identity.digest.as_str())
        .or_else(|| {
            preparation
                .as_ref()
                .map(|receipt| receipt.identity.digest.as_str())
        });
    let mut records = Vec::new();
    collect_json(&root.join("attempts"), &mut records)?;
    collect_json(&root.join("evidence"), &mut records)?;
    collect_json(&root.join("comparisons"), &mut records)?;
    let partitioned = partition_records(expected, records);
    let comparability = ReportComparability {
        status: if expected.is_none() {
            "identity_unavailable"
        } else if partitioned.incomparable.is_empty() {
            "comparable"
        } else {
            "contains_incomparable_evidence"
        },
        expected_identity_digest: expected.map(str::to_owned),
        comparable_records: partitioned.comparable.len(),
        incomparable_records: partitioned.incomparable.len(),
        unscoped_records: partitioned.unscoped.len(),
    };
    let report = serde_json::json!({
        "schema_version": hi_eval::PLATFORM_SCHEMA_VERSION,
        "profile": profile,
        "preparation": preparation,
        "run": run,
        "comparability": comparability,
        // Only exact-identity records are eligible for current-run scoring.
        "records": partitioned.comparable,
        // Retain recovery/debug evidence, but never label it a regression.
        "incomparable_records": partitioned.incomparable,
        "unscoped_records": partitioned.unscoped,
    });
    let path = state.write_report(&profile, &report)?;
    println!(
        "{}\nreport: {}",
        serde_json::to_string_pretty(&report)?,
        path.display()
    );
    Ok(())
}

fn partition_records(expected: Option<&str>, records: Vec<Value>) -> PartitionedRecords {
    let mut partitioned = PartitionedRecords {
        comparable: Vec::new(),
        incomparable: Vec::new(),
        unscoped: Vec::new(),
    };
    for record in records {
        let observed = record
            .get("identity_digest")
            .and_then(Value::as_str)
            .map(str::to_owned);
        match (expected, observed) {
            (Some(expected), Some(observed)) if observed == expected => {
                partitioned.comparable.push(record);
            }
            (Some(_), Some(observed)) => {
                partitioned.incomparable.push(IncomparableRecord {
                    observed_identity_digest: observed,
                    reason: "run identity differs; result is incomparable, not a regression",
                    record,
                });
            }
            (None, Some(observed)) => {
                partitioned.incomparable.push(IncomparableRecord {
                    observed_identity_digest: observed,
                    reason: "current run identity is unavailable",
                    record,
                });
            }
            (_, None) => partitioned.unscoped.push(record),
        }
    }
    partitioned
}

fn collect_json(root: &Path, output: &mut Vec<Value>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_json(&path, output)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
            && let Ok(value) = serde_json::from_slice(&fs::read(path)?)
        {
            output.push(value);
        }
    }
    Ok(())
}

fn required_profile(args: &[String]) -> Result<String> {
    flag_string(args, "--profile")
        .or_else(|| args.get(1).filter(|value| !value.starts_with('-')).cloned())
        .context("command requires --profile <name>")
}

fn flag_path(args: &[String], name: &str) -> Option<PathBuf> {
    flag_string(args, name).map(PathBuf::from)
}

fn flag_string(args: &[String], name: &str) -> Option<String> {
    let prefix = format!("{name}=");
    args.iter()
        .find_map(|arg| arg.strip_prefix(&prefix).map(str::to_string))
        .or_else(|| {
            args.iter()
                .position(|arg| arg == name)
                .and_then(|index| args.get(index + 1).cloned())
        })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn changed_identity_is_incomparable_and_never_current() {
        let records = vec![
            json!({"identity_digest": "current", "score": 1}),
            json!({"identity_digest": "old", "score": 0}),
            json!({"artifact": "diagnostic"}),
        ];
        let result = partition_records(Some("current"), records);
        assert_eq!(result.comparable.len(), 1);
        assert_eq!(result.incomparable.len(), 1);
        assert_eq!(result.unscoped.len(), 1);
        assert_eq!(
            result.incomparable[0].reason,
            "run identity differs; result is incomparable, not a regression"
        );
    }

    #[test]
    fn missing_current_identity_never_makes_records_comparable() {
        let result = partition_records(None, vec![serde_json::json!({"identity_digest": "old"})]);
        assert!(result.comparable.is_empty());
        assert_eq!(result.incomparable.len(), 1);
    }
}
