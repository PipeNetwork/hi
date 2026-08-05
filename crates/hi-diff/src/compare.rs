use std::collections::BTreeMap;

use crate::{
    ApiOutcome, CaseVerdict, Difference, EquivalenceContract, LocalOutcome, TensorSummary,
    ToolCallRecord, Verdict,
};

pub fn compare_tensor(
    location: impl Into<String>,
    reference: &[f32],
    candidate: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Option<Difference> {
    if reference.len() != candidate.len() {
        return Some(Difference {
            location: location.into(),
            message: format!(
                "shape/length mismatch: {} vs {}",
                reference.len(),
                candidate.len()
            ),
            max_error: None,
            rms_error: None,
            first_bad_index: None,
        });
    }

    let mut max_error = 0.0f32;
    let mut sum_squared = 0.0f64;
    let mut first_bad_index = None;
    for (index, (&left, &right)) in reference.iter().zip(candidate).enumerate() {
        if !left.is_finite() || !right.is_finite() {
            first_bad_index.get_or_insert(index);
            continue;
        }
        let error = (left - right).abs();
        max_error = max_error.max(error);
        sum_squared += f64::from(error) * f64::from(error);
        let limit = absolute_tolerance.max(relative_tolerance * left.abs());
        if error > limit {
            first_bad_index.get_or_insert(index);
        }
    }
    let rms = (sum_squared / reference.len().max(1) as f64).sqrt() as f32;
    first_bad_index.map(|index| Difference {
        location: location.into(),
        message: format!("tensor mismatch at index {index}"),
        max_error: Some(max_error),
        rms_error: Some(rms),
        first_bad_index: Some(index),
    })
}

pub fn compare_local(
    case_id: impl Into<String>,
    reference_name: &str,
    reference: &LocalOutcome,
    candidates: &[(String, LocalOutcome)],
    contract: &EquivalenceContract,
) -> CaseVerdict {
    let case_id = case_id.into();
    let mut differences = Vec::new();
    let mut target_errors = BTreeMap::new();
    for (name, candidate) in candidates {
        if reference.generated_tokens != candidate.generated_tokens {
            differences.push(Difference {
                location: format!("{name}.generated_tokens"),
                message: format!("generated tokens differ from {reference_name}"),
                max_error: None,
                rms_error: None,
                first_bad_index: first_difference(
                    &reference.generated_tokens,
                    &candidate.generated_tokens,
                ),
            });
        }
        if reference.next_token != candidate.next_token {
            differences.push(Difference {
                location: format!("{name}.next_token"),
                message: format!("next token differs from {reference_name}"),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            });
        }
        if let (Some(left), Some(right)) = (&reference.logits, &candidate.logits)
            && let Some(difference) = compare_tensor(
                format!("{name}.logits"),
                left,
                right,
                contract.absolute_tolerance,
                contract.relative_tolerance,
            )
        {
            differences.push(difference);
        }
        compare_checkpoints(
            &mut differences,
            name,
            &reference.checkpoints,
            &candidate.checkpoints,
            &reference.checkpoint_values,
            &candidate.checkpoint_values,
            contract,
        );
        if candidate.generated_tokens.is_empty() && candidate.next_token.is_none() {
            target_errors.insert(name.clone(), "target returned no token".into());
        }
    }
    let unavailable = differences
        .iter()
        .any(|difference| difference.message == "required checkpoint unavailable");
    let verdict = if !target_errors.is_empty() {
        Verdict::ExecutionError
    } else if unavailable {
        Verdict::Inconclusive
    } else if differences.is_empty() {
        Verdict::Equivalent
    } else {
        Verdict::Mismatch
    };
    CaseVerdict {
        case_id,
        verdict,
        differences,
        target_errors,
    }
}

fn compare_checkpoints(
    differences: &mut Vec<Difference>,
    target_name: &str,
    reference: &[crate::CheckpointRecord],
    candidate: &[crate::CheckpointRecord],
    reference_values: &BTreeMap<String, Vec<f32>>,
    candidate_values: &BTreeMap<String, Vec<f32>>,
    contract: &EquivalenceContract,
) {
    for required in &contract.required_checkpoints {
        let left = reference.iter().find(|item| &item.name == required);
        let right = candidate.iter().find(|item| &item.name == required);
        match (left, right) {
            (None, _) | (_, None) => differences.push(Difference {
                location: format!("{target_name}.{required}"),
                message: "required checkpoint unavailable".into(),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            }),
            (Some(left), Some(right)) if left.summary.shape != right.summary.shape => {
                differences.push(Difference {
                    location: format!("{target_name}.{required}"),
                    message: format!(
                        "checkpoint shape differs: {:?} vs {:?}",
                        left.summary.shape, right.summary.shape
                    ),
                    max_error: None,
                    rms_error: None,
                    first_bad_index: None,
                });
            }
            (Some(left), Some(right)) => {
                let location = format!("{target_name}.{required}");
                if let (Some(left_values), Some(right_values)) = (
                    reference_values.get(required),
                    candidate_values.get(required),
                ) && let Some(difference) = compare_tensor(
                    location.clone(),
                    left_values,
                    right_values,
                    contract.absolute_tolerance,
                    contract.relative_tolerance,
                ) {
                    differences.push(difference);
                } else {
                    let max_error =
                        (left.summary.max.unwrap_or(0.0) - right.summary.max.unwrap_or(0.0)).abs();
                    let l2_error = (left.summary.l2 - right.summary.l2).abs();
                    if max_error > contract.absolute_tolerance
                        && l2_error > contract.absolute_tolerance
                    {
                        differences.push(Difference {
                            location,
                            message:
                                "checkpoint summary differs; rerun with full probe for exact index"
                                    .into(),
                            max_error: Some(max_error),
                            rms_error: Some(l2_error),
                            first_bad_index: None,
                        });
                    }
                }
            }
        }
    }
}

pub fn compare_response(
    case_id: impl Into<String>,
    outcomes: &[(String, ApiOutcome)],
    contract: &EquivalenceContract,
) -> CaseVerdict {
    let case_id = case_id.into();
    let Some((reference_name, reference)) = outcomes.first() else {
        return CaseVerdict {
            case_id,
            verdict: Verdict::ExecutionError,
            differences: Vec::new(),
            target_errors: BTreeMap::new(),
        };
    };
    let mut differences = Vec::new();
    let mut errors = BTreeMap::new();
    for (name, outcome) in outcomes.iter().skip(1) {
        if let Some(error) = &outcome.error_category {
            errors.insert(name.clone(), error.clone());
            continue;
        }
        if reference.error_category != outcome.error_category {
            differences.push(Difference {
                location: format!("{name}.error"),
                message: format!("error category differs from {reference_name}"),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            });
        }
        if contract.exact_text {
            if reference.text != outcome.text {
                differences.push(Difference {
                    location: format!("{name}.text"),
                    message: "exact text differs".into(),
                    max_error: None,
                    rms_error: None,
                    first_bad_index: None,
                });
            }
        } else if normalize_text(&reference.text, contract.normalize_whitespace)
            != normalize_text(&outcome.text, contract.normalize_whitespace)
        {
            differences.push(Difference {
                location: format!("{name}.text"),
                message: "normalized text differs".into(),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            });
        }
        if reference.json != outcome.json {
            differences.push(Difference {
                location: format!("{name}.json"),
                message: "canonical JSON differs".into(),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            });
        }
        if contract.require_same_tool_calls
            && !same_tool_calls(&reference.tool_calls, &outcome.tool_calls)
        {
            differences.push(Difference {
                location: format!("{name}.tool_calls"),
                message: "tool call sequence differs".into(),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            });
        }
        if contract.require_schema_valid && outcome.schema_valid == Some(false) {
            differences.push(Difference {
                location: format!("{name}.schema"),
                message: "response failed schema validation".into(),
                max_error: None,
                rms_error: None,
                first_bad_index: None,
            });
        }
    }
    let verdict = if !errors.is_empty() {
        Verdict::ExecutionError
    } else if differences.is_empty() {
        Verdict::Equivalent
    } else {
        Verdict::Mismatch
    };
    CaseVerdict {
        case_id,
        verdict,
        differences,
        target_errors: errors,
    }
}

pub fn normalize_text(text: &str, whitespace: bool) -> String {
    if whitespace {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        text.to_string()
    }
}

fn same_tool_calls(left: &[ToolCallRecord], right: &[ToolCallRecord]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.name == b.name && a.arguments == b.arguments)
}

fn first_difference<T: PartialEq>(left: &[T], right: &[T]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
}

#[allow(dead_code)]
fn _summary_has_bad_values(summary: &TensorSummary) -> bool {
    summary.nan_count > 0 || summary.inf_count > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_first_bad_tensor_index() {
        let diff =
            compare_tensor("layer.0", &[1.0, 2.0, 3.0], &[1.0, 2.2, 3.0], 1e-3, 1e-3).unwrap();
        assert_eq!(diff.first_bad_index, Some(1));
    }

    #[test]
    fn normalizes_only_when_requested() {
        assert_eq!(normalize_text(" a\n b ", true), "a b");
        assert_eq!(normalize_text(" a\n b ", false), " a\n b ");
    }

    #[test]
    fn compares_full_checkpoint_values_when_available() {
        let summary = TensorSummary::from_values(vec![3], &[1.0, 2.0, 3.0]);
        let checkpoint = |name: &str| crate::CheckpointRecord {
            name: name.into(),
            step: 0,
            summary: summary.clone(),
            artifact: None,
        };
        let reference = LocalOutcome {
            generated_tokens: vec![1],
            next_token: Some(1),
            logits: None,
            checkpoints: vec![checkpoint("hidden")],
            checkpoint_values: BTreeMap::from([(String::from("hidden"), vec![1.0, 2.0, 3.0])]),
        };
        let candidate = LocalOutcome {
            checkpoint_values: BTreeMap::from([(String::from("hidden"), vec![1.0, 2.2, 3.0])]),
            ..reference.clone()
        };
        let contract = EquivalenceContract {
            required_checkpoints: vec!["hidden".into()],
            ..EquivalenceContract::default()
        };
        let verdict = compare_local(
            "case",
            "cpu",
            &reference,
            &[("cuda".into(), candidate)],
            &contract,
        );
        assert_eq!(verdict.verdict, Verdict::Mismatch);
        assert_eq!(verdict.differences[0].first_bad_index, Some(1));
    }

    #[test]
    fn api_comparison_can_ignore_whitespace_but_not_tool_calls() {
        let outcome = |text: &str, name: &str| {
            (
                name.to_string(),
                ApiOutcome {
                    text: text.into(),
                    json: None,
                    tool_calls: vec![ToolCallRecord {
                        name: "read".into(),
                        arguments: serde_json::json!({"path":"a"}),
                    }],
                    finish_reason: Some("stop".into()),
                    error_category: None,
                    input_tokens: 1,
                    output_tokens: 1,
                    latency_ms: 1,
                    schema_valid: Some(true),
                },
            )
        };
        let contract = EquivalenceContract {
            mode: crate::DiffMode::ApiResponse,
            ..EquivalenceContract::default()
        };
        assert_eq!(
            compare_response(
                "case",
                &[outcome("hello  world", "a"), outcome("hello world", "b")],
                &contract
            )
            .verdict,
            Verdict::Equivalent
        );
        let mut changed = outcome("hello world", "b");
        changed.1.tool_calls[0].name = "write".into();
        assert_eq!(
            compare_response("case", &[outcome("hello world", "a"), changed], &contract).verdict,
            Verdict::Mismatch
        );
    }
}
