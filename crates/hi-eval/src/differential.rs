//! Route/provider/model differential comparison for agent runs.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DifferentialObservation {
    pub route: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub outcome: Option<String>,
    pub event_trace_hash: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_microusd: Option<u64>,
    pub sandbox_backend: Option<String>,
    pub sandbox_status: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DifferentialDifference {
    pub field: &'static str,
    pub left: String,
    pub right: String,
}

impl DifferentialObservation {
    /// Extract the common comparison fields from a version-2 hi report. All
    /// additions are optional so older reports remain comparable.
    pub fn from_report(report: &Value) -> Self {
        let provider = report
            .get("route")
            .and_then(|route| route.get("provider"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let model = report
            .get("route")
            .and_then(|route| route.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let route = match (provider.as_deref(), model.as_deref()) {
            (Some(provider), Some(model)) => Some(format!("{provider}:{model}")),
            (Some(provider), None) => Some(provider.to_string()),
            _ => None,
        };
        let outcome = report
            .get("outcome")
            .and_then(|outcome| outcome.get("status"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let usage = report.get("usage").and_then(|usage| usage.get("turn"));
        let normalized = report
            .get("usage")
            .and_then(|usage| usage.get("normalized"));
        let sandbox = report.get("sandbox");
        Self {
            route,
            provider,
            model,
            outcome,
            event_trace_hash: report
                .get("event_trace_hash")
                .and_then(Value::as_str)
                .map(str::to_string),
            input_tokens: usage
                .and_then(|usage| usage.get("input_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            output_tokens: usage
                .and_then(|usage| usage.get("output_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            cost_microusd: normalized
                .and_then(|normalized| normalized.get("cost"))
                .and_then(|cost| cost.get("total_microusd"))
                .and_then(Value::as_u64),
            sandbox_backend: sandbox
                .and_then(|sandbox| sandbox.get("backend"))
                .and_then(Value::as_str)
                .map(str::to_string),
            sandbox_status: sandbox
                .and_then(|sandbox| sandbox.get("status"))
                .and_then(Value::as_str)
                .map(str::to_string),
        }
    }
}

pub fn compare(
    left: &DifferentialObservation,
    right: &DifferentialObservation,
) -> Vec<DifferentialDifference> {
    let mut differences = Vec::new();
    compare_field(&mut differences, "route", &left.route, &right.route);
    compare_field(
        &mut differences,
        "provider",
        &left.provider,
        &right.provider,
    );
    compare_field(&mut differences, "model", &left.model, &right.model);
    compare_field(&mut differences, "outcome", &left.outcome, &right.outcome);
    compare_field(
        &mut differences,
        "event_trace_hash",
        &left.event_trace_hash,
        &right.event_trace_hash,
    );
    compare_field(
        &mut differences,
        "input_tokens",
        &left.input_tokens,
        &right.input_tokens,
    );
    compare_field(
        &mut differences,
        "output_tokens",
        &left.output_tokens,
        &right.output_tokens,
    );
    compare_field(
        &mut differences,
        "cost_microusd",
        &left.cost_microusd,
        &right.cost_microusd,
    );
    compare_field(
        &mut differences,
        "sandbox_backend",
        &left.sandbox_backend,
        &right.sandbox_backend,
    );
    compare_field(
        &mut differences,
        "sandbox_status",
        &left.sandbox_status,
        &right.sandbox_status,
    );
    differences
}

pub fn hash_event_trace(events: &[Value]) -> String {
    let mut hasher = Hasher::new();
    for event in events {
        if let Ok(bytes) = serde_json::to_vec(event) {
            hasher.update(&bytes);
            hasher.update(b"\n");
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn compare_field<T: std::fmt::Debug + PartialEq>(
    differences: &mut Vec<DifferentialDifference>,
    field: &'static str,
    left: &T,
    right: &T,
) {
    if left != right {
        differences.push(DifferentialDifference {
            field,
            left: format!("{left:?}"),
            right: format!("{right:?}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_fields_are_compared_without_requiring_new_schema() {
        let report = serde_json::json!({
            "route": {"provider":"openai","model":"gpt"},
            "outcome": {"status":"completed"},
            "usage": {"turn": {"input_tokens": 3, "output_tokens": 4}},
            "sandbox": {"backend":"pipe-wrap","status":"enforced"}
        });
        let left = DifferentialObservation::from_report(&report);
        let mut right = left.clone();
        right.output_tokens = 9;
        assert_eq!(compare(&left, &right)[0].field, "output_tokens");
    }

    #[test]
    fn event_trace_hash_is_deterministic() {
        let events = vec![serde_json::json!({"sequence": 1})];
        assert_eq!(hash_event_trace(&events), hash_event_trace(&events));
    }
}
