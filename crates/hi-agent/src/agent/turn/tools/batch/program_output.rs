//! Provider-facing program results: preserve the selected value and failure
//! diagnostics without replaying every successful intermediate tool payload.

use hi_tools::TruncationState;
use hi_workflow::ProgramOutcome;
use serde_json::{Value, json};

pub(super) fn program_output(outcome: &ProgramOutcome) -> (String, TruncationState) {
    let (status, calls) = match outcome {
        ProgramOutcome::Succeeded { calls, .. } => ("succeeded", calls),
        ProgramOutcome::Failed { calls, .. } => ("failed", calls),
        ProgramOutcome::Cancelled { calls } => ("cancelled", calls),
    };
    let call_summaries = calls
        .iter()
        .map(|call| {
            let mut summary =
                json!({"index": call.index, "name": call.name, "status": call.status});
            if call.status != "succeeded" {
                summary["output"] = Value::String(call.output.clone());
            }
            summary
        })
        .collect::<Vec<_>>();
    let mut aggregate = json!({"status": status, "calls": call_summaries});
    match outcome {
        ProgramOutcome::Succeeded { result, .. } => aggregate["result"] = result.clone(),
        ProgramOutcome::Failed { error, .. } => aggregate["error"] = Value::String(error.clone()),
        ProgramOutcome::Cancelled { .. } => aggregate["error"] = json!("program cancelled"),
    }
    let raw = serde_json::to_string(&aggregate).expect("program results are JSON values");
    let (mut preview, truncation) = hi_tools::bound_tool_content(raw);
    if truncation == TruncationState::Complete {
        return (preview, truncation);
    }
    let TruncationState::Truncated { original_bytes, .. } = truncation else {
        unreachable!();
    };
    let mut envelope = json!({
        "status": status,
        "output_truncated": true,
        "calls_total": calls.len(),
        "calls_failed": calls.iter().filter(|call| call.status != "succeeded").count(),
        "preview": "",
    });
    // Many intermediate failures or call summaries can exhaust the budget
    // even when the selected answer is tiny. Keep that answer structured if
    // it fits on its own; only a large selected value needs a text preview.
    if let Some(result) = aggregate.get("result") {
        envelope["result"] = result.clone();
        let raw = serde_json::to_string(&envelope).expect("program envelope is JSON");
        if hi_tools::bound_tool_content(raw).1 != TruncationState::Complete {
            envelope.as_object_mut().unwrap().remove("result");
        }
    }
    // String-clipping a JSON envelope produces an unusable protocol fragment.
    // Keep the terminal status and call counts structured, and label the
    // bounded head/tail as a preview when the selected result itself is huge.
    loop {
        envelope["preview"] = Value::String(preview.clone());
        let raw = serde_json::to_string(&envelope).expect("program preview is JSON");
        let (bounded, state) = hi_tools::bound_tool_content(raw);
        if state == TruncationState::Complete {
            return (
                bounded.clone(),
                TruncationState::Truncated {
                    original_bytes,
                    retained_bytes: bounded.len() as u64,
                },
            );
        }
        let count = preview.chars().count();
        if count <= 128 {
            // The envelope was checked with an empty preview above. Avoid a
            // fixed-size truncation marker preventing progress near the cap.
            preview.clear();
            continue;
        }
        let head = preview.chars().take(count / 4).collect::<String>();
        let tail = preview.chars().skip(count - count / 4).collect::<String>();
        preview = format!("{head}\n[preview shortened]\n{tail}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_workflow::ProgramToolResult;

    #[test]
    fn program_selected_output_avoids_duplicate_success_payloads_but_keeps_errors() {
        let outcome = ProgramOutcome::Succeeded {
            result: json!({"answer": 42}),
            calls: vec![
                ProgramToolResult {
                    index: 0,
                    name: "read".into(),
                    status: "succeeded".into(),
                    output: "source ".repeat(1000),
                },
                ProgramToolResult {
                    index: 1,
                    name: "read".into(),
                    status: "failed".into(),
                    output: "Error: src/missing.rs was not found".into(),
                },
            ],
        };
        let (output, truncated) = program_output(&outcome);
        let ProgramOutcome::Succeeded { result, calls } = &outcome else {
            unreachable!()
        };
        let previous = hi_tools::bound_tool_content(
            serde_json::to_string(
                &json!({"status": "succeeded", "result": result, "calls": calls}),
            )
            .unwrap(),
        )
        .0;
        let previous_tokens =
            hi_ai::estimate_messages_tokens(&[hi_ai::Message::tool_result("program", &previous)]);
        let current_tokens =
            hi_ai::estimate_messages_tokens(&[hi_ai::Message::tool_result("program", &output)]);
        eprintln!(
            "selected program result fixture: {} -> {} output bytes; {previous_tokens} -> {current_tokens} estimated input tokens",
            previous.len(),
            output.len()
        );
        assert!(previous.len() > output.len() * 3);
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(truncated, TruncationState::Complete);
        assert_eq!(value["result"]["answer"], 42);
        assert!(value["calls"][0].get("output").is_none());
        assert_eq!(
            value["calls"][1]["output"],
            "Error: src/missing.rs was not found"
        );
        assert!(output.len() < 300);
    }

    #[test]
    fn oversized_program_values_keep_valid_json_and_explicit_truncation() {
        for result in [
            json!("quote \" slash \\ unicode 🦀\n".repeat(10_000)),
            json!((0..30_000).collect::<Vec<_>>()),
        ] {
            let (output, truncated) = program_output(&ProgramOutcome::Succeeded {
                result,
                calls: Vec::new(),
            });
            let value: Value = serde_json::from_str(&output).unwrap();
            assert_eq!(value["status"], "succeeded");
            assert_eq!(value["output_truncated"], true);
            assert!(matches!(truncated, TruncationState::Truncated { .. }));
            assert_eq!(
                hi_tools::bound_tool_content(output).1,
                TruncationState::Complete,
                "transcript storage must not clip the valid envelope again"
            );
        }
    }

    #[test]
    fn many_intermediate_failures_do_not_hide_a_small_selected_answer() {
        let calls = (0..100)
            .map(|index| ProgramToolResult {
                index,
                name: "read".into(),
                status: "failed".into(),
                output: format!("missing candidate file {index}\n").repeat(20),
            })
            .collect();
        let (output, truncation) = program_output(&ProgramOutcome::Succeeded {
            result: json!({"chosen_path": "src/parser.rs"}),
            calls,
        });
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["result"]["chosen_path"], "src/parser.rs");
        assert_eq!(value["calls_failed"], 100);
        assert!(
            value["preview"]
                .as_str()
                .unwrap()
                .contains("missing candidate")
        );
        assert!(matches!(truncation, TruncationState::Truncated { .. }));
    }

    #[test]
    fn selected_answer_near_the_output_cap_does_not_stall_preview_reduction() {
        let selected = "answer".repeat(800);
        let (output, truncation) = program_output(&ProgramOutcome::Succeeded {
            result: json!(selected),
            calls: vec![ProgramToolResult {
                index: 0,
                name: "read".into(),
                status: "failed".into(),
                output: "failure details\n".repeat(10_000),
            }],
        });
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["result"], selected);
        assert!(matches!(truncation, TruncationState::Truncated { .. }));
        assert_eq!(
            hi_tools::bound_tool_content(output).1,
            TruncationState::Complete
        );
    }
}
