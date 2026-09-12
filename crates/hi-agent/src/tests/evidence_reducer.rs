use super::common::*;
use super::*;
use hi_ai::Content;
use std::sync::Arc;

fn diagnostic_failure_command() -> &'static str {
    "printf 'running 3 tests\\nerror: mismatch\\nFAILED tests::it_breaks\\n'; \
i=0; while [ \"$i\" -lt 200 ]; do printf 'ok noise\\n'; i=$((i+1)); done; \
printf 'UNIQUE_MIDDLE_LINE\\n'; \
i=0; while [ \"$i\" -lt 200 ]; do printf 'ok noise\\n'; i=$((i+1)); done; \
printf 'test result: FAILED\\n'; exit 1"
}

fn canned_mismatch_hook() -> hi_tools::EvidenceReducerHook {
    Arc::new(|source: &str, is_error: bool| {
        let quote = "error: mismatch";
        if !source.contains(quote) {
            return Err("missing-quote");
        }
        Ok(serde_json::json!({
            "schema": hi_tools::REDUCER_RECEIPT_SCHEMA,
            "source_sha256": hi_tools::sha256_hex(source.as_bytes()),
            "status": if is_error { "failure" } else { "success" },
            "uncertain": false,
            "evidence": [{"kind": "failure", "quote": quote}]
        })
        .to_string())
    })
}

fn tool_outputs(agent: &Agent) -> Vec<String> {
    agent
        .messages()
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn agent_bash_reduces_after_condense_with_canned_receipt() {
    let mut cfg = config();
    cfg.memory.evidence_preserving_reducer = true;
    cfg.loop_limits.max_recovery_interventions = 0;
    let mut agent = agent(
        vec![
            bash_completion(diagnostic_failure_command()),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent.runtime.process_runner().set_evidence_reducer(
        hi_tools::EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        },
        Some(canned_mismatch_hook()),
    );
    agent.run_turn("run the tests", &mut NullUi).await.unwrap();
    let outputs = tool_outputs(&agent);
    let bash = outputs
        .iter()
        .find(|output| output.contains(hi_tools::EVIDENCE_RECEIPT_PREFIX))
        .unwrap_or_else(|| panic!("expected evidence receipt in bash output: {outputs:?}"));
    assert!(bash.contains("error: mismatch"), "{bash}");
    assert!(
        !bash.contains("UNIQUE_MIDDLE_LINE"),
        "condense must run first: {bash}"
    );
}

#[tokio::test]
async fn agent_flag_on_without_receipt_keeps_condensed_log() {
    let mut cfg = config();
    cfg.memory.evidence_preserving_reducer = true;
    cfg.loop_limits.max_recovery_interventions = 0;
    let mut agent = agent(
        vec![
            bash_completion(diagnostic_failure_command()),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent.run_turn("run the tests", &mut NullUi).await.unwrap();
    let outputs = tool_outputs(&agent);
    let bash = outputs
        .iter()
        .find(|output| output.contains("error: mismatch"))
        .unwrap_or_else(|| panic!("expected condensed diagnostic log: {outputs:?}"));
    assert!(
        !bash.contains(hi_tools::EVIDENCE_RECEIPT_PREFIX),
        "missing nested receipt must fail open: {bash}"
    );
}
