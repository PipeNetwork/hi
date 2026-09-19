use super::*;
fn accepted_value() -> Value {
    json!({"id":"managed-id","choices":[{"index":0,"message":{"role":"assistant","content":"I will inspect the repository."},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":12},"pipe":{"status":"completed","verification":{"status":"passed","scope":"client_reported_execution_evidence"},"total_charge_usd":"0.001000","unresolved_micros":0}})
}
fn wire(value: &Value, done: bool) -> Vec<u8> {
    let first = json!({"id":value["id"],"choices":[{"index":0,"delta":value["choices"][0]["message"],"finish_reason":null}]});
    let last = json!({"id":value["id"],"choices":[{"index":0,"delta":{},"finish_reason":value["choices"][0]["finish_reason"]}],"pipe":value["pipe"],"usage":value["usage"]});
    format!(
        "data: {first}\n\ndata: {last}\n\n{}",
        if done { "data: [DONE]\n\n" } else { "" }
    )
    .into_bytes()
}
#[test]
fn managed_json_sse_equivalence_and_missing_terminal_are_errors() {
    let v = accepted_value();
    let decoded = parse_response(&wire(&v, true)).unwrap();
    assert_eq!(
        accepted(&v, &[]).unwrap().text,
        accepted(&decoded, &[]).unwrap().text
    );
    assert!(parse_response(&wire(&v, false)).is_err());
    let mut missing = v.clone();
    missing.as_object_mut().unwrap().remove("pipe");
    assert!(accepted(&missing, &[]).is_err());
    let mut rejected = v.clone();
    rejected["pipe"]["status"] = json!("insufficient_evidence");
    assert!(accepted(&rejected, &[]).is_err());
    let mut lost = v.clone();
    lost["pipe"]["result_not_retained"] = json!(true);
    assert!(accepted(&lost, &[]).is_err());
    assert!(parse_response(b"data: broken\n\ndata: [DONE]\n\n").is_err());
}
#[test]
fn malformed_or_truncated_managed_tools_cannot_execute() {
    let tools = vec![ToolSpec {
        name: "read".into(),
        description: "read".into(),
        parameters: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
    }];
    let mut v = accepted_value();
    v["choices"][0]["message"]["tool_calls"] = json!([{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"file\"}"}}]);
    v["choices"][0]["finish_reason"] = json!("tool_calls");
    assert_eq!(accepted(&v, &tools).unwrap().tool_calls.len(), 1);
    v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json!("{\"path\":");
    assert!(accepted(&v, &tools).is_err());
    v["choices"][0]["finish_reason"] = json!("length");
    assert!(accepted(&v, &tools).is_err());
}
fn client(path: &std::path::Path) -> PipeClient {
    PipeClient::new("http://127.0.0.1:1/v1", "fixture")
        .with_managed(path.to_path_buf(), ManagedSettings::default())
}
#[test]
fn crash_preserves_original_budget_and_never_repeats_started_tool() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("managed.json");
    let first = client(&path);
    first.begin_managed_turn("turn1".into(), false).unwrap();
    first
        .managed
        .update(|j| {
            j.budget = 1500000;
            j.operations.push(Operation {
                auxiliary: false,
                key: "key".into(),
                input_hash: "input".into(),
                payload_hash: "hash".into(),
                payload: "{}".into(),
                reserved: 1000000,
                charge: Some(123),
                unresolved: false,
                status: "completed".into(),
                completion: Some(PipeCompletion::default()),
                tools: BTreeMap::from([("call".into(), ToolState::Pending)]),
            });
            Ok(())
        })
        .unwrap();
    assert!(first.managed_tool_start("call").unwrap().is_none());
    drop(first);
    let resumed = client(&path);
    resumed.begin_managed_turn("turn1".into(), true).unwrap();
    assert_eq!(resumed.managed.read(|j| Ok(j.budget)).unwrap(), 1500000);
    assert!(resumed.managed_tool_start("call").is_err());
    assert!(resumed.recover_managed_tools(&mut Vec::new()).is_err());
    assert!(resumed.begin_managed_turn("turn2".into(), false).is_err());
}
#[test]
fn completed_tool_results_are_reused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("managed.json");
    let client = client(&path);
    client.begin_managed_turn("turn".into(), false).unwrap();
    let outcome = crate::tools::interrupted_outcome();
    client
        .managed
        .update(|j| {
            j.operations.push(Operation {
                auxiliary: false,
                key: "key".into(),
                input_hash: "input".into(),
                payload_hash: "hash".into(),
                payload: "{}".into(),
                reserved: 1000000,
                charge: Some(1),
                unresolved: false,
                status: "completed".into(),
                completion: None,
                tools: BTreeMap::from([(
                    "a".into(),
                    ToolState::Completed {
                        outcome: outcome.clone(),
                    },
                )]),
            });
            Ok(())
        })
        .unwrap();
    assert_eq!(client.managed_tool_start("a").unwrap(), Some(outcome));
    assert!(micros("0.0000001").is_err());
    assert_eq!(micros("1.010001").unwrap(), 1010001);
}

#[test]
fn recovery_repairs_partial_transcript_without_restoring_compacted_batches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("managed.json");
    let client = client(&path);
    client.begin_managed_turn("turn".into(), false).unwrap();
    let completion = PipeCompletion {
        tool_calls: vec![
            ToolCall {
                id: "a".into(),
                name: "read".into(),
                arguments: r#"{"path":"a"}"#.into(),
            },
            ToolCall {
                id: "b".into(),
                name: "read".into(),
                arguments: r#"{"path":"b"}"#.into(),
            },
        ],
        ..Default::default()
    };
    let outcome = crate::tools::interrupted_outcome();
    client
        .managed
        .update(|j| {
            j.operations.push(Operation {
                auxiliary: false,
                key: "key".into(),
                input_hash: "input".into(),
                payload_hash: "hash".into(),
                payload: "{}".into(),
                reserved: 1000000,
                charge: Some(1),
                unresolved: false,
                status: "completed".into(),
                completion: Some(completion.clone()),
                tools: ["a", "b"]
                    .into_iter()
                    .map(|id| {
                        (
                            id.into(),
                            ToolState::Completed {
                                outcome: outcome.clone(),
                            },
                        )
                    })
                    .collect(),
            });
            Ok(())
        })
        .unwrap();
    let mut messages = vec![
        crate::turn::assistant_message(&completion),
        Message::tool_result("a", reported_outcome(&outcome)),
    ];
    client.recover_managed_tools(&mut messages).unwrap();
    assert_eq!(messages.len(), 3);
    client.recover_managed_tools(&mut messages).unwrap();
    assert_eq!(messages.len(), 3);
    client
        .managed
        .update(|j| {
            let mut summary = j.operations[0].clone();
            summary.auxiliary = true;
            summary.tools.clear();
            summary.completion = Some(PipeCompletion {
                text: "summary".into(),
                ..Default::default()
            });
            j.operations.push(summary);
            Ok(())
        })
        .unwrap();
    let mut compacted = vec![Message::user("Continue from summary")];
    client.recover_managed_tools(&mut compacted).unwrap();
    assert_eq!(compacted.len(), 1);
    assert!(client.validate_pending_workflow("turn", false).is_err());
}

#[test]
fn compaction_preserves_original_call_ids_and_structured_results() {
    let output = json!({"evidence_source":"client_reported","status":"succeeded","process":{"exit_code":0},"content":"x".repeat(5000)}).to_string();
    let source = vec![
        Message::user("Fix it"),
        Message::assistant(vec![Content::ToolCall {
            id: "test-call".into(),
            name: "bash".into(),
            arguments: r#"{"command":"python3 -m unittest"}"#.into(),
        }]),
        Message::tool_result("test-call", output),
    ];
    let mut summary = vec![
        Message::user("Fix it"),
        Message::assistant(vec![Content::Text("Summary".into())]),
    ];
    retain_execution_evidence(&source, &mut summary);
    assert_eq!(summary.len(), 4);
    let Content::ToolResult { call_id, output } = &summary[3].content[0] else {
        panic!("result missing")
    };
    assert_eq!(call_id, "test-call");
    let result: Value = serde_json::from_str(output).unwrap();
    assert_eq!(result["process"]["exit_code"], 0);
    assert_eq!(result["content"].as_str().unwrap().len(), 1024);
    assert_eq!(result["content_omission"]["original_bytes"], 5000);
}

#[tokio::test]
async fn recovered_final_is_reused_without_network_or_new_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("managed.json");
    let first = client(&path);
    first.begin_managed_turn("turn".into(), false).unwrap();
    let completion = accepted(&accepted_value(), &[]).unwrap();
    first
        .managed
        .update(|j| {
            j.operations.push(Operation {
                auxiliary: false,
                key: "key".into(),
                input_hash: "hash".into(),
                payload_hash: "hash".into(),
                payload: "{}".into(),
                reserved: 1000000,
                charge: Some(1000),
                unresolved: false,
                status: "completed".into(),
                completion: Some(completion),
                tools: BTreeMap::new(),
            });
            Ok(())
        })
        .unwrap();
    drop(first);
    let resumed = client(&path);
    resumed.begin_managed_turn("turn".into(), true).unwrap();
    resumed.recover_managed_tools(&mut Vec::new()).unwrap();
    let result = resumed
        .stream_managed(json!({}), &[], &mut |_| {}, &TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(result.text, "I will inspect the repository.");
    assert_eq!(resumed.managed.read(|j| Ok(j.operations.len())).unwrap(), 1);
}
