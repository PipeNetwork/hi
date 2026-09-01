use hi_workflow::{
    AgentResult, DeclarativeOutcome, DeclarativeRunParams, DeclarativeWorkflow, HostError,
    WorkflowHostRequest, run_declarative_workflow,
};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn result(id: &str, success: bool, output: serde_json::Value) -> AgentResult {
    AgentResult {
        agent_id: id.into(),
        success,
        output,
        cancelled: false,
        tokens_used: 1,
        duration_ms: 1,
    }
}

#[test]
fn parses_and_validates_model() {
    let workflow = DeclarativeWorkflow::from_json(r#"{
      "metadata":{"name":"review","description":"Review code","version":"1"},
      "steps":[
        {"type":"agent","id":"reviewer","prompt":"Review {{args.target}}"},
        {"type":"if_agent_success","agent":"reviewer","then":[{"type":"complete","result":{"status":"ok"}}],"else":[{"type":"pause","kind":"verification","message":"review failed"}]}
      ]
    }"#).unwrap();
    workflow.validate().unwrap();
    assert_eq!(workflow.metadata.name, "review");
}

#[test]
fn rejects_invalid_static_structure() {
    let workflow = DeclarativeWorkflow::from_json(
        r#"{
      "metadata":{"name":""},
      "steps":[
        {"type":"if_agent_success","agent":"missing","then":[]},
        {"type":"agent","id":"same","prompt":"{{"},
        {"type":"agent","id":"same","prompt":"x"},
        {"type":"parallel_agents","jobs":[]}
      ]
    }"#,
    )
    .unwrap();
    let error = workflow.validate().unwrap_err().errors.join("\n");
    assert!(error.contains("metadata.name"));
    assert!(error.contains("before it is defined"));
    assert!(error.contains("duplicate agent id"));
    assert!(error.contains("must not be empty"));
    assert!(error.contains("unclosed"));
}

#[test]
fn denies_unknown_json_fields() {
    let error = DeclarativeWorkflow::from_json(r#"{"metadata":{"name":"x","extra":1},"steps":[]}"#)
        .unwrap_err();
    assert!(error.to_string().contains("unknown field"));
}

#[tokio::test]
async fn executes_serial_branch_templates_and_events() {
    let workflow = DeclarativeWorkflow::from_json(r#"{
      "metadata":{"name":"flow"},
      "steps":[
        {"type":"phase","title":"Working on {{args.target}}"},
        {"type":"agent","id":"first","prompt":"Inspect {{args.target}}","label":"job {{args.target}}"},
        {"type":"if_agent_success","agent":"first","then":[
          {"type":"log","message":"Found {{agents.first.output.summary}}"},
          {"type":"complete","result":{"target":"{{args.target}}","answer":"{{agents.first.output.summary}}"}}
        ],"else":[{"type":"pause","kind":"verification","message":"failed"}]}
      ]
    }"#).unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    let host = tokio::spawn(async move {
        let mut kinds = Vec::new();
        while let Some(request) = rx.recv().await {
            kinds.push(request.kind());
            match request {
                WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                    assert_eq!(count, 1);
                    reply.send(Ok(())).unwrap();
                }
                WorkflowHostRequest::SpawnAgent { opts, reply } => {
                    assert_eq!(opts.prompt, "Inspect src/lib.rs");
                    assert_eq!(opts.label.as_deref(), Some("job src/lib.rs"));
                    reply
                        .send(Ok(result("host-1", true, json!({"summary":"clean"}))))
                        .unwrap();
                }
                WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                    assert_eq!(count, 1);
                    reply.send(Ok(())).unwrap();
                }
                WorkflowHostRequest::Phase { title, replayed } => {
                    assert_eq!(title, "Working on src/lib.rs");
                    assert!(!replayed);
                }
                WorkflowHostRequest::Log { message, replayed } => {
                    assert_eq!(message, "Found clean");
                    assert!(!replayed);
                    break;
                }
                _ => panic!("unexpected host request"),
            }
        }
        kinds
    });
    let outcome = run_declarative_workflow(DeclarativeRunParams {
        workflow,
        args: json!({"target":"src/lib.rs"}),
        host_tx: tx,
        cancel: CancellationToken::new(),
    })
    .await;
    match outcome {
        DeclarativeOutcome::Completed { result, agents } => {
            assert_eq!(result, json!({"target":"src/lib.rs","answer":"clean"}));
            assert_eq!(agents.len(), 1);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
    assert_eq!(
        host.await.unwrap(),
        [
            "phase",
            "reserve_agent_calls",
            "spawn_agent",
            "release_agent_calls",
            "log"
        ]
    );
}

#[tokio::test]
async fn parallel_results_are_deterministic_despite_reply_order() {
    let workflow = DeclarativeWorkflow::from_json(
        r#"{
      "metadata":{"name":"parallel"},
      "steps":[{"type":"parallel_agents","jobs":[
        {"id":"a","prompt":"A"},{"id":"b","prompt":"B"}
      ]},{"type":"complete","result":"{{agents.b.output}}"}]
    }"#,
    )
    .unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    tokio::spawn(async move {
        let mut replies = Vec::new();
        while let Some(request) = rx.recv().await {
            match request {
                WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
                    assert_eq!(count, 2);
                    reply.send(Ok(())).unwrap();
                }
                WorkflowHostRequest::SpawnAgent { opts, reply } => {
                    replies.push((opts.prompt, reply));
                    if replies.len() == 2 {
                        let (b_prompt, b) = replies.pop().unwrap();
                        let (a_prompt, a) = replies.pop().unwrap();
                        assert_eq!(a_prompt, "A");
                        assert_eq!(b_prompt, "B");
                        b.send(Ok(result("b", true, json!(2)))).unwrap();
                        a.send(Ok(result("a", true, json!(1)))).unwrap();
                    }
                }
                WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                    assert_eq!(count, 2);
                    reply.send(Ok(())).unwrap();
                    break;
                }
                _ => panic!("unexpected host request"),
            }
        }
    });
    let outcome = run_declarative_workflow(DeclarativeRunParams {
        workflow,
        args: json!({}),
        host_tx: tx,
        cancel: CancellationToken::new(),
    })
    .await;
    match outcome {
        DeclarativeOutcome::Completed { result, agents } => {
            assert_eq!(result, json!("2"));
            assert_eq!(
                agents.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
                ["a", "b"]
            );
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn reports_pause_false_branch_and_budget() {
    let workflow = DeclarativeWorkflow::from_json(r#"{
      "metadata":{"name":"branch"},"steps":[
        {"type":"agent","id":"check","prompt":"check"},
        {"type":"if_agent_success","agent":"check","then":[{"type":"complete"}],"else":[{"type":"pause","kind":"no_progress","message":"No {{args.item}}"}]}
      ]
    }"#).unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    tokio::spawn(async move {
        while let Some(request) = rx.recv().await {
            match request {
                WorkflowHostRequest::ReserveAgentCalls { reply, .. } => reply.send(Ok(())).unwrap(),
                WorkflowHostRequest::SpawnAgent { reply, .. } => {
                    reply.send(Ok(result("x", false, json!(null)))).unwrap()
                }
                WorkflowHostRequest::ReleaseAgentCalls { reply, .. } => {
                    reply.send(Ok(())).unwrap();
                    break;
                }
                _ => panic!(),
            }
        }
    });
    match run_declarative_workflow(DeclarativeRunParams {
        workflow: workflow.clone(),
        args: json!({"item":"progress"}),
        host_tx: tx,
        cancel: CancellationToken::new(),
    })
    .await
    {
        DeclarativeOutcome::Paused {
            message, agents, ..
        } => {
            assert_eq!(message, "No progress");
            assert_eq!(agents.len(), 1);
        }
        other => panic!("{other:?}"),
    }

    let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    tokio::spawn(async move {
        if let Some(WorkflowHostRequest::ReserveAgentCalls { reply, .. }) = rx.recv().await {
            reply
                .send(Err(HostError::AgentCallQuotaExceeded {
                    requested: 1,
                    maximum: 0,
                }))
                .unwrap();
        }
    });
    match run_declarative_workflow(DeclarativeRunParams {
        workflow,
        args: json!({}),
        host_tx: tx,
        cancel: CancellationToken::new(),
    })
    .await
    {
        DeclarativeOutcome::BudgetExceeded { message, agents } => {
            assert!(message.contains("requested 1"));
            assert!(agents.is_empty());
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn cancellation_interrupts_waiting_agent_and_releases_reservation() {
    let workflow = DeclarativeWorkflow::from_json(
        r#"{"metadata":{"name":"cancel"},"steps":[{"type":"agent","id":"a","prompt":"wait"}]}"#,
    )
    .unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    let cancel = CancellationToken::new();
    let cancel_host = cancel.clone();
    let host = tokio::spawn(async move {
        let mut held_reply = None;
        while let Some(request) = rx.recv().await {
            match request {
                WorkflowHostRequest::ReserveAgentCalls { reply, .. } => reply.send(Ok(())).unwrap(),
                WorkflowHostRequest::SpawnAgent { reply, .. } => {
                    held_reply = Some(reply);
                    cancel_host.cancel();
                }
                WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
                    assert_eq!(count, 1);
                    reply.send(Ok(())).unwrap();
                    break;
                }
                _ => panic!(),
            }
        }
        drop(held_reply);
    });
    let outcome = run_declarative_workflow(DeclarativeRunParams {
        workflow,
        args: json!({}),
        host_tx: tx,
        cancel,
    })
    .await;
    assert!(matches!(outcome, DeclarativeOutcome::Cancelled { .. }));
    host.await.unwrap();
}

#[tokio::test]
async fn unknown_template_variable_is_structured_failure() {
    let workflow = DeclarativeWorkflow::from_json(
        r#"{"metadata":{"name":"bad"},"steps":[{"type":"log","message":"{{args.missing}}"}]}"#,
    )
    .unwrap();
    let (tx, _rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    match run_declarative_workflow(DeclarativeRunParams {
        workflow,
        args: json!({}),
        host_tx: tx,
        cancel: CancellationToken::new(),
    })
    .await
    {
        DeclarativeOutcome::Failed { error, .. } => {
            assert!(error.to_string().contains("unknown variable"))
        }
        other => panic!("{other:?}"),
    }
}
