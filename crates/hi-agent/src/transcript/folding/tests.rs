use super::*;
use hi_ai::Message;

const READ_ARGS: &str = r#"{"path":"src/parser.rs"}"#;

fn append_tool(transcript: &mut Transcript, id: &str, name: &str, arguments: &str, output: &str) {
    transcript.push_assistant_with_results(
        vec![Content::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }],
        vec![(id.into(), output.into())],
    );
}

fn results(transcript: &Transcript) -> Vec<&str> {
    transcript
        .as_slice()
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect()
}

fn fold_reads(transcript: &mut Transcript) {
    transcript.fold_superseded_file_reads(&read_call_key(READ_ARGS).unwrap());
}

#[test]
fn reused_call_id_does_not_erase_the_newest_read() {
    let mut transcript = Transcript::new(vec![Message::user("inspect the parser")]);
    append_tool(
        &mut transcript,
        "call_1",
        "read",
        READ_ARGS,
        &"old source\n".repeat(100),
    );
    let current = "current parser implementation\n".repeat(100);
    append_tool(&mut transcript, "call_1", "read", READ_ARGS, &current);
    transcript.validate_for_provider().unwrap();
    fold_reads(&mut transcript);
    let outputs = results(&transcript);
    assert!(outputs[0].contains("superseded read"));
    assert_eq!(
        outputs[1], current,
        "folding treated a reused ID as the older result occurrence"
    );
    transcript.validate_for_provider().unwrap();
}

#[test]
fn reused_call_id_on_an_unrelated_tool_keeps_its_evidence() {
    let mut transcript = Transcript::new(vec![Message::user("fix the parser")]);
    append_tool(
        &mut transcript,
        "call_1",
        "read",
        READ_ARGS,
        &"old source\n".repeat(100),
    );
    let failure =
        "FAILED parser::rejects_invalid_input: unexpected token at parser.rs:42\n".repeat(30);
    append_tool(
        &mut transcript,
        "call_1",
        "bash",
        r#"{"command":"cargo test parser"}"#,
        &failure,
    );
    append_tool(
        &mut transcript,
        "call_2",
        "read",
        READ_ARGS,
        "current source",
    );
    fold_reads(&mut transcript);
    assert_eq!(
        results(&transcript)[1],
        failure,
        "read folding erased a different tool's failure"
    );
    transcript.validate_for_provider().unwrap();
}

#[test]
fn legacy_duplicate_ids_within_a_batch_are_paired_by_occurrence() {
    let mut transcript = Transcript::new(vec![Message::user("inspect the parser")]);
    transcript.push_assistant_with_results(
        (0..2)
            .map(|_| Content::ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: READ_ARGS.into(),
            })
            .collect(),
        vec![
            ("call_1".into(), "old source\n".repeat(100)),
            (
                "call_1".into(),
                "current parser implementation\n".repeat(100),
            ),
        ],
    );
    transcript.validate_for_provider().unwrap();
    fold_reads(&mut transcript);
    assert!(results(&transcript)[0].contains("superseded read"));
    assert!(results(&transcript)[1].starts_with("current parser implementation"));
}

#[test]
fn batched_result_order_does_not_change_the_latest_call() {
    let mut transcript = Transcript::new(vec![Message::user("inspect the parser")]);
    transcript.push_assistant_with_results(
        ["older", "newer"]
            .into_iter()
            .map(|id| Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: READ_ARGS.into(),
            })
            .collect(),
        vec![
            ("newer".into(), "current source".into()),
            ("older".into(), "old source\n".repeat(100)),
        ],
    );
    // Imported sessions can retain a valid provider result order different
    // from the constructor's normalized call order.
    transcript.mutate_slice().swap(2, 3);
    fold_reads(&mut transcript);
    assert_eq!(results(&transcript)[0], "current source");
    assert!(results(&transcript)[1].contains("superseded read"));
    transcript.validate_for_provider().unwrap();
}

#[test]
fn folding_tiny_reads_never_inflates_request_context() {
    let mut transcript = Transcript::new(vec![Message::user("inspect the parser")]);
    for index in 0..100 {
        append_tool(
            &mut transcript,
            &format!("read_{index}"),
            "read",
            READ_ARGS,
            "1: true\n",
        );
    }
    let before_bytes = serde_json::to_vec(transcript.as_slice()).unwrap().len();
    let before_estimate = hi_ai::estimate_messages_tokens(transcript.as_slice());
    fold_reads(&mut transcript);
    let after_bytes = serde_json::to_vec(transcript.as_slice()).unwrap().len();
    let after_estimate = hi_ai::estimate_messages_tokens(transcript.as_slice());
    println!(
        "tiny-read fixture: serialized bytes {before_bytes} -> {after_bytes}; estimated message tokens {before_estimate} -> {after_estimate}"
    );
    assert!(
        after_bytes <= before_bytes,
        "compaction inflated a 100-read transcript"
    );
    assert!(after_estimate <= before_estimate);
}

#[test]
fn folded_observations_remain_idempotent_after_serialized_resume() {
    let mut transcript = Transcript::new(vec![Message::user("inspect the parser")]);
    append_tool(
        &mut transcript,
        "call_1",
        "read",
        READ_ARGS,
        &"old source\n".repeat(100),
    );
    append_tool(
        &mut transcript,
        "call_2",
        "read",
        READ_ARGS,
        "current source",
    );
    fold_reads(&mut transcript);
    let stored = serde_json::to_string(transcript.as_slice()).unwrap();
    let mut resumed = Transcript::new(serde_json::from_str(&stored).unwrap());
    let revision = resumed.revision();
    fold_reads(&mut resumed);
    assert_eq!(serde_json::to_string(resumed.as_slice()).unwrap(), stored);
    assert_eq!(
        resumed.revision(),
        revision,
        "idempotent folding rewrote the cached prefix"
    );
}
