use super::*;
use hi_ai::{Content, Message};

fn convo() -> Vec<Message> {
    vec![
        Message::system("sys"),
        Message::user("turn one"),
        Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]),
        Message::tool_result("c1", "x".repeat(500)),
        Message::user("turn two"),
        Message::assistant(vec![Content::Text("answer".into())]),
    ]
}

#[test]
fn turn_starts_and_split() {
    let m = convo();
    assert_eq!(user_turn_starts(&m), vec![1, 4]);
    // Two user turns: keeping 1 splits at the second (index 4).
    assert_eq!(recent_split(&m, 1), Some(4));
    // Keeping ≥ all turns → nothing old to compact.
    assert_eq!(recent_split(&m, 2), None);
    assert_eq!(recent_split(&m, 5), None);
    // Keeping zero turns means everything after the system message is old.
    assert_eq!(recent_split(&m, 0), Some(m.len()));
}

#[test]
fn from_arg_maps_known_kinds() {
    assert_eq!(
        CompactionKind::from_arg("full"),
        Some(CompactionKind::Summarize)
    );
    assert_eq!(
        CompactionKind::from_arg("Hybrid"),
        Some(CompactionKind::Hybrid {
            keep_recent: DEFAULT_KEEP_RECENT
        })
    );
    assert!(matches!(
        CompactionKind::from_arg("elide"),
        Some(CompactionKind::ElideToolOutput { .. })
    ));
    assert_eq!(CompactionKind::from_arg(""), None);
    assert_eq!(CompactionKind::from_arg("bogus"), None);
    assert_eq!(
        CompactionKind::from_arg("window"),
        Some(CompactionKind::FreshWindow)
    );
    assert_eq!(
        CompactionKind::from_arg("fresh"),
        Some(CompactionKind::FreshWindow)
    );
}

#[test]
fn shrink_tool_arguments_stubs_write_payload_and_keeps_path() {
    let args = serde_json::json!({
        "path": "src/lib.rs",
        "content": "fn main() {}\n".repeat(80),
    })
    .to_string();
    let shrunk = shrink_tool_arguments(&args).expect("large write should shrink");
    let value: serde_json::Value = serde_json::from_str(&shrunk).unwrap();
    assert_eq!(value["path"], "src/lib.rs");
    let content = value["content"].as_str().unwrap();
    assert!(content.starts_with("[elided"), "{content}");
    assert!(content.contains("chars"), "{content}");
    assert!(
        shrunk.len() < args.len() / 4,
        "payload must drop: {} vs {}",
        shrunk.len(),
        args.len()
    );
    assert!(shrink_tool_arguments(&shrunk).is_none(), "idempotent");
    assert!(shrink_tool_arguments(r#"{"path":"a.rs","content":"short"}"#).is_none());
}

#[test]
fn elide_shrinks_old_write_arguments_with_results() {
    let mut m = vec![
        Message::system("sys"),
        Message::user("write it"),
        Message::assistant(vec![Content::ToolCall {
            id: "w1".into(),
            name: "write".into(),
            arguments: serde_json::json!({"path":"a.rs","content":"x".repeat(800)}).to_string(),
        }]),
        Message::tool_result("w1", "wrote a.rs"),
    ];
    let len = m.len();
    let freed = elide_tool_outputs(&mut m, len);
    assert!(freed > 0, "should reclaim write payload");
    let Content::ToolCall { arguments, .. } = &m[2].content[0] else {
        panic!("expected tool call");
    };
    assert!(arguments.contains("a.rs"), "{arguments}");
    assert!(arguments.contains("[elided"), "{arguments}");
    assert!(!arguments.contains(&"x".repeat(800)), "{arguments}");
}

#[test]
fn in_turn_elide_keeps_recent_write_arguments() {
    let recent_args = serde_json::json!({
        "path": "new.rs",
        "content": "y".repeat(800),
    })
    .to_string();
    let mut m = vec![Message::system("sys"), Message::user("q")];
    m.push(Message::assistant(vec![Content::ToolCall {
        id: "old".into(),
        name: "write".into(),
        arguments: serde_json::json!({"path":"old.rs","content":"x".repeat(800)}).to_string(),
    }]));
    m.push(Message::tool_result("old", "wrote old.rs"));
    m.push(Message::assistant(vec![Content::ToolCall {
        id: "new".into(),
        name: "write".into(),
        arguments: recent_args.clone(),
    }]));
    m.push(Message::tool_result("new", "wrote new.rs"));
    let freed = elide_tool_outputs_except_recent(&mut m, 1);
    assert!(freed > 0);
    let Content::ToolCall { arguments: old, .. } = &m[2].content[0] else {
        panic!("old call");
    };
    let Content::ToolCall { arguments: new, .. } = &m[4].content[0] else {
        panic!("new call");
    };
    assert!(old.contains("[elided"), "{old}");
    assert_eq!(new, &recent_args, "newest write must stay quoteable");
}

fn user_with_image(text: &str, data: String) -> Message {
    Message {
        role: Role::User,
        content: vec![
            Content::Text(text.into()),
            Content::Image {
                data,
                media_type: "image/png".into(),
            },
        ],
    }
}

#[test]
fn elide_stubs_old_images_and_keeps_recent() {
    let mut m = vec![
        Message::system("sys"),
        user_with_image("old shot", "A".repeat(2_000)),
        Message::assistant(vec![Content::Text("ok".into())]),
        user_with_image("new shot", "B".repeat(400)),
        Message::assistant(vec![Content::Text("done".into())]),
    ];
    let split = recent_split(&m, 1).unwrap();
    let freed = elide_tool_outputs(&mut m, split);
    assert!(freed > 0);
    assert!(
        matches!(&m[1].content[1], Content::Text(text) if text.starts_with("[elided image")),
        "old image stubbed: {:?}",
        m[1].content[1]
    );
    assert!(
        matches!(&m[3].content[1], Content::Image { data, .. } if data == &"B".repeat(400)),
        "recent image stays"
    );
    assert_eq!(elide_tool_outputs(&mut m, split), 0, "idempotent");
}

#[test]
fn elide_stubs_old_thinking_and_keeps_recent() {
    let mut m = vec![
        Message::system("sys"),
        Message::user("q1"),
        Message::assistant(vec![Content::Thinking {
            text: "T".repeat(800),
            signature: Some("sig-old".into()),
        }]),
        Message::user("q2"),
        Message::assistant(vec![
            Content::Thinking {
                text: "U".repeat(800),
                signature: Some("sig-new".into()),
            },
            Content::ToolCall {
                id: "c-new".into(),
                name: "read".into(),
                arguments: "{}".into(),
            },
        ]),
        Message::tool_result("c-new", "ok"),
    ];
    let split = recent_split(&m, 1).unwrap();
    let freed = elide_tool_outputs(&mut m, split);
    assert!(freed > 0);
    let Content::Text(text) = &m[2].content[0] else {
        panic!("old thinking");
    };
    assert!(text.starts_with("[elided thinking"), "{text}");
    let Content::Thinking { text, .. } = &m[4].content[0] else {
        panic!("new thinking");
    };
    assert_eq!(text, &"U".repeat(800), "recent thinking stays");
}

#[test]
fn elide_shrinks_old_outputs_only_and_is_idempotent() {
    let mut m = convo();
    // keep_recent = 1 → "turn two" is recent; c1's output (in turn one) is old.
    let split = recent_split(&m, 1).unwrap();
    let freed = elide_tool_outputs(&mut m, split);
    assert!(freed >= 500, "reclaimed the big output: {freed}");

    let outputs: Vec<String> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert!(
        outputs[0].starts_with(ELIDED_MARK),
        "old elided: {}",
        outputs[0]
    );
    assert!(
        outputs[0].contains("read"),
        "names the tool: {}",
        outputs[0]
    );

    // Running again frees nothing (idempotent).
    assert_eq!(elide_tool_outputs(&mut m, split), 0);
}

#[test]
fn elide_keeps_small_and_recent_outputs() {
    let mut m = vec![
        Message::system("sys"),
        Message::user("q"),
        Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            arguments: "{}".into(),
        }]),
        Message::tool_result("c1", "tiny"), // below threshold
    ];
    // No recent split (one turn) → caller passes len; small output untouched.
    let len = m.len();
    assert_eq!(elide_tool_outputs(&mut m, len), 0);
}

#[test]
fn in_turn_elide_keeps_newest_tool_results() {
    let mut m = vec![Message::system("sys"), Message::user("q")];
    for i in 1..=4 {
        let id = format!("c{i}");
        m.push(Message::assistant(vec![Content::ToolCall {
            id: id.clone(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
        m.push(Message::tool_result(
            &id,
            format!("{i}\n{}", "x".repeat(500)),
        ));
    }

    let freed = elide_tool_outputs_except_recent(&mut m, 2);
    assert!(freed >= 1000, "reclaimed old tool outputs: {freed}");

    let outputs: Vec<String> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert!(outputs[0].starts_with(ELIDED_MARK), "{outputs:?}");
    assert!(outputs[1].starts_with(ELIDED_MARK), "{outputs:?}");
    assert!(outputs[2].starts_with("3\n"), "{outputs:?}");
    assert!(outputs[3].starts_with("4\n"), "{outputs:?}");
    assert_eq!(elide_tool_outputs_except_recent(&mut m, 2), 0);
}

#[test]
fn in_turn_elide_keeps_newest_read_of_each_path() {
    let bulky = "x".repeat(500);
    let mut m = vec![Message::system("sys"), Message::user("q")];
    m.push(Message::assistant(vec![Content::ToolCall {
        id: "html".into(),
        name: "read".into(),
        arguments: r#"{"path":"src/web/index.html"}"#.into(),
    }]));
    m.push(Message::tool_result(
        "html",
        format!("<span>offline</span>\n{bulky}"),
    ));
    m.push(Message::assistant(vec![Content::ToolCall {
        id: "rs".into(),
        name: "read".into(),
        arguments: r#"{"path":"src/web.rs"}"#.into(),
    }]));
    m.push(Message::tool_result(
        "rs",
        format!("fn register()\n{bulky}"),
    ));
    for i in 1..=3 {
        let id = format!("g{i}");
        m.push(Message::assistant(vec![Content::ToolCall {
            id: id.clone(),
            name: "grep".into(),
            arguments: format!(r#"{{"pattern":"p{i}"}}"#),
        }]));
        m.push(Message::tool_result(&id, format!("grep-{i}\n{bulky}")));
    }
    let freed = elide_tool_outputs_except_recent(&mut m, 1);
    assert!(freed > 0);
    let outputs: Vec<String> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert!(
        outputs.iter().any(|output| output.contains("offline")),
        "newest index.html read must survive elision: {outputs:?}"
    );
    assert!(
        outputs
            .iter()
            .any(|output| output.contains("fn register()")),
        "newest web.rs read must survive elision: {outputs:?}"
    );
    assert!(
        outputs
            .iter()
            .any(|output| output.starts_with(ELIDED_MARK) && output.contains("grep")),
        "old greps may be stubbed: {outputs:?}"
    );
}

#[test]
fn estimate_counts_outputs_and_args() {
    let m = vec![
        Message::user("a".repeat(40)),             // 10 tokens
        Message::tool_result("c", "b".repeat(40)), // 10 tokens
    ];
    assert_eq!(estimate_tokens(&m), 21);
}

#[test]
fn conversational_tail_excludes_tool_turns() {
    let m = convo(); // system, q1, read call, big result, q2, answer
    let split = recent_split(&m, 1).unwrap(); // q2 onward is recent → split at q2
    // Turn one's assistant reply made a tool call, so it's NOT part of the
    // conversational tail — the tail is empty for this conversation.
    let tail = conversational_tail(&m, split);
    assert!(
        tail.is_empty(),
        "tool turn excluded from Q&A tail: {tail:?}"
    );

    // A conversation with a real Q&A turn: system, q1 + text answer (Q&A),
    // q2 + tool turn (recent).
    let m2 = vec![
        Message::system("sys"),
        Message::user("q1"),
        Message::assistant(vec![Content::Text("a1".into())]), // no tool call → Q&A
        Message::user("q2"),
        Message::assistant(vec![Content::Text("a2".into())]),
    ];
    let split2 = recent_split(&m2, 1).unwrap(); // q2 is recent
    let tail2 = conversational_tail(&m2, split2);
    assert_eq!(tail2.len(), 2, "q1 + a1 form the Q&A tail: {tail2:?}");
    assert_eq!(tail2[0].role, Role::User);
    assert_eq!(tail2[1].role, Role::Assistant);
}

#[test]
fn tool_bearing_turns_preserves_the_complete_tool_turn() {
    let m = vec![
        Message::system("sys"),
        Message::user("q1: inspect file"),
        Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]),
        Message::tool_result("c1", "file contents"),
        Message::assistant(vec![Content::Text("a1: found the issue".into())]),
        Message::user("q2: explain rust ownership"),
        Message::assistant(vec![Content::Text("a2: ownership answer".into())]),
        Message::user("q3: fix it"),
    ];
    let split = recent_split(&m, 1).unwrap();

    let tool_turns = tool_bearing_turns(&m, split);

    assert_eq!(
        tool_turns.iter().map(|msg| msg.role).collect::<Vec<_>>(),
        vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
    assert_eq!(tool_turns[0].text(), "q1: inspect file");
    assert!(matches!(
        &tool_turns[2].content[0],
        Content::ToolResult { output, .. } if output == "file contents"
    ));
    assert_eq!(tool_turns[3].text(), "a1: found the issue");
    assert!(
        tool_turns.iter().all(|msg| !msg.text().contains("q2")),
        "tool-free Q&A turn should be summarized instead: {tool_turns:?}"
    );
}
