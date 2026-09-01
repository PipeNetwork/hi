use super::common::*;
use super::*;

#[tokio::test]
async fn update_memory_writes_file_without_polluting_history() {
    // Use a unique subdir so the per-directory memory lock doesn't collide
    // with other parallel tests writing into the shared temp root.
    let dir = std::env::temp_dir().join(format!(
        "hi-mem-write-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("memory.md");
    let _ = std::fs::remove_file(&path);
    // The model returns a distilled bullet list.
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "- always run cargo fmt\n- tests live in tests/".into(),
            )],
            7,
            4,
        )],
        config(),
    );
    agent
        .messages_mut()
        .push(Message::user("Actually, always run cargo fmt"));
    let before = agent.messages().len();
    agent.update_memory_at(path.clone(), &mut NullUi).await;

    let written = std::fs::read_to_string(&path).expect("memory file written");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        written.contains("always run cargo fmt"),
        "distilled: {written}"
    );
    assert_eq!(
        agent.messages().len(),
        before,
        "session history not polluted"
    );
    assert_eq!(agent.totals().output_tokens, 4, "usage counted");
}

#[tokio::test]
async fn update_memory_persists_usage_without_new_messages() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "hi-memory-persist-{}-{}.md",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(
        vec![completion(vec![Content::Text("- note".into())], 10, 5)],
        config(),
    );
    agent
        .messages_mut()
        .push(Message::user("I prefer a short durable note"));
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));

    agent.update_memory_at(path.clone(), &mut NullUi).await;
    let _ = std::fs::remove_file(path);

    assert_eq!(
        *records.lock().unwrap(),
        vec![Usage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        }]
    );
}

#[tokio::test]
async fn update_memory_is_best_effort_on_error() {
    // A provider error at quit must not panic or leave a file behind.
    let path = std::env::temp_dir().join(format!("hi-mem-{}-err.md", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let (mut agent, _requests) = scripted_agent(
        vec![ProviderStep::Error(ProviderErrorKind::Outage)],
        config(),
    );
    agent
        .messages_mut()
        .push(Message::user("Actually, remember this correction"));
    agent.update_memory_at(path.clone(), &mut NullUi).await;
    assert!(!path.exists(), "nothing written when distillation fails");
}
