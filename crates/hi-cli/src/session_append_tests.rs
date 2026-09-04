use super::*;

#[test]
fn append_after_interrupted_tail_preserves_next_message_and_approval() {
    for tail in [
        b"{\"role\":\"assistant\"".as_slice(),
        b"{\"text\":\"\xf0\x9f",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let original = Message::user("original task");
        let mut bytes = serde_json::to_vec(&original).unwrap();
        bytes.push(b'\n');
        bytes.extend_from_slice(tail);
        fs::write(&path, &bytes).unwrap();

        let mut session = JsonlSession::new(path.clone());
        session.record_plan_approval_parked(true).unwrap();
        let next = Message::user("continue after restart");
        session
            .record(std::slice::from_ref(&next), Usage::default())
            .unwrap();

        let loaded = load_history(&path).unwrap();
        assert!(
            loaded.plan_approval_parked,
            "the first new record must survive a broken tail"
        );
        assert_eq!(
            serde_json::to_value(loaded.messages).unwrap(),
            serde_json::to_value([original, next]).unwrap()
        );
        assert!(
            fs::read(&path).unwrap().starts_with(&bytes),
            "recovery must preserve the original bytes"
        );
    }
}

#[test]
fn append_preserves_complete_record_without_final_newline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let original = Message::user("original task");
    fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    let mut session = JsonlSession::new(path.clone());
    let next = Message::user("next task");
    session
        .record(std::slice::from_ref(&next), Usage::default())
        .unwrap();
    assert_eq!(
        serde_json::to_value(load_history(&path).unwrap().messages).unwrap(),
        serde_json::to_value([original, next]).unwrap()
    );
}

#[test]
fn concurrent_appenders_preserve_large_records_and_settings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    fs::write(&path, b"{\"unfinished\":").unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
    std::thread::scope(|scope| {
        for writer in 0..4 {
            let path = &path;
            let barrier = barrier.clone();
            scope.spawn(move || {
                let mut session = JsonlSession::new(path.clone());
                barrier.wait();
                for record in 0..8 {
                    session
                        .record(
                            &[Message::user(format!(
                                "{writer}:{record} {}",
                                "x".repeat(32 * 1024)
                            ))],
                            Usage::default(),
                        )
                        .unwrap();
                }
            });
        }
        let path = &path;
        scope.spawn(move || {
            barrier.wait();
            for _ in 0..8 {
                crate::session_harness::append(path, &crate::session_harness::empty_layer())
                    .unwrap();
            }
        });
    });
    assert_eq!(load_history(&path).unwrap().messages.len(), 32);
    let records = fs::read_to_string(&path).unwrap();
    assert_eq!(
        records
            .lines()
            .skip(1)
            .filter(|line| serde_json::from_str::<serde_json::Value>(line).is_ok())
            .count(),
        72
    );
}
