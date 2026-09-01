use super::*;

fn temp_store_path(tag: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "hi-sync-{tag}-{}-{nonce}.sqlite",
        std::process::id()
    ))
}

#[test]
fn breaker_trips_with_doubling_cooldown_and_resets_on_success() {
    let path = temp_store_path("breaker");
    let store = SyncStore::open_at(path.clone()).unwrap();
    assert_eq!(store.breaker_open_until().unwrap(), None);

    let now = 1_000_000;
    let first = store.trip_breaker(now).unwrap();
    assert_eq!(first, now + 60, "first trip: 60s cooldown");
    assert_eq!(store.breaker_open_until().unwrap(), Some(first));

    let second = store.trip_breaker(now + 60).unwrap();
    assert_eq!(second, now + 60 + 120, "consecutive trips double");
    let mut last = second;
    for i in 0..6 {
        last = store.trip_breaker(now + 100 + i).unwrap();
    }
    assert!(
        last <= now + 106 + 900,
        "cooldown caps at 15 minutes: {last}"
    );

    store.reset_breaker().unwrap();
    assert_eq!(store.breaker_open_until().unwrap(), None);
    let after_reset = store.trip_breaker(now).unwrap();
    assert_eq!(after_reset, now + 60, "reset also clears the backoff");
    drop(store);
    let _ = std::fs::remove_file(path);
}

#[test]
fn fresh_installs_leave_mode_unset_for_the_provider_default() {
    let path = temp_store_path("mode-unset");
    let store = SyncStore::open_at(path.clone()).unwrap();
    // No legacy opt-in: nothing is persisted, so the provider default
    // (pipenetwork => on) can apply at read time.
    assert_eq!(store.initialize_mode(false).unwrap(), None);
    assert_eq!(store.stored_mode().unwrap(), None);
    // Legacy `[sync] enabled = true` still migrates to a persisted `on`.
    assert_eq!(store.initialize_mode(true).unwrap(), Some(SyncMode::On));
    assert_eq!(store.stored_mode().unwrap(), Some(SyncMode::On));
    drop(store);
    let _ = std::fs::remove_file(path);
}

#[test]
fn heal_deletes_only_the_implicit_off_row() {
    let path = temp_store_path("mode-heal");
    let store = SyncStore::open_at(path.clone()).unwrap();
    // Simulate the poisoned store the old initialize left behind: an
    // `off` row with no user marker.
    store
        .connection
        .lock()
        .unwrap()
        .execute(
            "INSERT INTO sync_settings(key,value) VALUES('mode','off')",
            [],
        )
        .unwrap();
    assert!(store.heal_implicit_off().unwrap());
    assert_eq!(store.stored_mode().unwrap(), None);
    // A second heal is a no-op.
    assert!(!store.heal_implicit_off().unwrap());

    // An off the user chose (set_mode stamps the marker) survives.
    store.set_mode(SyncMode::Off).unwrap();
    assert!(!store.heal_implicit_off().unwrap());
    assert_eq!(store.stored_mode().unwrap(), Some(SyncMode::Off));
    drop(store);
    let _ = std::fs::remove_file(path);
}

#[test]
fn mode_resolution_prefers_override_then_choice_then_default() {
    use super::resolve_mode;
    // Nothing anywhere: off.
    assert_eq!(resolve_mode(None, None, None), SyncMode::Off);
    // Provider default fills an unset store.
    assert_eq!(resolve_mode(None, None, Some(SyncMode::On)), SyncMode::On);
    // The user's persisted choice beats the default…
    assert_eq!(
        resolve_mode(None, Some(SyncMode::Off), Some(SyncMode::On)),
        SyncMode::Off
    );
    // …and the --sync process override beats everything.
    assert_eq!(
        resolve_mode(Some(SyncMode::On), Some(SyncMode::Off), None),
        SyncMode::On
    );
}

#[test]
fn mode_outbox_and_purge_survive_reopen() {
    let path = temp_store_path("mode-outbox");
    let store = SyncStore::open_at(path.clone()).unwrap();
    assert_eq!(store.initialize_mode(false).unwrap(), None);
    store.enqueue_record("s", "message", "{}").unwrap();
    assert!(store.ready_records("s", 10).unwrap().is_empty());
    store.set_mode(SyncMode::Paused).unwrap();
    store.enqueue_record("s", "message", "{}").unwrap();
    drop(store);
    let store = SyncStore::open_at(path.clone()).unwrap();
    assert_eq!(store.stored_mode().unwrap(), Some(SyncMode::Paused));
    assert_eq!(store.ready_records("s", 10).unwrap().len(), 1);
    store.purge().unwrap();
    assert!(store.ready_records("s", 10).unwrap().is_empty());
    drop(store);
    let _ = std::fs::remove_file(path);
}

/// A peer holding the write lock while another process opens the store
/// must delay the open, not fail it — the "database is locked" turn
/// failures came from lock-taking setup running before any busy_timeout
/// was configured.
#[test]
fn open_waits_for_peer_write_lock_instead_of_failing() {
    let path = temp_store_path("open-contended");
    // Seed a pre-WAL database so open_at's journal_mode switch needs an
    // exclusive lock, then hold the write lock from a peer connection.
    let peer = Connection::open(&path).unwrap();
    peer.execute_batch("CREATE TABLE seed(x); BEGIN IMMEDIATE; INSERT INTO seed VALUES(1);")
        .unwrap();
    let opener = {
        let path = path.clone();
        std::thread::spawn(move || SyncStore::open_at(path).map(|_| ()))
    };
    std::thread::sleep(std::time::Duration::from_millis(300));
    peer.execute_batch("COMMIT;").unwrap();
    opener
        .join()
        .unwrap()
        .expect("open should wait out the peer's short write lock");
    drop(peer);
    let _ = std::fs::remove_file(path);
}

#[test]
fn track_jsonl_identifies_files_canonically() {
    let store_path = temp_store_path("track-canonical");
    let store = SyncStore::open_at(store_path.clone()).unwrap();
    store.set_mode(SyncMode::On).unwrap();
    let dir = std::env::temp_dir().join(format!(
        "hi-track-canonical-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s.jsonl");
    std::fs::write(&path, "{}\n{}\n").unwrap();

    store.track_jsonl("canon", &path).unwrap();
    store.set_jsonl_offset("canon", 3).unwrap();
    // A dotted spelling of the same file canonicalizes to the same row —
    // the committed offset is shared, not reset to the file length (6).
    let dotted = dir.join(".").join("s.jsonl");
    assert_eq!(store.track_jsonl("canon", &dotted).unwrap(), 3);

    drop(store);
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_file(store_path);
}

#[test]
fn live_event_drop_delta_counts_only_removed_rows() {
    assert_eq!(live_event_drop_delta(2, 2), 0);
    assert_eq!(live_event_drop_delta(5, 2), 3);
    assert_eq!(live_event_drop_delta(2, 3), 0);
}

#[test]
fn opening_store_resets_polluted_legacy_drop_counts_once() {
    let path = temp_store_path("event-drop-migration");
    let store = SyncStore::open_at(path.clone()).unwrap();
    {
        let connection = store.connection.lock().unwrap();
        connection
            .execute(
                "DELETE FROM sync_settings WHERE key='live_event_drop_formula'",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO session_sync(session_id,event_drops) VALUES('legacy',633)",
                [],
            )
            .unwrap();
    }
    drop(store);

    let store = SyncStore::open_at(path.clone()).unwrap();
    assert_eq!(store.status(Some("legacy")).unwrap().event_drops, 0);
    store
        .connection
        .lock()
        .unwrap()
        .execute(
            "UPDATE session_sync SET event_drops=7 WHERE session_id='legacy'",
            [],
        )
        .unwrap();
    drop(store);

    let reopened = SyncStore::open_at(path.clone()).unwrap();
    assert_eq!(reopened.status(Some("legacy")).unwrap().event_drops, 7);
    drop(reopened);
    let _ = std::fs::remove_file(path);
}

#[test]
fn pending_sessions_list_excludes_the_current_one_and_dead_quarantine() {
    let path = temp_store_path("pending-sessions");
    let store = SyncStore::open_at(path.clone()).unwrap();
    store.set_mode(SyncMode::On).unwrap();
    store.enqueue_record("current", "message", "{}").unwrap();
    store
        .enqueue_record("stranded-ready", "message", "{}")
        .unwrap();
    store
        .enqueue_record("stranded-quarantined", "message", "{}")
        .unwrap();
    store.enqueue_record("given-up", "message", "{}").unwrap();

    // One quarantined after a single attempt still deserves a retry; one
    // that has been retried to death does not.
    let young = store.ready_records("stranded-quarantined", 1).unwrap();
    store
        .fail_records("stranded-quarantined", &young, "HTTP 400", None, true)
        .unwrap();
    let dead = store.ready_records("given-up", 1).unwrap();
    for _ in 0..8 {
        store
            .fail_records("given-up", &dead, "HTTP 400", None, true)
            .unwrap();
    }

    let pending = store.sessions_with_pending_records("current", 10).unwrap();
    assert_eq!(
        pending,
        vec![
            "stranded-ready".to_string(),
            "stranded-quarantined".to_string()
        ]
    );
    assert_eq!(
        store
            .sessions_with_pending_records("current", 1)
            .unwrap()
            .len(),
        1
    );
    drop(store);
    let _ = std::fs::remove_file(path);
}

#[test]
fn timeout_streak_counts_until_reset() {
    let path = temp_store_path("timeout-streak");
    let store = SyncStore::open_at(path.clone()).unwrap();
    assert_eq!(store.note_timeout().unwrap(), 1);
    assert_eq!(store.note_timeout().unwrap(), 2);
    // Any successful round-trip clears the streak with the breaker.
    store.reset_breaker().unwrap();
    assert_eq!(store.note_timeout().unwrap(), 1);
    store
        .note_endpoint_failure("timeout: slow persist")
        .unwrap();
    drop(store);
    let _ = std::fs::remove_file(path);
}
