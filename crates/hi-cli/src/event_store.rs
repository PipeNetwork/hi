//! Durable local event store and live bus for interactive lifecycle events.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use hi_events::{EventBus, EventError, EventReceipt, EventSink, RunEvent};
use rusqlite::{Connection, OptionalExtension, params};
use tokio::sync::broadcast;

#[derive(Clone)]
pub(crate) struct EventStore {
    connection: Arc<Mutex<Connection>>,
    control: hi_control::ControlStore,
    live: broadcast::Sender<RunEvent>,
    compatibility_activity: Option<PathBuf>,
}

impl EventStore {
    #[allow(dead_code)]
    pub(crate) fn open(path: &std::path::Path) -> Result<Self> {
        Self::open_with_activity(path, None)
    }

    pub(crate) fn open_with_activity(
        path: &std::path::Path,
        compatibility_activity: Option<&Path>,
    ) -> Result<Self> {
        let control = hi_control::ControlStore::open(path)?;
        let connection = hi_sqlite_journal::JournalMode::for_db_path(path).open(path)?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS run_events (
               sequence INTEGER PRIMARY KEY,
               event_id TEXT NOT NULL UNIQUE,
               occurred_at_ms INTEGER NOT NULL,
               event_json TEXT NOT NULL,
               event_bytes INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS run_events_event_id ON run_events(event_id);
             CREATE TABLE IF NOT EXISTS event_dispatch (
               trigger_id TEXT NOT NULL,
               source_event_id TEXT NOT NULL,
               state TEXT NOT NULL,
               created_at_ms INTEGER NOT NULL,
               PRIMARY KEY(trigger_id, source_event_id)
             );
             CREATE TABLE IF NOT EXISTS trigger_checkpoint (
               singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
               sequence INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS trigger_fired (
               trigger_id TEXT NOT NULL,
               concurrency_key TEXT NOT NULL,
               fired_at_ms INTEGER NOT NULL,
               PRIMARY KEY(trigger_id, concurrency_key)
             );",
        )?;
        let (live, _) = broadcast::channel(512);
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            control,
            live,
            compatibility_activity: compatibility_activity.map(Path::to_path_buf),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<RunEvent> {
        self.live.subscribe()
    }

    pub(crate) fn load_since(&self, sequence: u64) -> Result<Vec<RunEvent>> {
        let connection = self.connection.lock().unwrap();
        let mut statement = connection.prepare(
            "SELECT event_json FROM run_events WHERE sequence > ?1 ORDER BY sequence ASC",
        )?;
        let rows = statement.query_map([sequence as i64], |row| row.get::<_, String>(0))?;
        rows.map(|row| {
            let json = row?;
            serde_json::from_str(&json)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
        })
        .collect::<rusqlite::Result<Vec<RunEvent>>>()
        .map_err(Into::into)
    }

    #[allow(dead_code)]
    pub(crate) fn max_sequence(&self) -> Result<u64> {
        let connection = self.connection.lock().unwrap();
        Ok(connection.query_row(
            "SELECT COALESCE(MAX(sequence), 0) FROM run_events",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64)
    }

    pub(crate) fn claim_trigger(&self, trigger_id: &str, event_id: &str) -> Result<bool> {
        let connection = self.connection.lock().unwrap();
        let tx = connection.unchecked_transaction()?;
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO event_dispatch
             (trigger_id, source_event_id, state, created_at_ms)
             VALUES (?1, ?2, 'accepted', ?3)",
            params![trigger_id, event_id, hi_events::now_ms() as i64],
        )?;
        let retried = if inserted == 0 {
            tx.execute(
                "UPDATE event_dispatch SET state = 'accepted', created_at_ms = ?3
                 WHERE trigger_id = ?1 AND source_event_id = ?2 AND state = 'failed'",
                params![trigger_id, event_id, hi_events::now_ms() as i64],
            )?
        } else {
            0
        };
        tx.commit()?;
        Ok(inserted == 1 || retried == 1)
    }
}

impl EventSink for EventStore {
    fn publish(&self, mut event: RunEvent) -> Result<EventReceipt, EventError> {
        let receipt = self
            .control
            .append_event(event.clone())
            .map_err(|error| EventError::Persistence(error.to_string()))?;
        event.sequence = receipt.sequence;

        // Keep the existing, bounded JSONL feed useful for older commands and
        // clients. It is a projection only; the SQLite row above remains the
        // canonical source and contains the full redacted event envelope.
        self.project_compatibility_activity(&event);

        let receipt = EventReceipt {
            event_id: event.event_id.clone(),
            sequence: event.sequence,
        };
        let _ = self.live.send(event);
        Ok(receipt)
    }
}

impl EventBus for EventStore {
    fn replay_since(&self, sequence: u64) -> Result<Vec<RunEvent>, EventError> {
        self.load_since(sequence)
            .map_err(|error| EventError::Persistence(error.to_string()))
    }
}

impl EventStore {
    fn project_compatibility_activity(&self, event: &RunEvent) {
        let Some(path) = &self.compatibility_activity else {
            return;
        };
        // Successful reads are intentionally quiet in the semantic feed.
        if matches!(
            (
                &event.activity.object,
                &event.activity.verb,
                &event.activity.state
            ),
            (
                hi_events::ActivityObject::Tool,
                hi_events::ActivityVerb::Read,
                hi_events::ActivityState::Succeeded
            )
        ) {
            return;
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let detail = event
            .activity
            .detail
            .as_deref()
            .filter(|detail| !detail.is_empty())
            .map(|detail| format!("{} — {detail}", event.activity.title))
            .unwrap_or_else(|| event.activity.title.clone());
        let state = serde_json::to_value(&event.activity.state)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned));
        let entry = serde_json::json!({
            "at_ms": event.occurred_at_ms,
            "loop_id": 0,
            "source": event.activity.group_key,
            "text": detail,
            "event_id": event.event_id,
            "group_key": event.activity.group_key,
            "state": state,
            "detail": event.activity.detail,
        });
        if let Ok(line) = serde_json::to_string(&entry) {
            let mut replaced = false;
            if let Ok(content) = std::fs::read_to_string(path) {
                let mut lines = content.lines().map(str::to_owned).collect::<Vec<_>>();
                for previous in lines.iter_mut().rev() {
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(previous) else {
                        continue;
                    };
                    if value.get("group_key") == entry.get("group_key") {
                        *previous = line.clone();
                        replaced = true;
                        break;
                    }
                }
                if replaced {
                    let _ = std::fs::write(path, lines.join("\n") + "\n");
                }
            }
            if !replaced
                && let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
        if std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0) > 256 * 1024
            && let Ok(content) = std::fs::read_to_string(path)
        {
            let mut lines = content.lines().collect::<Vec<_>>();
            if lines.len() > 500 {
                lines.drain(..lines.len() - 500);
                let body = lines.join("\n") + "\n";
                let _ = std::fs::write(path, body);
            }
        }
    }
}

impl hi_workflow::TriggerLedger for EventStore {
    fn claim(&self, trigger_id: &str, source_event_id: &str) -> Result<bool, String> {
        self.claim_trigger(trigger_id, source_event_id)
            .map_err(|error| error.to_string())
    }

    fn high_watermark(&self) -> Result<u64, String> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| "event store lock poisoned".to_string())?;
        connection
            .query_row(
                "SELECT COALESCE((SELECT sequence FROM trigger_checkpoint WHERE singleton = 1), 0)",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|sequence| sequence as u64)
            .map_err(|error| error.to_string())
    }

    fn set_high_watermark(&self, sequence: u64) -> Result<(), String> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| "event store lock poisoned".to_string())?;
        connection
            .execute(
                "INSERT INTO trigger_checkpoint(singleton, sequence) VALUES (1, ?1)
                 ON CONFLICT(singleton) DO UPDATE SET sequence = MAX(sequence, excluded.sequence)",
                [sequence as i64],
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn last_fired(&self, trigger_id: &str, key: &str) -> Result<Option<u64>, String> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| "event store lock poisoned".to_string())?;
        connection
            .query_row(
                "SELECT fired_at_ms FROM trigger_fired WHERE trigger_id = ?1 AND concurrency_key = ?2",
                params![trigger_id, key],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map(|value| value.map(|at| at as u64))
            .map_err(|error| error.to_string())
    }

    fn record_fired(&self, trigger_id: &str, key: &str, at_ms: u64) -> Result<(), String> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| "event store lock poisoned".to_string())?;
        connection
            .execute(
                "INSERT INTO trigger_fired(trigger_id, concurrency_key, fired_at_ms)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(trigger_id, concurrency_key) DO UPDATE SET fired_at_ms = excluded.fired_at_ms",
                params![trigger_id, key, at_ms as i64],
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn mark_failed(&self, trigger_id: &str, source_event_id: &str) -> Result<(), String> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| "event store lock poisoned".to_string())?;
        connection
            .execute(
                "UPDATE event_dispatch SET state = 'failed'
                 WHERE trigger_id = ?1 AND source_event_id = ?2",
                params![trigger_id, source_event_id],
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

pub(crate) fn state_event_path(state_root: &std::path::Path) -> std::path::PathBuf {
    state_root.join("events.sqlite3")
}

#[allow(dead_code)]
pub(crate) fn publish_best_effort(sink: Option<&dyn EventSink>, event: RunEvent) -> Result<()> {
    let Some(sink) = sink else { return Ok(()) };
    let result = sink.publish(event.clone());
    if event.durability == hi_events::EventDurability::Required {
        result
            .map(|_| ())
            .map_err(|error| anyhow!(error.to_string()))
    } else {
        let _ = result;
        Ok(())
    }
}

pub(crate) fn open_for_state(
    state_root: &std::path::Path,
    compatibility_activity: Option<&Path>,
) -> Result<EventStore> {
    let _ = hi_control::ControlStore::open_for_state(state_root)?;
    EventStore::open_with_activity(&state_event_path(state_root), compatibility_activity)
        .with_context(|| format!("opening event store under {}", state_root.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_events::{
        ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, SemanticActivity,
    };

    fn event() -> RunEvent {
        RunEvent::new(
            EventKind::RunStarted,
            EventContext::default(),
            SemanticActivity {
                verb: ActivityVerb::Start,
                object: ActivityObject::Run,
                state: ActivityState::Running,
                group_key: "run:test".into(),
                title: "test".into(),
                detail: None,
                refs: vec![],
                progress: None,
            },
        )
    }

    #[test]
    fn append_is_idempotent_and_sequences_events() {
        let dir = tempfile::tempdir().unwrap();
        let store = EventStore::open(&dir.path().join("events.sqlite3")).unwrap();
        let first = event();
        let a = store.publish(first.clone()).unwrap();
        let b = store.publish(first).unwrap();
        assert_eq!(a, b);
        let c = store.publish(event()).unwrap();
        assert_eq!(c.sequence, 2);
        assert_eq!(store.load_since(0).unwrap().len(), 2);
    }

    #[test]
    fn compatibility_projection_is_redacted_and_trigger_ledger_is_durable() {
        let dir = tempfile::tempdir().unwrap();
        let activity = dir.path().join("activity.jsonl");
        let store =
            EventStore::open_with_activity(&dir.path().join("events.sqlite3"), Some(&activity))
                .unwrap();
        let mut event = event();
        event.activity.detail = Some("safe summary".into());
        event
            .payload
            .fields
            .insert("status".into(), serde_json::Value::String("ok".into()));
        store.publish(event.clone()).unwrap();
        let text = std::fs::read_to_string(&activity).unwrap();
        assert!(text.contains("safe summary"));
        assert!(!text.contains("arguments"));

        let ledger: &dyn hi_workflow::TriggerLedger = &store;
        assert!(ledger.claim("t", &event.event_id).unwrap());
        assert!(!ledger.claim("t", &event.event_id).unwrap());
        ledger.set_high_watermark(7).unwrap();
        assert_eq!(ledger.high_watermark().unwrap(), 7);
        ledger.record_fired("t", "workspace", 42).unwrap();
        assert_eq!(ledger.last_fired("t", "workspace").unwrap(), Some(42));
    }
}
