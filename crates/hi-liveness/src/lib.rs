//! Heartbeat schema and OS-thread writer for Hi Sentinel.
//!
//! This crate is the child-side liveness surface. The supervisor (`hi-sentinel`)
//! only reads the files this crate writes. It is not the RSI launcher
//! (`hi-bootstrap`) and is not clap's "unlimited internal sentinel" cap.

mod atomic;
mod events;
mod invariants;
mod publisher;
mod schema;
mod writer;

use std::sync::OnceLock;

pub use events::EventLog;
pub use invariants::report as report_invariant;
pub use publisher::Publisher;
pub use schema::{
    AUTO_REPAIR_SET, ENV_CRASH_DIR, ENV_EVENTS, ENV_GENERATION, ENV_HEARTBEAT, ENV_HI_BINARY,
    ENV_INSTANCE, ENV_PANIC_FILE, ENV_RESUME_INCOMPLETE, ENV_ROLE, ENV_SUPERVISED, ENV_TURN_INTENT,
    EVENT_LOG_CAP, EventCode, HEARTBEAT_PERIOD_MS, HarnessState, Heartbeat,
    IDENTICAL_TOOL_CONSECUTIVE, IDENTICAL_TOOL_ERROR_REPEAT, IDENTICAL_TOOL_IN_TURN, InvariantCode,
    InvariantViolation, LiveEvent, SCHEMA_VERSION, TurnIntent, env_flag_on, unix_ms,
    writer_may_publish,
};
pub use writer::{WriterConfig, WriterHandle, read_heartbeat, spawn};

use crate::atomic::write_atomic_json;

static INSTALLED: OnceLock<Publisher> = OnceLock::new();

/// Remember the publisher the supervised writer thread copies from.
pub fn install_publisher(publisher: Publisher) {
    let _ = INSTALLED.set(publisher);
}

pub fn installed_publisher() -> Option<Publisher> {
    INSTALLED.get().cloned()
}

/// Start the heartbeat thread when `HI_SENTINEL_SUPERVISED=1` and a heartbeat path is set.
pub fn spawn_from_env() -> Option<WriterHandle> {
    let config = WriterConfig::from_env()?;
    let publisher = Publisher::new();
    install_publisher(publisher.clone());
    spawn(config, publisher).ok()
}

pub fn set_state_if_installed(state: HarnessState) {
    if let Some(publisher) = installed_publisher() {
        publisher.set_state(state);
    }
}

pub fn write_turn_intent(path: &std::path::Path, intent: &TurnIntent) -> std::io::Result<()> {
    write_atomic_json(path, intent)
}

pub fn write_turn_intent_from_env(intent: &TurnIntent) -> std::io::Result<()> {
    let Some(path) = std::env::var_os(ENV_TURN_INTENT) else {
        return Ok(());
    };
    write_turn_intent(std::path::Path::new(&path), intent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Duration;

    #[test]
    fn fake_snapshot_records_progress_and_state() {
        let publisher = Publisher::new();
        publisher.set_state(HarnessState::AwaitingUser);
        publisher.note_progress();
        publisher.set_workspace("/tmp/ws");
        let snap = publisher.snapshot();
        assert_eq!(snap.state, HarnessState::AwaitingUser);
        assert!(snap.last_progress_unix_ms > 0);
        assert_eq!(snap.workspace, "/tmp/ws");
        assert!(snap.invariant.is_none());
    }

    #[test]
    fn tool_unclosed_sets_invariant_without_panicking() {
        let publisher = Publisher::new();
        publisher.note_tool_start("call_1", "bash");
        publisher.note_tool_start("call_2", "bash");
        let snap = publisher.snapshot();
        let invariant = snap.invariant.expect("tool_unclosed");
        assert_eq!(invariant.code, InvariantCode::ToolUnclosed);
        publisher.set_invariant(InvariantCode::ChildLeak);
        assert_eq!(
            publisher.snapshot().invariant.unwrap().code,
            InvariantCode::ToolUnclosed,
            "invariants are sticky"
        );
    }

    #[test]
    fn writer_seq_advances_and_binds_pid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heartbeat.json");
        let publisher = Publisher::new();
        publisher.set_state(HarnessState::Idle);
        let handle = spawn(
            WriterConfig {
                heartbeat_path: path.clone(),
                events_path: Some(dir.path().join("events.jsonl")),
                instance: "test-instance".into(),
                generation: 0,
                workspace: dir.path().display().to_string(),
                session_path: None,
                period: Duration::from_millis(20),
            },
            publisher,
        )
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let mut beat = None;
        while std::time::Instant::now() < deadline {
            if let Ok(parsed) = read_heartbeat(&path)
                && parsed.seq >= 2
            {
                beat = Some(parsed);
                break;
            }
            std::thread::sleep(Duration::from_millis(15));
        }
        let beat = beat.expect("heartbeat seq should advance");
        assert!(beat.seq >= 2);
        assert_eq!(beat.pid, std::process::id());
        assert_eq!(beat.instance, "test-instance");
        assert_eq!(beat.schema_version, SCHEMA_VERSION);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        drop(handle);
        let last = read_heartbeat(&path).unwrap();
        assert_eq!(last.state, HarnessState::ShuttingDown);
    }

    #[test]
    fn writer_skips_publish_when_pid_mismatches() {
        assert!(!writer_may_publish(0, "token"));
        assert!(writer_may_publish(std::process::id(), "token"));
    }

    #[test]
    fn event_log_rotates_at_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let log = EventLog::with_cap(&path, 80);
        for i in 0..40 {
            log.append(&LiveEvent {
                ts_unix_ms: i,
                code: EventCode::Progress,
                state: HarnessState::Idle,
                tool: None,
                tool_id: None,
                detail: Some("x".repeat(8)),
            });
        }
        assert!(path.exists());
        let rotated = {
            let mut p = path.as_os_str().to_os_string();
            p.push(".1");
            std::path::PathBuf::from(p)
        };
        assert!(rotated.exists(), "2 MiB (test cap) rotation should fire");
    }

    #[test]
    fn turn_intent_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("turn-intent.json");
        write_turn_intent(
            &path,
            &TurnIntent {
                schema_version: SCHEMA_VERSION,
                turn_index: 1,
                prompt: "fix the parser".into(),
                session_path: None,
                pre_checkpoint: Some("abc".into()),
                started_unix_ms: 1,
                workspace: "/tmp/ws".into(),
                oneshot: true,
                plain: false,
            },
        )
        .unwrap();
        let parsed: TurnIntent = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(parsed.prompt, "fix the parser");
        assert!(parsed.oneshot);
    }
}
